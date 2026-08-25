// ============================================================================
// GPU InspiRING Offline Precomputation
//
// Replaces the CPU full_packing_with_preprocessing_offline (packing.rs:2247)
// for word SimplePIR. Keeps hint_0 on GPU (no D2H transfer), computes bold_t
// results directly on GPU (no H2D transfer), and is ~100x faster.
//
// Algorithm:
//   Phase 1 (parallel): compute r_all[i] for i=0..1023 via monomial*a_ct_tilde
//                        multiply-adds + automorphism, per output
//   Phase 2 (sequential): backward recursion i=1022..0:
//                          gadget_invert(r_all[i+1]) -> bold_t[i]
//                          r_all[i] += w_all[i] · bold_t[i]
//   Final: bold_t_hat = gadget_invert(r_bar_all[0]),
//          r_all[0] += v_mask · bold_t_hat -> a_hat
// ============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include "common/ntt.cuh"
#include "common/log.cuh"

// ============================================================================
// Context
// ============================================================================

struct SwInspirPrecompContext {
    uint32_t poly_len;         // 2048
    uint32_t crt_count;        // 2
    uint32_t t_exp_left;       // 3
    uint32_t bits_per;         // floor(modulus_log2 / t_exp_left) + 1
    uint32_t num_outputs;
    uint32_t num_to_pack_half; // poly_len / 2
    uint32_t num_iter;         // poly_len / 2 - 1
    uint32_t q2_bits;

    NTTParams ntt_params;

    uint32_t* d_gen_pows;      // [poly_len]
    uint32_t* d_tables;        // [num_tables * poly_len]
    uint32_t num_tables;

    uint64_t* d_monomial_ntts;     // [poly_len * crt_count * poly_len]
    uint64_t* d_neg_monomial_ntts; // same
    uint64_t* d_mod_inv_poly;      // [crt_count * poly_len]
    uint64_t* d_a_ct_tilde;       // [num_outputs * poly_len * crt_count * poly_len]
    uint64_t* d_w_all;            // [num_iter * t_exp_left * crt_count * poly_len]
    uint64_t* d_w_bar_all;        // same
    uint64_t* d_v_mask;           // [t_exp_left * crt_count * poly_len]
    uint64_t* d_r_all;            // [num_outputs * num_to_pack_half * crt_count * poly_len]
    uint64_t* d_r_bar_all;        // same

    // Outputs (condensed, stay on GPU)
    uint64_t* d_bold_t_condensed;      // [num_outputs * num_iter * t_exp_left * poly_len]
    uint64_t* d_bold_t_bar_condensed;  // same
    uint64_t* d_bold_t_hat_condensed;  // [num_outputs * t_exp_left * poly_len]
    uint64_t* d_a_hat;                 // [num_outputs * poly_len]

    const uint64_t* d_hint_0;  // borrowed
    uint32_t db_cols;

    cudaStream_t stream;
};

// ============================================================================
// Kernel 1: Compute Monomial NTTs
// Each block computes NTT(X^j) and NTT(-X^j) for one j.
// Block: 1024 threads, Grid: poly_len, Smem: crt_count * poly_len u64
// ============================================================================

__global__ void sw_compute_monomials(
    uint64_t* __restrict__ d_monomial_ntts,
    uint64_t* __restrict__ d_neg_monomial_ntts,
    NTTParams params)
{
    uint32_t j = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t poly_len = params.poly_len;
    uint32_t crt_count = params.crt_count;
    uint32_t cpn = crt_count * poly_len;

    extern __shared__ uint64_t smem[];

    // Build raw monomial e_j per CRT modulus (value is 0 or 1, same for all moduli)
    for (uint32_t m = 0; m < crt_count; m++) {
        for (uint32_t k = tid; k < poly_len; k += blockDim.x) {
            smem[m * poly_len + k] = (k == j) ? 1ULL : 0ULL;
        }
    }
    __syncthreads();

    for (uint32_t m = 0; m < crt_count; m++) {
        ntt_forward_alt_parallel(smem + m * poly_len, &params, m, tid, blockDim.x);
    }

    uint64_t* dst = d_monomial_ntts + (size_t)j * cpn;
    for (uint32_t k = tid; k < cpn; k += blockDim.x) {
        dst[k] = smem[k];
    }

    // Negate: -X^j in NTT domain = modulus - coeff for each CRT
    for (uint32_t m = 0; m < crt_count; m++) {
        uint64_t mod_m = params.moduli[m];
        for (uint32_t k = tid; k < poly_len; k += blockDim.x) {
            uint64_t val = smem[m * poly_len + k];
            smem[m * poly_len + k] = (val == 0) ? 0 : (mod_m - val);
        }
    }
    __syncthreads();

    uint64_t* dst_neg = d_neg_monomial_ntts + (size_t)j * cpn;
    for (uint32_t k = tid; k < cpn; k += blockDim.x) {
        dst_neg[k] = smem[k];
    }
}

// ============================================================================
// Kernel 2: Prep Pack LWEs
// Converts hint_0 columns into a_ct_tilde NTT polynomials.
// Each block: one (output, column_j) pair.
// Block: 1024 threads, Grid: (poly_len, num_outputs), Smem: cpn u64
// ============================================================================

__global__ void sw_prep_pack_lwes(
    uint64_t* __restrict__ d_a_ct_tilde,
    const uint64_t* __restrict__ d_hint_0,
    uint32_t db_cols,
    uint64_t modulus_Q,
    NTTParams params)
{
    uint32_t col_j = blockIdx.x;
    uint32_t output = blockIdx.y;
    uint32_t tid = threadIdx.x;
    uint32_t poly_len = params.poly_len;
    uint32_t crt_count = params.crt_count;
    uint32_t cpn = crt_count * poly_len;

    extern __shared__ uint64_t smem[];

    // Extract column, negacyclic_perm, CRT decompose — all in one pass.
    // negacyclic_perm(shift=0): out[0]=a[0]; out[k]=Q-a[N-k] for k>0
    uint32_t col_in_hint = output * poly_len + col_j;
    for (uint32_t k = tid; k < poly_len; k += blockDim.x) {
        uint64_t val;
        if (k == 0) {
            val = d_hint_0[col_in_hint];
        } else {
            uint64_t raw = d_hint_0[(size_t)(poly_len - k) * db_cols + col_in_hint];
            val = (raw == 0) ? 0 : (modulus_Q - raw);
        }
        for (uint32_t m = 0; m < crt_count; m++) {
            smem[m * poly_len + k] = val % params.moduli[m];
        }
    }
    __syncthreads();

    for (uint32_t m = 0; m < crt_count; m++) {
        ntt_forward_alt_parallel(smem + m * poly_len, &params, m, tid, blockDim.x);
    }

    uint64_t* dst = d_a_ct_tilde + ((size_t)output * poly_len + col_j) * cpn;
    for (uint32_t k = tid; k < cpn; k += blockDim.x) {
        dst[k] = smem[k];
    }
}

// ============================================================================
// Kernel 3: Generate W Rotations
// Applies automorphisms to w_mask to produce w_all and w_bar_all.
// Block: 1024 threads, Grid: ceil(num_iter / ROTS_PER_BLK)
// ============================================================================

#define ROTS_PER_BLK 8

__global__ void sw_generate_w_rotations(
    uint64_t* __restrict__ d_w_all,
    uint64_t* __restrict__ d_w_bar_all,
    const uint64_t* __restrict__ d_w_mask,
    const uint32_t* __restrict__ d_tables,
    const uint32_t* __restrict__ d_gen_pows,
    uint32_t t_exp_left,
    uint32_t poly_len,
    uint32_t crt_count,
    uint32_t num_iter)
{
    extern __shared__ uint64_t s_wmask[];
    uint32_t rot_base = blockIdx.x * ROTS_PER_BLK;
    uint32_t cpn = crt_count * poly_len;

    for (uint32_t k = 0; k < t_exp_left; k++) {
        for (uint32_t m = 0; m < crt_count; m++) {
            const uint64_t* src = d_w_mask + k * cpn + m * poly_len;
            for (uint32_t z = threadIdx.x; z < poly_len; z += blockDim.x)
                s_wmask[z] = src[z];
            __syncthreads();

            for (uint32_t r = 0; r < ROTS_PER_BLK; r++) {
                uint32_t rot = rot_base + r;
                if (rot >= num_iter) break;

                uint32_t t = d_gen_pows[rot];
                uint32_t tidx1 = (t - 1) / 2;
                uint32_t tidx2 = (2 * poly_len - t - 1) / 2;
                const uint32_t* tab1 = d_tables + (size_t)tidx1 * poly_len;
                const uint32_t* tab2 = d_tables + (size_t)tidx2 * poly_len;

                uint64_t* dst1 = d_w_all + ((size_t)rot * t_exp_left + k) * cpn + m * poly_len;
                uint64_t* dst2 = d_w_bar_all + ((size_t)rot * t_exp_left + k) * cpn + m * poly_len;

                for (uint32_t z = threadIdx.x; z < poly_len; z += blockDim.x) {
                    dst1[z] = s_wmask[__ldg(&tab1[z])];
                    dst2[z] = s_wmask[__ldg(&tab2[z])];
                }
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// Kernel 4a: Compute r_all — Tiled (Phase 1, CRT=1 specialization)
//
// Tiles TILE_I rotation indices per block, reading a_ct_tilde ONCE per j
// across all i values in the tile. Uses running products α^j instead of
// monomial table lookups (saves ~8x bandwidth).
//
// Grid: (ceil(num_to_pack_half/TILE_I), num_outputs), Block: 1024
// Smem: poly_len u64 (for automorphism gather)
// ============================================================================

// Uses running products α^j instead of per-j monomial table reads.
// Same grid as original (one block per (i, output)) but eliminates
// ~half the global memory bandwidth (no monomial reads in inner loop).
// Register budget: ~60 32-bit regs at 1024 threads (fits in 64 max).

// 128-bit accumulators need SM 7.5+ register budget.
// On older GPUs, the launch will fail with cudaErrorLaunchOutOfResources;
// the caller falls back to sw_compute_r_all.
__global__ void sw_compute_r_all_fast(
    uint64_t* __restrict__ d_r_all,
    uint64_t* __restrict__ d_r_bar_all,
    const uint64_t* __restrict__ d_monomial_ntts,
    const uint64_t* __restrict__ d_neg_monomial_ntts,
    const uint64_t* __restrict__ d_a_ct_tilde,
    const uint64_t* __restrict__ d_mod_inv_poly,
    const uint32_t* __restrict__ d_tables,
    const uint32_t* __restrict__ d_gen_pows,
    NTTParams params)
{
    uint32_t i = blockIdx.x;
    uint32_t output = blockIdx.y;
    uint32_t tid = threadIdx.x;
    uint32_t poly_len = params.poly_len;
    uint32_t cpn = poly_len;  // crt_count=1
    uint32_t num_to_pack_half = poly_len / 2;

    extern __shared__ uint64_t smem[];
    const uint64_t* a_ct_base = d_a_ct_tilde + (size_t)output * poly_len * cpn;
    uint64_t mod0 = params.moduli[0];
    uint64_t cr0 = params.barrett_cr[0];

    // Load running-product base: α = NTT(X^{gp_val})[z]
    // α^j = NTT(X^{j·gp_val mod 2d})[z] (negacyclic NTT handles wrapping)
    uint32_t gp_val = __ldg(&d_gen_pows[(poly_len - i) % poly_len]);

    const uint64_t* src_r = (gp_val < poly_len)
        ? d_monomial_ntts + (size_t)gp_val * cpn
        : d_neg_monomial_ntts + (size_t)(gp_val - poly_len) * cpn;
    uint64_t alpha_r0 = __ldg(&src_r[tid]);
    uint64_t alpha_r1 = __ldg(&src_r[tid + 1024]);

    uint32_t gp_bar = (2 * poly_len - gp_val) % (2 * poly_len);
    const uint64_t* src_rb = (gp_bar < poly_len)
        ? d_monomial_ntts + (size_t)gp_bar * cpn
        : d_neg_monomial_ntts + (size_t)(gp_bar - poly_len) * cpn;
    uint64_t alpha_rb0 = __ldg(&src_rb[tid]);
    uint64_t alpha_rb1 = __ldg(&src_rb[tid + 1024]);

    // Running products (start at 1 = α^0) and 128-bit accumulators.
    // Products m*a are < Q^2 < 2^64, so we skip Barrett on products
    // and accumulate raw u64 values in __uint128_t.
    // After 2048 iterations: sum < 2048 * 2^64 < 2^75, fits in 128 bits.
    // This eliminates 4 of the 8 Barrett reductions per iteration.
    uint64_t m_r0 = 1, m_r1 = 1, m_rb0 = 1, m_rb1 = 1;
    unsigned __int128 acc_r0 = 0, acc_r1 = 0, acc_rb0 = 0, acc_rb1 = 0;

    for (uint32_t j = 0; j < poly_len; j++) {
        const uint64_t* a_ct_j = a_ct_base + (size_t)j * cpn;
        uint64_t a0 = __ldg(&a_ct_j[tid]);
        uint64_t a1 = __ldg(&a_ct_j[tid + 1024]);

        // Accumulate raw products (no Barrett — 128-bit handles overflow)
        acc_r0  += (unsigned __int128)(m_r0  * a0);
        acc_r1  += (unsigned __int128)(m_r1  * a1);
        acc_rb0 += (unsigned __int128)(m_rb0 * a0);
        acc_rb1 += (unsigned __int128)(m_rb1 * a1);

        // Running product updates (must Barrett to keep m < Q for next multiply)
        m_r0  = barrett_raw_u64(m_r0  * alpha_r0,  cr0, mod0);
        m_r1  = barrett_raw_u64(m_r1  * alpha_r1,  cr0, mod0);
        m_rb0 = barrett_raw_u64(m_rb0 * alpha_rb0, cr0, mod0);
        m_rb1 = barrett_raw_u64(m_rb1 * alpha_rb1, cr0, mod0);
    }

    // Reduce 128-bit accumulators mod Q.
    // acc < 2048 * Q^2 < 2^75. Since Q < 2^32, acc/Q < 2^43, fits in u64.
    // Use: result = (uint64_t)(acc - (acc/Q)*Q) where acc/Q is computed via shifts.
    // Simpler: repeated Barrett on lo/hi halves.
    auto reduce128 = [&](unsigned __int128 v) -> uint64_t {
        // v < 2^75. Split: v = hi * 2^64 + lo. hi < 2^11.
        // v mod Q = (hi * (2^64 mod Q) + lo) mod Q
        // 2^64 mod Q = 2^64 - floor(2^64/Q)*Q. Since Q ≈ 2^32: 2^64 mod Q ≈ 2^32 * (2^32 - Q) ≈ small
        uint64_t lo = (uint64_t)v;
        uint64_t hi = (uint64_t)(v >> 64);
        // 2^64 mod Q: compute once
        // Q = 4294955009, 2^64 / Q ≈ 4294979584.0007
        // 2^64 mod Q = 2^64 - 4294979584 * Q ... hard without 128-bit on device.
        // Just use the % operator but move it outside the hot loop (it's called once per thread)
        return (uint64_t)(v % mod0);
    };
    uint64_t r0  = reduce128(acc_r0);
    uint64_t r1  = reduce128(acc_r1);
    uint64_t rb0 = reduce128(acc_rb0);
    uint64_t rb1 = reduce128(acc_rb1);

    uint64_t inv0 = __ldg(&d_mod_inv_poly[tid]);
    uint64_t inv1 = __ldg(&d_mod_inv_poly[tid + 1024]);
    r0  = barrett_raw_u64(r0  * inv0, cr0, mod0);
    r1  = barrett_raw_u64(r1  * inv1, cr0, mod0);
    rb0 = barrett_raw_u64(rb0 * inv0, cr0, mod0);
    rb1 = barrett_raw_u64(rb1 * inv1, cr0, mod0);

    // r automorphism via shared memory gather
    smem[tid] = r0;
    smem[tid + 1024] = r1;
    __syncthreads();

    uint32_t t_val = __ldg(&d_gen_pows[i]);
    const uint32_t* tab1 = d_tables + (size_t)((t_val - 1) / 2) * poly_len;
    size_t r_off = ((size_t)output * num_to_pack_half + i) * cpn;
    d_r_all[r_off + tid]        = smem[__ldg(&tab1[tid])];
    d_r_all[r_off + tid + 1024] = smem[__ldg(&tab1[tid + 1024])];
    __syncthreads();

    // r_bar automorphism
    smem[tid] = rb0;
    smem[tid + 1024] = rb1;
    __syncthreads();

    uint32_t t_val2 = 2 * poly_len - t_val;
    const uint32_t* tab2 = d_tables + (size_t)((t_val2 - 1) / 2) * poly_len;
    d_r_bar_all[r_off + tid]        = smem[__ldg(&tab2[tid])];
    d_r_bar_all[r_off + tid + 1024] = smem[__ldg(&tab2[tid + 1024])];
}
// ============================================================================
// Kernel 4b: Compute r_all — Original (Phase 1, general CRT)
//
// Each thread handles 2 coefficient positions (tid, tid+1024) across all CRT
// moduli. Accumulators stay in registers: 2 pos × 2 CRT × 2 paths = 8 regs.
// Grid: (num_to_pack_half, num_outputs), Block: 1024
// Smem: cpn u64 (for automorphism gather)
// ============================================================================

__launch_bounds__(1024, 2)  // Force 32 regs/thread max — ensures launch on SM 6.x even with __int128 in same TU
__global__ void sw_compute_r_all(
    uint64_t* __restrict__ d_r_all,
    uint64_t* __restrict__ d_r_bar_all,
    const uint64_t* __restrict__ d_monomial_ntts,
    const uint64_t* __restrict__ d_neg_monomial_ntts,
    const uint64_t* __restrict__ d_a_ct_tilde,
    const uint64_t* __restrict__ d_mod_inv_poly,
    const uint32_t* __restrict__ d_tables,
    const uint32_t* __restrict__ d_gen_pows,
    NTTParams params,
    uint32_t q2_bits)
{
    uint32_t i = blockIdx.x;
    uint32_t output = blockIdx.y;
    uint32_t tid = threadIdx.x;
    uint32_t poly_len = params.poly_len;
    uint32_t crt_count = params.crt_count;
    uint32_t cpn = crt_count * poly_len;

    extern __shared__ uint64_t smem[];

    uint32_t gp_val = d_gen_pows[(poly_len - i) % poly_len];
    // For single CRT (crt_count=1), always reduce every step (accumulator can overflow u64).
    uint32_t reduction_steps;
    if (params.crt_count == 1) {
        reduction_steps = 1;
    } else {
        reduction_steps = 1u << (64 - 2 * q2_bits - 1);
        if (reduction_steps < 1) reduction_steps = 1;
    }

    const uint64_t* a_ct_base = d_a_ct_tilde + (size_t)output * poly_len * cpn;

    // Register accumulators: [crt][pos] for r and r_bar
    uint64_t acc_r[4] = {0, 0, 0, 0};   // [m0_z0, m0_z1, m1_z0, m1_z1]
    uint64_t acc_rb[4] = {0, 0, 0, 0};

    for (uint32_t j = 0; j < poly_len; j++) {
        uint32_t index = (uint32_t)(((uint64_t)j * gp_val) % (2 * poly_len));
        uint32_t index_bar = (uint32_t)((2 * poly_len
            - ((uint64_t)j * gp_val) % (2 * poly_len)) % (2 * poly_len));

        const uint64_t* mono = (index < poly_len)
            ? d_monomial_ntts + (size_t)(index % poly_len) * cpn
            : d_neg_monomial_ntts + (size_t)(index % poly_len) * cpn;
        const uint64_t* mono_bar = (index_bar < poly_len)
            ? d_monomial_ntts + (size_t)(index_bar % poly_len) * cpn
            : d_neg_monomial_ntts + (size_t)(index_bar % poly_len) * cpn;

        const uint64_t* a_ct_j = a_ct_base + (size_t)j * cpn;

        for (uint32_t m = 0; m < crt_count; m++) {
            uint64_t mod_m = params.moduli[m];
            uint64_t cr = params.barrett_cr[m];
            uint32_t base = m * poly_len;
            {
                uint64_t a = __ldg(&a_ct_j[base + tid]);
                acc_r[m * 2]  += barrett_raw_u64(__ldg(&mono[base + tid]) * a, cr, mod_m);
                acc_rb[m * 2] += barrett_raw_u64(__ldg(&mono_bar[base + tid]) * a, cr, mod_m);
            }
            if (tid + 1024 < poly_len) {
                uint64_t a = __ldg(&a_ct_j[base + tid + 1024]);
                acc_r[m * 2 + 1]  += barrett_raw_u64(__ldg(&mono[base + tid + 1024]) * a, cr, mod_m);
                acc_rb[m * 2 + 1] += barrett_raw_u64(__ldg(&mono_bar[base + tid + 1024]) * a, cr, mod_m);
            }
        }

        if ((j + 1) % reduction_steps == 0) {
            for (uint32_t m = 0; m < crt_count; m++) {
                uint64_t mod_m = params.moduli[m];
                uint64_t cr = params.barrett_cr[m];
                acc_r[m*2]     = barrett_raw_u64(acc_r[m*2], cr, mod_m);
                acc_r[m*2+1]   = barrett_raw_u64(acc_r[m*2+1], cr, mod_m);
                acc_rb[m*2]    = barrett_raw_u64(acc_rb[m*2], cr, mod_m);
                acc_rb[m*2+1]  = barrett_raw_u64(acc_rb[m*2+1], cr, mod_m);
            }
        }
    }

    // Final reduction + multiply by mod_inv_poly
    for (uint32_t m = 0; m < crt_count; m++) {
        uint64_t mod_m = params.moduli[m];
        uint64_t cr = params.barrett_cr[m];
        acc_r[m*2]   = barrett_raw_u64(acc_r[m*2], cr, mod_m);
        acc_r[m*2+1] = barrett_raw_u64(acc_r[m*2+1], cr, mod_m);
        acc_rb[m*2]  = barrett_raw_u64(acc_rb[m*2], cr, mod_m);
        acc_rb[m*2+1]= barrett_raw_u64(acc_rb[m*2+1], cr, mod_m);

        uint64_t inv0 = __ldg(&d_mod_inv_poly[m * poly_len + tid]);
        acc_r[m*2]  = barrett_raw_u64(acc_r[m*2] * inv0, cr, mod_m);
        acc_rb[m*2] = barrett_raw_u64(acc_rb[m*2] * inv0, cr, mod_m);
        if (tid + 1024 < poly_len) {
            uint64_t inv1 = __ldg(&d_mod_inv_poly[m * poly_len + tid + 1024]);
            acc_r[m*2+1]  = barrett_raw_u64(acc_r[m*2+1] * inv1, cr, mod_m);
            acc_rb[m*2+1] = barrett_raw_u64(acc_rb[m*2+1] * inv1, cr, mod_m);
        }
    }

    // Automorphism for r: τ_{gen_pows[i]}
    // Write to smem, then gather via table
    for (uint32_t m = 0; m < crt_count; m++) {
        smem[m * poly_len + tid] = acc_r[m * 2];
        if (tid + 1024 < poly_len)
            smem[m * poly_len + tid + 1024] = acc_r[m * 2 + 1];
    }
    __syncthreads();

    uint32_t t_val = d_gen_pows[i];
    uint32_t tidx1 = (t_val - 1) / 2;
    const uint32_t* tab1 = d_tables + (size_t)tidx1 * poly_len;

    size_t r_off = ((size_t)output * gridDim.x + i) * cpn;
    for (uint32_t m = 0; m < crt_count; m++) {
        d_r_all[r_off + m * poly_len + tid] = smem[m * poly_len + __ldg(&tab1[tid])];
        if (tid + 1024 < poly_len)
            d_r_all[r_off + m * poly_len + tid + 1024] = smem[m * poly_len + __ldg(&tab1[tid + 1024])];
    }
    __syncthreads();

    // Automorphism for r_bar: τ_{2*poly_len - gen_pows[i]}
    for (uint32_t m = 0; m < crt_count; m++) {
        smem[m * poly_len + tid] = acc_rb[m * 2];
        if (tid + 1024 < poly_len)
            smem[m * poly_len + tid + 1024] = acc_rb[m * 2 + 1];
    }
    __syncthreads();

    uint32_t t_val2 = 2 * poly_len - t_val;
    uint32_t tidx2 = (t_val2 - 1) / 2;
    const uint32_t* tab2 = d_tables + (size_t)tidx2 * poly_len;

    for (uint32_t m = 0; m < crt_count; m++) {
        d_r_bar_all[r_off + m * poly_len + tid] = smem[m * poly_len + __ldg(&tab2[tid])];
        if (tid + 1024 < poly_len)
            d_r_bar_all[r_off + m * poly_len + tid + 1024] = smem[m * poly_len + __ldg(&tab2[tid + 1024])];
    }
}

// ============================================================================
// Kernel 5: Backward Reduction (Phase 2)
//
// Persistent kernel: each block runs num_iter sequential iterations for one output.
// Grid: num_outputs, Block: 1024
// Smem: cpn u64 (= 32 KB for poly_len=2048, crt_count=2)
//
// Key design: processes gadget digits one at a time to avoid 128 KB smem.
// After INTT + CRT compose, composed values live in registers (2 per thread).
// Each digit is extracted, NTT'd in ntt_buf, condensed+stored, and accumulated.
// ============================================================================

__launch_bounds__(1024, 1)
__global__ void sw_backward_reduction(
    uint64_t* __restrict__ d_r_all,
    uint64_t* __restrict__ d_r_bar_all,
    uint64_t* __restrict__ d_bold_t_condensed,
    uint64_t* __restrict__ d_bold_t_bar_condensed,
    uint64_t* __restrict__ d_bold_t_hat_condensed,
    uint64_t* __restrict__ d_a_hat,
    const uint64_t* __restrict__ d_w_all,
    const uint64_t* __restrict__ d_w_bar_all,
    const uint64_t* __restrict__ d_v_mask,
    NTTParams params,
    uint32_t t_exp_left,
    uint32_t bits_per,
    uint32_t num_to_pack_half)
{
    uint32_t output = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t poly_len = params.poly_len;
    uint32_t crt_count = params.crt_count;
    uint32_t cpn = crt_count * poly_len;
    uint32_t num_iter = num_to_pack_half - 1;
    uint64_t bit_mask = (1ULL << bits_per) - 1;

    extern __shared__ uint64_t smem[];
    uint64_t* ntt_buf = smem;  // cpn u64

    size_t r_base = (size_t)output * num_to_pack_half * cpn;
    size_t bt_base = (size_t)output * num_iter * t_exp_left * poly_len;

    // Two paths: p=0 is r/w/bold_t, p=1 is r_bar/w_bar/bold_t_bar
    uint64_t* d_r_ptrs[2]  = {d_r_all, d_r_bar_all};
    const uint64_t* d_w_ptrs[2]  = {d_w_all, d_w_bar_all};
    uint64_t* d_bt_ptrs[2] = {d_bold_t_condensed, d_bold_t_bar_condensed};

    // ---- Main backward loop ----
    for (int ii = (int)num_iter - 1; ii >= 0; ii--) {
        uint32_t i = (uint32_t)ii;

        for (uint32_t p = 0; p < 2; p++) {
            // Load r[i+1] into ntt_buf
            uint64_t* r_ip1 = d_r_ptrs[p] + r_base + (size_t)(i + 1) * cpn;
            for (uint32_t kk = tid; kk < cpn; kk += blockDim.x)
                ntt_buf[kk] = r_ip1[kk];
            __syncthreads();

            // INTT
            for (uint32_t m = 0; m < crt_count; m++)
                ntt_inverse_alt_parallel(ntt_buf + m * poly_len, &params, m, tid, blockDim.x);

            // CRT compose -> registers (2 values per thread, positions tid and tid+1024)
            uint64_t comp0 = ntt_buf[tid];
            uint64_t comp1 = ntt_buf[tid + 1024];

            // Accumulators for w multiply-add (per CRT × position)
            uint64_t macc[4] = {0, 0, 0, 0};

            // Process each gadget digit one at a time
            for (uint32_t k = 0; k < t_exp_left; k++) {
                uint32_t boff = k * bits_per;
                uint64_t d0 = (boff >= 64) ? 0ULL : ((comp0 >> boff) & bit_mask);
                uint64_t d1 = (boff >= 64) ? 0ULL : ((comp1 >> boff) & bit_mask);

                // Write digit to ntt_buf (same value for both CRT slots)
                ntt_buf[tid] = d0;
                ntt_buf[tid + 1024] = d1;
                ntt_buf[poly_len + tid] = d0;
                ntt_buf[poly_len + tid + 1024] = d1;
                __syncthreads();

                // Forward NTT per CRT
                for (uint32_t m = 0; m < crt_count; m++)
                    ntt_forward_alt_parallel(ntt_buf + m * poly_len, &params, m, tid, blockDim.x);

                // Condense & store bold_t[i][k]
                uint64_t* bt_dst = d_bt_ptrs[p] + bt_base + ((size_t)i * t_exp_left + k) * poly_len;
                if (crt_count >= 2) {
                    bt_dst[tid] = ntt_buf[tid] | (ntt_buf[poly_len + tid] << 32);
                    bt_dst[tid + 1024] = ntt_buf[tid + 1024] | (ntt_buf[poly_len + tid + 1024] << 32);
                } else {
                    bt_dst[tid] = ntt_buf[tid];
                    bt_dst[tid + 1024] = ntt_buf[tid + 1024];
                }

                // Accumulate w[i,k] * ntt_val (Barrett reduce each product for Q≈2^32)
                for (uint32_t m = 0; m < crt_count; m++) {
                    uint64_t mod_m = params.moduli[m];
                    uint64_t cr = params.barrett_cr[m];
                    size_t w_off = ((size_t)i * t_exp_left + k) * cpn + m * poly_len;
                    macc[m * 2]     += barrett_raw_u64(__ldg(&d_w_ptrs[p][w_off + tid])     * ntt_buf[m * poly_len + tid], cr, mod_m);
                    macc[m * 2 + 1] += barrett_raw_u64(__ldg(&d_w_ptrs[p][w_off + tid + 1024]) * ntt_buf[m * poly_len + tid + 1024], cr, mod_m);
                }
            }

            // Barrett reduce and add to r[i]
            for (uint32_t m = 0; m < crt_count; m++) {
                uint64_t mod_m = params.moduli[m];
                uint64_t cr = params.barrett_cr[m];
                uint64_t rv0 = barrett_raw_u64(macc[m * 2], cr, mod_m);
                uint64_t rv1 = barrett_raw_u64(macc[m * 2 + 1], cr, mod_m);

                uint64_t* r_i = d_r_ptrs[p] + r_base + (size_t)i * cpn + m * poly_len;
                uint64_t new0 = r_i[tid] + rv0;       if (new0 >= mod_m) new0 -= mod_m;
                uint64_t new1 = r_i[tid + 1024] + rv1; if (new1 >= mod_m) new1 -= mod_m;
                r_i[tid] = new0;
                r_i[tid + 1024] = new1;
            }
        }
    }

    // ---- Final: bold_t_hat from r_bar_all[0] ----
    {
        uint64_t* rb_0 = d_r_bar_all + r_base;
        for (uint32_t kk = tid; kk < cpn; kk += blockDim.x)
            ntt_buf[kk] = rb_0[kk];
        __syncthreads();

        for (uint32_t m = 0; m < crt_count; m++)
            ntt_inverse_alt_parallel(ntt_buf + m * poly_len, &params, m, tid, blockDim.x);

        uint64_t comp0 = ntt_buf[tid];
        uint64_t comp1 = ntt_buf[tid + 1024];

        // v_mask accumulators (for updating r_all[0])
        uint64_t vacc[4] = {0, 0, 0, 0};
        size_t bt_hat_base = (size_t)output * t_exp_left * poly_len;

        for (uint32_t k = 0; k < t_exp_left; k++) {
            uint32_t boff = k * bits_per;
            uint64_t d0 = (boff >= 64) ? 0ULL : ((comp0 >> boff) & bit_mask);
            uint64_t d1 = (boff >= 64) ? 0ULL : ((comp1 >> boff) & bit_mask);

            ntt_buf[tid] = d0;
            ntt_buf[tid + 1024] = d1;
            ntt_buf[poly_len + tid] = d0;
            ntt_buf[poly_len + tid + 1024] = d1;
            __syncthreads();

            for (uint32_t m = 0; m < crt_count; m++)
                ntt_forward_alt_parallel(ntt_buf + m * poly_len, &params, m, tid, blockDim.x);

            // Condense bold_t_hat[k]
            uint64_t* bth_dst = d_bold_t_hat_condensed + bt_hat_base + k * poly_len;
            if (crt_count >= 2) {
                bth_dst[tid] = ntt_buf[tid] | (ntt_buf[poly_len + tid] << 32);
                bth_dst[tid + 1024] = ntt_buf[tid + 1024] | (ntt_buf[poly_len + tid + 1024] << 32);
            } else {
                bth_dst[tid] = ntt_buf[tid];
                bth_dst[tid + 1024] = ntt_buf[tid + 1024];
            }

            // Accumulate v_mask[k] * ntt_val (Barrett reduce each for Q≈2^32)
            for (uint32_t m = 0; m < crt_count; m++) {
                uint64_t mod_m = params.moduli[m];
                uint64_t cr = params.barrett_cr[m];
                size_t v_off = k * cpn + m * poly_len;
                vacc[m * 2]     += barrett_raw_u64(__ldg(&d_v_mask[v_off + tid])     * ntt_buf[m * poly_len + tid], cr, mod_m);
                vacc[m * 2 + 1] += barrett_raw_u64(__ldg(&d_v_mask[v_off + tid + 1024]) * ntt_buf[m * poly_len + tid + 1024], cr, mod_m);
            }
        }

        // Add v_mask accumulator to r_all[0]
        for (uint32_t m = 0; m < crt_count; m++) {
            uint64_t mod_m = params.moduli[m];
            uint64_t cr = params.barrett_cr[m];
            uint64_t rv0 = barrett_raw_u64(vacc[m * 2], cr, mod_m);
            uint64_t rv1 = barrett_raw_u64(vacc[m * 2 + 1], cr, mod_m);

            uint64_t* r_0 = d_r_all + r_base + m * poly_len;
            uint64_t new0 = r_0[tid] + rv0;       if (new0 >= mod_m) new0 -= mod_m;
            uint64_t new1 = r_0[tid + 1024] + rv1; if (new1 >= mod_m) new1 -= mod_m;
            r_0[tid] = new0;
            r_0[tid + 1024] = new1;
        }
        __syncthreads();

        // a_hat = INTT(r_all[0]) -> CRT compose
        uint64_t* r0_ptr = d_r_all + r_base;
        for (uint32_t kk = tid; kk < cpn; kk += blockDim.x)
            ntt_buf[kk] = r0_ptr[kk];
        __syncthreads();

        for (uint32_t m = 0; m < crt_count; m++)
            ntt_inverse_alt_parallel(ntt_buf + m * poly_len, &params, m, tid, blockDim.x);

        uint64_t* a_hat_out = d_a_hat + (size_t)output * poly_len;
        a_hat_out[tid] = ntt_buf[tid];
        a_hat_out[tid + 1024] = ntt_buf[tid + 1024];
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

void* sw_inspir_precomp_init(
    const uint64_t* d_hint_0,
    uint32_t db_cols,
    uint32_t poly_len,
    uint32_t crt_count,
    uint32_t t_exp_left,
    uint32_t modulus_log2,
    uint32_t q2_bits,
    uint32_t num_outputs,
    const uint64_t* moduli,
    const uint64_t* barrett_cr,
    const uint64_t* forward_table,
    const uint64_t* forward_prime_table,
    const uint64_t* inverse_table,
    const uint64_t* inverse_prime_table,
    uint64_t mod0_inv_mod1,
    uint64_t mod1_inv_mod0,
    uint64_t barrett_cr_0_modulus,
    uint64_t barrett_cr_1_modulus,
    uint64_t modulus,
    const uint64_t* w_mask,        // host [t_exp * cpn]
    const uint64_t* v_mask,        // host [t_exp * cpn]
    const uint64_t* mod_inv_poly,  // host [cpn]
    const uint32_t* tables,        // host [num_tables * poly_len]
    uint32_t num_tables,
    const uint32_t* gen_pows,      // host [poly_len]
    uint32_t gen_pows_len)
{
    SwInspirPrecompContext* ctx = new SwInspirPrecompContext();
    ctx->poly_len = poly_len;
    ctx->crt_count = crt_count;
    ctx->t_exp_left = t_exp_left;
    ctx->num_outputs = num_outputs;
    ctx->num_to_pack_half = poly_len / 2;
    ctx->num_iter = poly_len / 2 - 1;
    ctx->d_hint_0 = d_hint_0;
    ctx->db_cols = db_cols;
    // Match spiral-rs gadget.rs get_bits_per: floor(modulus_log2 / dim) + 1
    ctx->bits_per = modulus_log2 / t_exp_left + 1;
    ctx->q2_bits = q2_bits;

    CUDA_ASSERT(cudaStreamCreate(&ctx->stream));

    uint32_t cpn = crt_count * poly_len;
    NTTParams& np = ctx->ntt_params;
    np.poly_len = poly_len;
    np.log2_poly_len = 0;
    for (uint32_t v = poly_len; v > 1; v >>= 1) np.log2_poly_len++;
    np.crt_count = crt_count;
    np.mod0_inv_mod1 = mod0_inv_mod1;
    np.mod1_inv_mod0 = mod1_inv_mod0;
    np.barrett_cr_0_modulus = barrett_cr_0_modulus;
    np.barrett_cr_1_modulus = barrett_cr_1_modulus;
    np.modulus = modulus;

    CUDA_ALLOC_AND_COPY(np.moduli, moduli, crt_count * sizeof(uint64_t));
    CUDA_ALLOC_AND_COPY(np.barrett_cr, barrett_cr, crt_count * sizeof(uint64_t));
    CUDA_ALLOC_AND_COPY(np.forward_table, forward_table, cpn * sizeof(uint64_t));
    CUDA_ALLOC_AND_COPY(np.forward_prime_table, forward_prime_table, cpn * sizeof(uint64_t));
    CUDA_ALLOC_AND_COPY(np.inverse_table, inverse_table, cpn * sizeof(uint64_t));
    CUDA_ALLOC_AND_COPY(np.inverse_prime_table, inverse_prime_table, cpn * sizeof(uint64_t));

    CUDA_ALLOC_AND_COPY(ctx->d_tables, tables, (size_t)num_tables * poly_len * sizeof(uint32_t));
    ctx->num_tables = num_tables;
    CUDA_ALLOC_AND_COPY(ctx->d_gen_pows, gen_pows, gen_pows_len * sizeof(uint32_t));

    size_t mask_size = (size_t)t_exp_left * cpn * sizeof(uint64_t);
    uint64_t* d_w_mask;
    CUDA_ALLOC_AND_COPY(d_w_mask, w_mask, mask_size);
    CUDA_ALLOC_AND_COPY(ctx->d_v_mask, v_mask, mask_size);
    CUDA_ALLOC_AND_COPY(ctx->d_mod_inv_poly, mod_inv_poly, cpn * sizeof(uint64_t));

    size_t mono_size = (size_t)poly_len * cpn * sizeof(uint64_t);
    CUDA_ASSERT(cudaMalloc(&ctx->d_monomial_ntts, mono_size));
    CUDA_ASSERT(cudaMalloc(&ctx->d_neg_monomial_ntts, mono_size));

    CUDA_ASSERT(cudaMalloc(&ctx->d_a_ct_tilde,
        (size_t)num_outputs * poly_len * cpn * sizeof(uint64_t)));

    size_t w_size = (size_t)ctx->num_iter * t_exp_left * cpn * sizeof(uint64_t);
    CUDA_ASSERT(cudaMalloc(&ctx->d_w_all, w_size));
    CUDA_ASSERT(cudaMalloc(&ctx->d_w_bar_all, w_size));

    size_t r_size = (size_t)num_outputs * ctx->num_to_pack_half * cpn * sizeof(uint64_t);
    CUDA_ASSERT(cudaMalloc(&ctx->d_r_all, r_size));
    CUDA_ASSERT(cudaMalloc(&ctx->d_r_bar_all, r_size));

    // bold_t allocation is DEFERRED to compute() — after monomials + a_ct_tilde are freed.
    // This avoids peak memory = DB + precomp exceeding GPU VRAM on tight devices (T4 16GB).
    ctx->d_bold_t_condensed = nullptr;
    ctx->d_bold_t_bar_condensed = nullptr;

    // bold_t_hat and a_hat are small — allocate now
    size_t bth_size = (size_t)num_outputs * t_exp_left * poly_len * sizeof(uint64_t);
    CUDA_ASSERT(cudaMalloc(&ctx->d_bold_t_hat_condensed, bth_size));

    size_t ah_size = (size_t)num_outputs * poly_len * sizeof(uint64_t);
    CUDA_ASSERT(cudaMalloc(&ctx->d_a_hat, ah_size));

    size_t bt_est = (size_t)num_outputs * ctx->num_iter * t_exp_left * poly_len * sizeof(uint64_t);
    SW_LOG("InspiRING GPU precomp init: poly_len=%u, crt=%u, t_exp=%u, bits_per=%u, outputs=%u\n",
           poly_len, crt_count, t_exp_left, ctx->bits_per, num_outputs);
    SW_LOG("  monomials=%.1f MiB, a_ct=%.1f MiB, w_all=%.1f MiB, r_all=%.1f MiB, bold_t=%.1f MiB (deferred)\n",
           2 * mono_size / (double)(1ULL << 20),
           (size_t)num_outputs * poly_len * cpn * 8 / (double)(1ULL << 20),
           2 * w_size / (double)(1ULL << 20), 2 * r_size / (double)(1ULL << 20), 2 * bt_est / (double)(1ULL << 20));

    // Run Kernel 1: monomial NTTs
    {
        size_t smem = cpn * sizeof(uint64_t);
        sw_compute_monomials<<<poly_len, 1024, smem, ctx->stream>>>(
            ctx->d_monomial_ntts, ctx->d_neg_monomial_ntts, ctx->ntt_params);
        CUDA_ASSERT(cudaGetLastError());
    }

    // Run Kernel 3: w rotations
    {
        uint32_t num_blocks = (ctx->num_iter + ROTS_PER_BLK - 1) / ROTS_PER_BLK;
        size_t smem = poly_len * sizeof(uint64_t);
        sw_generate_w_rotations<<<num_blocks, 1024, smem, ctx->stream>>>(
            ctx->d_w_all, ctx->d_w_bar_all, d_w_mask,
            ctx->d_tables, ctx->d_gen_pows,
            t_exp_left, poly_len, crt_count, ctx->num_iter);
        CUDA_ASSERT(cudaGetLastError());
    }

    CUDA_ASSERT(cudaFree(d_w_mask));

    return ctx;
}

void sw_inspir_precomp_compute(void* context)
{
    SwInspirPrecompContext* ctx = (SwInspirPrecompContext*)context;
    if (!ctx) return;

    uint32_t poly_len = ctx->poly_len;
    uint32_t crt_count = ctx->crt_count;
    uint32_t cpn = crt_count * poly_len;
    uint32_t t_exp_left = ctx->t_exp_left;

    bool verbose = sw_verbose();

    // Conditional event timing — zero overhead when VERBOSE != 1
    cudaEvent_t ev_prep_start, ev_prep_end;
    cudaEvent_t ev_p1_start, ev_p1_end;
    cudaEvent_t ev_p2_start, ev_p2_end;
    if (verbose) {
        CUDA_ASSERT(cudaEventCreate(&ev_prep_start));
        CUDA_ASSERT(cudaEventCreate(&ev_prep_end));
        CUDA_ASSERT(cudaEventCreate(&ev_p1_start));
        CUDA_ASSERT(cudaEventCreate(&ev_p1_end));
        CUDA_ASSERT(cudaEventCreate(&ev_p2_start));
        CUDA_ASSERT(cudaEventCreate(&ev_p2_end));
    }

    // Kernel 2: prep pack LWEs
    if (verbose) CUDA_ASSERT(cudaEventRecord(ev_prep_start, ctx->stream));
    {
        size_t smem = cpn * sizeof(uint64_t);
        dim3 grid(poly_len, ctx->num_outputs);
        sw_prep_pack_lwes<<<grid, 1024, smem, ctx->stream>>>(
            ctx->d_a_ct_tilde, ctx->d_hint_0,
            ctx->db_cols, ctx->ntt_params.modulus, ctx->ntt_params);
        CUDA_ASSERT(cudaGetLastError());
    }
    if (verbose) CUDA_ASSERT(cudaEventRecord(ev_prep_end, ctx->stream));

    // Kernel 4: compute r_all (Phase 1)
    if (verbose) CUDA_ASSERT(cudaEventRecord(ev_p1_start, ctx->stream));
    if (crt_count == 1) {
        dim3 grid(ctx->num_to_pack_half, ctx->num_outputs);
        size_t smem = poly_len * sizeof(uint64_t);
        bool use_fast = false;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
        use_fast = true;
#else
        // Runtime check: try fast kernel, fall back if register pressure is too high
        {
            int dev; cudaGetDevice(&dev);
            int sm_major; cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, dev);
            use_fast = (sm_major >= 7);  // SM 7.0+ likely has enough register headroom
        }
#endif
        if (use_fast) {
            sw_compute_r_all_fast<<<grid, 1024, smem, ctx->stream>>>(
                ctx->d_r_all, ctx->d_r_bar_all,
                ctx->d_monomial_ntts, ctx->d_neg_monomial_ntts,
                ctx->d_a_ct_tilde, ctx->d_mod_inv_poly,
                ctx->d_tables, ctx->d_gen_pows,
                ctx->ntt_params);
            cudaError_t err = cudaGetLastError();
            if (err == cudaErrorLaunchOutOfResources || err == cudaErrorInvalidConfiguration) {
                SW_LOG("Phase 1: 128-bit kernel too large, falling back\n");
                use_fast = false;
            } else {
                CUDA_ASSERT(err);
            }
        }
        if (!use_fast) {
            sw_compute_r_all<<<grid, 1024, smem, ctx->stream>>>(
                ctx->d_r_all, ctx->d_r_bar_all,
                ctx->d_monomial_ntts, ctx->d_neg_monomial_ntts,
                ctx->d_a_ct_tilde, ctx->d_mod_inv_poly,
                ctx->d_tables, ctx->d_gen_pows,
                ctx->ntt_params, ctx->q2_bits);
            CUDA_ASSERT(cudaGetLastError());
        }
    } else {
        dim3 grid(ctx->num_to_pack_half, ctx->num_outputs);
        size_t smem = cpn * sizeof(uint64_t);
        sw_compute_r_all<<<grid, 1024, smem, ctx->stream>>>(
            ctx->d_r_all, ctx->d_r_bar_all,
            ctx->d_monomial_ntts, ctx->d_neg_monomial_ntts,
            ctx->d_a_ct_tilde, ctx->d_mod_inv_poly,
            ctx->d_tables, ctx->d_gen_pows,
            ctx->ntt_params, ctx->q2_bits);
        CUDA_ASSERT(cudaGetLastError());
    }
    if (verbose) CUDA_ASSERT(cudaEventRecord(ev_p1_end, ctx->stream));

    // Free monomials and a_ct_tilde (no longer needed)
    CUDA_ASSERT(cudaFree(ctx->d_monomial_ntts));  ctx->d_monomial_ntts = nullptr;
    CUDA_ASSERT(cudaFree(ctx->d_neg_monomial_ntts)); ctx->d_neg_monomial_ntts = nullptr;
    CUDA_ASSERT(cudaFree(ctx->d_a_ct_tilde)); ctx->d_a_ct_tilde = nullptr;

    // NOW allocate bold_t (deferred from init to reduce peak memory)
    {
        size_t bt_size = (size_t)ctx->num_outputs * ctx->num_iter * t_exp_left * poly_len * sizeof(uint64_t);
        CUDA_ASSERT(cudaMalloc(&ctx->d_bold_t_condensed, bt_size));
        CUDA_ASSERT(cudaMalloc(&ctx->d_bold_t_bar_condensed, bt_size));
    }

    // Kernel 5: backward reduction (Phase 2)
    // This is a persistent kernel (1023 sequential iterations per block).
    // Sub-phase breakdown (INTT vs gadget vs key-switch) is not possible via
    // host events — all iterations run inside a single kernel launch.
    if (verbose) CUDA_ASSERT(cudaEventRecord(ev_p2_start, ctx->stream));
    {
        // Need 2*poly_len for ntt_buf even when crt_count=1 (kernel writes both CRT slots)
        size_t smem = 2 * (size_t)ctx->poly_len * sizeof(uint64_t);

        sw_backward_reduction<<<ctx->num_outputs, 1024, smem, ctx->stream>>>(
            ctx->d_r_all, ctx->d_r_bar_all,
            ctx->d_bold_t_condensed, ctx->d_bold_t_bar_condensed,
            ctx->d_bold_t_hat_condensed, ctx->d_a_hat,
            ctx->d_w_all, ctx->d_w_bar_all,
            ctx->d_v_mask, ctx->ntt_params,
            t_exp_left, ctx->bits_per, ctx->num_to_pack_half);
        CUDA_ASSERT(cudaGetLastError());
    }
    if (verbose) CUDA_ASSERT(cudaEventRecord(ev_p2_end, ctx->stream));

    // Synchronize only once (needed for the frees below regardless of verbose)
    CUDA_ASSERT(cudaStreamSynchronize(ctx->stream));

    if (verbose) {
        float prep_ms, phase1_ms, phase2_ms;
        cudaEventElapsedTime(&prep_ms, ev_prep_start, ev_prep_end);
        cudaEventElapsedTime(&phase1_ms, ev_p1_start, ev_p1_end);
        cudaEventElapsedTime(&phase2_ms, ev_p2_start, ev_p2_end);
        SW_LOG("  Prep pack LWEs: %.2f ms\n", prep_ms);
        SW_LOG("  Phase 1 (compute r_all): %.2f ms\n", phase1_ms);
        SW_LOG("  Phase 2 (backward reduction): %.2f ms\n", phase2_ms);
        SW_LOG("InspiRING GPU precomp: prep=%.2f, phase1=%.2f, phase2=%.2f, total=%.2f ms\n",
               prep_ms, phase1_ms, phase2_ms, prep_ms + phase1_ms + phase2_ms);
        cudaEventDestroy(ev_prep_start);
        cudaEventDestroy(ev_prep_end);
        cudaEventDestroy(ev_p1_start);
        cudaEventDestroy(ev_p1_end);
        cudaEventDestroy(ev_p2_start);
        cudaEventDestroy(ev_p2_end);
    }

    // Free intermediates
    CUDA_ASSERT(cudaFree(ctx->d_r_all)); ctx->d_r_all = nullptr;
    CUDA_ASSERT(cudaFree(ctx->d_r_bar_all)); ctx->d_r_bar_all = nullptr;
    CUDA_ASSERT(cudaFree(ctx->d_w_all)); ctx->d_w_all = nullptr;
    CUDA_ASSERT(cudaFree(ctx->d_w_bar_all)); ctx->d_w_bar_all = nullptr;
    CUDA_ASSERT(cudaFree(ctx->d_mod_inv_poly)); ctx->d_mod_inv_poly = nullptr;
    CUDA_ASSERT(cudaFree(ctx->d_v_mask)); ctx->d_v_mask = nullptr;
}

void sw_inspir_precomp_get_results(
    void* context,
    uint64_t** out_bold_t_condensed,
    uint64_t** out_bold_t_bar_condensed,
    uint64_t** out_bold_t_hat_condensed,
    uint64_t** out_a_hat,
    size_t* out_bold_t_size,
    size_t* out_bold_t_bar_size,
    size_t* out_bold_t_hat_size,
    size_t* out_a_hat_size)
{
    SwInspirPrecompContext* ctx = (SwInspirPrecompContext*)context;
    *out_bold_t_condensed = ctx->d_bold_t_condensed;
    *out_bold_t_bar_condensed = ctx->d_bold_t_bar_condensed;
    *out_bold_t_hat_condensed = ctx->d_bold_t_hat_condensed;
    *out_a_hat = ctx->d_a_hat;

    uint32_t poly_len = ctx->poly_len;
    uint32_t t_exp = ctx->t_exp_left;
    uint32_t ni = ctx->num_iter;
    uint32_t no = ctx->num_outputs;

    *out_bold_t_size = (size_t)no * ni * t_exp * poly_len * sizeof(uint64_t);
    *out_bold_t_bar_size = *out_bold_t_size;
    *out_bold_t_hat_size = (size_t)no * t_exp * poly_len * sizeof(uint64_t);
    *out_a_hat_size = (size_t)no * poly_len * sizeof(uint64_t);
}

void sw_inspir_precomp_free(void* context, bool free_outputs)
{
    SwInspirPrecompContext* ctx = (SwInspirPrecompContext*)context;
    if (!ctx) return;

    if (ctx->ntt_params.moduli) cudaFree(ctx->ntt_params.moduli);
    if (ctx->ntt_params.barrett_cr) cudaFree(ctx->ntt_params.barrett_cr);
    if (ctx->ntt_params.forward_table) cudaFree(ctx->ntt_params.forward_table);
    if (ctx->ntt_params.forward_prime_table) cudaFree(ctx->ntt_params.forward_prime_table);
    if (ctx->ntt_params.inverse_table) cudaFree(ctx->ntt_params.inverse_table);
    if (ctx->ntt_params.inverse_prime_table) cudaFree(ctx->ntt_params.inverse_prime_table);
    if (ctx->d_tables) cudaFree(ctx->d_tables);
    if (ctx->d_gen_pows) cudaFree(ctx->d_gen_pows);
    if (ctx->d_monomial_ntts) cudaFree(ctx->d_monomial_ntts);
    if (ctx->d_neg_monomial_ntts) cudaFree(ctx->d_neg_monomial_ntts);
    if (ctx->d_a_ct_tilde) cudaFree(ctx->d_a_ct_tilde);
    if (ctx->d_w_all) cudaFree(ctx->d_w_all);
    if (ctx->d_w_bar_all) cudaFree(ctx->d_w_bar_all);
    if (ctx->d_r_all) cudaFree(ctx->d_r_all);
    if (ctx->d_r_bar_all) cudaFree(ctx->d_r_bar_all);
    if (ctx->d_mod_inv_poly) cudaFree(ctx->d_mod_inv_poly);
    if (ctx->d_v_mask) cudaFree(ctx->d_v_mask);

    if (free_outputs) {
        if (ctx->d_bold_t_condensed) cudaFree(ctx->d_bold_t_condensed);
        if (ctx->d_bold_t_bar_condensed) cudaFree(ctx->d_bold_t_bar_condensed);
        if (ctx->d_bold_t_hat_condensed) cudaFree(ctx->d_bold_t_hat_condensed);
        if (ctx->d_a_hat) cudaFree(ctx->d_a_hat);
    }

    if (ctx->stream) cudaStreamDestroy(ctx->stream);
    delete ctx;
}

} // extern "C"
