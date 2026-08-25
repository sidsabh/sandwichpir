/**
 * Shared InspiRING packing kernels for GPU online computation.
 *
 * Used by both online_kernel_word.cu (Word SimplePIR) and online_kernel_sp.cu
 * (ring SimplePIR). All kernels are static to avoid linker conflicts when
 * included from multiple translation units.
 *
 * Requires: ntt.cuh (NTTParams, barrett_raw_u64, ntt_inverse_kernel_parallel, crt_compose_2)
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

// ---------- Helper functions ----------

static __device__ __forceinline__
void inspir_write_arbitrary_bits(uint8_t* out, uint64_t val, size_t bit_offs, size_t num_bits) {
    for (size_t i = 0; i < num_bits; i++) {
        size_t abs_bit = bit_offs + i;
        size_t byte_idx = abs_bit >> 3;
        size_t bit_in_byte = abs_bit & 7;
        if ((val >> i) & 1ULL) {
            size_t word_byte = byte_idx & ~3ULL;
            uint32_t* wordp = (uint32_t*)(out + word_byte);
            uint32_t bitpos = (uint32_t)((byte_idx - word_byte) * 8 + bit_in_byte);
            atomicOr(wordp, 1u << bitpos);
        }
    }
}

static __host__ __device__ __forceinline__
uint32_t inspir_ceil_log2_u64(uint64_t x) {
    if (x <= 1) return 0;
    uint64_t y = x - 1;
#if defined(__CUDA_ARCH__)
    return 64u - (uint32_t)__clzll(y);
#else
    return 64u - (uint32_t)__builtin_clzll(y);
#endif
}

// ---------- Constants ----------

#define FIP_TILE_D 32  // d-values per shared memory tile (16KB smem -> 2 blocks/SM)
#define FIP_TILE_Z 32  // z-values per block (= warp size)

// ---------- InspiRING Fused Expand + Inner Product ----------
//
// For each (client, output, z): accumulate
//   sum = SUM_{rot,k} y_body[rot,k,perm(z)] * bold_t[rot*t+k, z]
//       + SUM_{rot,k} y_body[rot,k,perm_bar(z)] * bold_t_bar[rot*t+k, z]
//       + SUM_k z_body[k, z] * bold_t_hat[k, z]
//
// Block: dim3(FIP_TILE_Z, batch_per_block)
// Grid:  dim3(ceil(poly_len/FIP_TILE_Z), ceil(batch/bpb), num_outputs)

static __global__ void __launch_bounds__(256)
inspir_fused_expand_ip(
    uint64_t* __restrict__ d_scratch_base,       // contiguous scratch [batch, num_outputs, scratch_per_output]
    const uint64_t* __restrict__ d_bold_t,       // [num_outputs, D, poly_len] -- SHARED across all clients
    const uint64_t* __restrict__ d_bold_t_bar,   // same
    const uint64_t* __restrict__ d_bold_t_hat,   // [num_outputs * t_exp_left * poly_len]
    const uint64_t* __restrict__ d_y_body_base,  // [batch, t_exp_left * poly_len] -- tiny, L2-cached
    const uint64_t* __restrict__ d_z_body_base,  // [batch, t_exp_left * poly_len]
    const uint32_t* __restrict__ d_tables,       // [num_tables * poly_len] permutation tables
    const uint32_t* __restrict__ d_gen_pows,     // [num_rotations] generator powers
    size_t actual_batch_size,
    size_t num_outputs,
    size_t D,                                     // num_iter * t_exp_left
    size_t poly_len,
    size_t t_exp_left,
    size_t inspir_scratch_per_output,
    size_t y_body_stride,                         // t_exp_left * poly_len (elements per client)
    size_t z_body_stride,                         // same
    size_t scratch_stride,                        // num_outputs * inspir_scratch_per_output
    NTTParams params)
{
    size_t z = blockIdx.x * FIP_TILE_Z + threadIdx.x;
    if (z >= poly_len) return;

    size_t client = blockIdx.y * blockDim.y + threadIdx.y;
    if (client >= actual_batch_size) return;

    size_t o = blockIdx.z;  // output index -- parallelized across grid

    // Barrett constants
    uint64_t bcr0 = params.barrett_cr[0];
    uint64_t bcr1 = params.barrett_cr[1];
    uint64_t mod0 = params.moduli[0];
    uint64_t mod1 = params.moduli[1];

    // Per-client data pointers
    const uint64_t* my_y_body = d_y_body_base + client * y_body_stride;
    const uint64_t* my_z_body = d_z_body_base + client * z_body_stride;
    uint64_t* my_scratch      = d_scratch_base + client * scratch_stride;

    // Shared memory: bold_t tile (shared by ALL clients in this block)
    __shared__ uint64_t s_bt[FIP_TILE_D][FIP_TILE_Z];
    __shared__ uint64_t s_btbar[FIP_TILE_D][FIP_TILE_Z];

    uint64_t acc_lo = 0, acc_hi = 0;

    for (size_t d_start = 0; d_start < D; d_start += FIP_TILE_D) {
        size_t tile_len = FIP_TILE_D;
        if (d_start + tile_len > D) tile_len = D - d_start;

        // Cooperative load of bold_t into shared memory.
        // All threads participate for balanced loading.
        {
            size_t tid = threadIdx.y * FIP_TILE_Z + threadIdx.x;
            size_t block_threads = blockDim.y * FIP_TILE_Z;
            for (size_t idx = tid; idx < tile_len * FIP_TILE_Z; idx += block_threads) {
                size_t dd = idx / FIP_TILE_Z;
                size_t zz = idx % FIP_TILE_Z;
                size_t gz = blockIdx.x * FIP_TILE_Z + zz;
                if (gz < poly_len) {
                    s_bt[dd][zz]    = __ldg(&d_bold_t[o * D * poly_len + (d_start + dd) * poly_len + gz]);
                    s_btbar[dd][zz] = __ldg(&d_bold_t_bar[o * D * poly_len + (d_start + dd) * poly_len + gz]);
                }
            }
        }
        __syncthreads();

        // Each client accumulates against shared bold_t.
        // On-the-fly expand: gather y_body via permutation tables.
        // Barrett at tile boundary only (FIP_TILE_D=32, adds 64 products <= addition_capacity).
        {
            size_t prev_rot = (size_t)-1;
            uint32_t perm_idx1 = 0, perm_idx2 = 0;
            for (size_t dd = 0; dd < tile_len; dd++) {
                size_t d_idx = d_start + dd;
                size_t rot = d_idx / t_exp_left;
                size_t k   = d_idx - rot * t_exp_left;

                if (rot != prev_rot) {
                    uint32_t gen = __ldg(&d_gen_pows[rot]);
                    uint32_t tidx1 = (gen - 1) / 2;
                    uint32_t tidx2 = (2 * (uint32_t)poly_len - gen - 1) / 2;
                    perm_idx1 = __ldg(&d_tables[(size_t)tidx1 * poly_len + z]);
                    perm_idx2 = __ldg(&d_tables[(size_t)tidx2 * poly_len + z]);
                    prev_rot = rot;
                }

                uint64_t y_val  = __ldg(&my_y_body[k * poly_len + perm_idx1]);
                uint64_t yb_val = __ldg(&my_y_body[k * poly_len + perm_idx2]);

                uint64_t t_val  = s_bt[dd][threadIdx.x];
                uint64_t tb_val = s_btbar[dd][threadIdx.x];

                acc_lo += (uint64_t)(uint32_t)y_val  * (uint32_t)t_val
                        + (uint64_t)(uint32_t)yb_val * (uint32_t)tb_val;
                acc_hi += (uint64_t)(uint32_t)(y_val >> 32)  * (uint32_t)(t_val >> 32)
                        + (uint64_t)(uint32_t)(yb_val >> 32) * (uint32_t)(tb_val >> 32);
            }
        }
        // Barrett reduce once per tile (branch-free inner loop)
        acc_lo = barrett_raw_u64(acc_lo, bcr0, mod0);
        acc_hi = barrett_raw_u64(acc_hi, bcr1, mod1);
        __syncthreads();
    }

    // Add z_body x bold_t_hat (t_exp_left terms)
    uint64_t fin_lo = 0, fin_hi = 0;
    for (size_t k = 0; k < t_exp_left; k++) {
        uint64_t z_val  = __ldg(&my_z_body[k * poly_len + z]);
        uint64_t th_val = __ldg(&d_bold_t_hat[o * t_exp_left * poly_len + k * poly_len + z]);
        fin_lo += (z_val & 0xFFFFFFFF) * (th_val & 0xFFFFFFFF);
        fin_hi += (z_val >> 32)         * (th_val >> 32);
    }
    fin_lo = barrett_raw_u64(fin_lo, bcr0, mod0);
    fin_hi = barrett_raw_u64(fin_hi, bcr1, mod1);

    acc_lo += fin_lo;
    if (acc_lo >= mod0) acc_lo -= mod0;
    acc_hi += fin_hi;
    if (acc_hi >= mod1) acc_hi -= mod1;

    // Write CRT halves to this client's scratch
    uint64_t* temp_ntt = my_scratch + o * inspir_scratch_per_output;
    temp_ntt[z]            = acc_lo;
    temp_ntt[poly_len + z] = acc_hi;
}

// ---------- InspiRING Multi-Output Fused kernel ----------
// Each thread processes ALL outputs, reading y_body ONCE and reusing across outputs.
// This eliminates num_outputs x redundant y_body scattered reads and table lookups.
// bold_t is read from global memory via __ldg (coalesced per output).
//
// Block: dim3(32, batch_per_block) -- 32 z-values x N clients
// Grid: dim3(ceil(poly_len/32), ceil(batch/bpb)) -- NO output dimension
// No shared memory needed (bold_t coalesced from global, y_body from L2).
// Templated on NUM_OUTPUTS so compiler can keep accumulators in registers.

template <int NUM_OUTPUTS>
static __global__ void __launch_bounds__(256)
inspir_fused_multi_output(
    uint64_t* __restrict__ d_scratch_base,
    const uint64_t* __restrict__ d_bold_t,
    const uint64_t* __restrict__ d_bold_t_bar,
    const uint64_t* __restrict__ d_bold_t_hat,
    const uint64_t* __restrict__ d_y_body_base,
    const uint64_t* __restrict__ d_z_body_base,
    const uint32_t* __restrict__ d_tables,
    const uint32_t* __restrict__ d_gen_pows,
    size_t batch_size,
    size_t D,
    size_t poly_len,
    size_t t_exp_left,
    size_t inspir_scratch_per_output,
    size_t y_body_stride,
    size_t z_body_stride,
    size_t scratch_stride,
    NTTParams params)
{
    size_t z = blockIdx.x * 32 + threadIdx.x;
    if (z >= poly_len) return;
    size_t client = blockIdx.y * blockDim.y + threadIdx.y;
    if (client >= batch_size) return;

    uint64_t bcr0 = params.barrett_cr[0];
    uint64_t bcr1 = params.barrett_cr[1];
    uint64_t mod0 = params.moduli[0];
    uint64_t mod1 = params.moduli[1];

    const uint64_t* my_y_body = d_y_body_base + client * y_body_stride;
    const uint64_t* my_z_body = d_z_body_base + client * z_body_stride;
    uint64_t* my_scratch      = d_scratch_base + client * scratch_stride;

    // Compile-time sized accumulators -> compiler keeps in registers
    uint64_t acc_lo[NUM_OUTPUTS];
    uint64_t acc_hi[NUM_OUTPUTS];
    #pragma unroll
    for (int o = 0; o < NUM_OUTPUTS; o++) {
        acc_lo[o] = 0;
        acc_hi[o] = 0;
    }

    size_t prev_rot = (size_t)-1;
    uint32_t perm_idx1 = 0, perm_idx2 = 0;
    size_t num_added = 0;

    for (size_t d_idx = 0; d_idx < D; d_idx++) {
        size_t rot = d_idx / t_exp_left;
        size_t k   = d_idx - rot * t_exp_left;

        if (rot != prev_rot) {
            uint32_t gen = __ldg(&d_gen_pows[rot]);
            uint32_t tidx1 = (gen - 1) / 2;
            uint32_t tidx2 = (2 * (uint32_t)poly_len - gen - 1) / 2;
            perm_idx1 = __ldg(&d_tables[(size_t)tidx1 * poly_len + z]);
            perm_idx2 = __ldg(&d_tables[(size_t)tidx2 * poly_len + z]);
            prev_rot = rot;
        }

        // Read y_body ONCE -- reuse across ALL outputs
        uint64_t y_val  = __ldg(&my_y_body[k * poly_len + perm_idx1]);
        uint64_t yb_val = __ldg(&my_y_body[k * poly_len + perm_idx2]);
        uint32_t y_lo  = (uint32_t)y_val;
        uint32_t y_hi  = (uint32_t)(y_val >> 32);
        uint32_t yb_lo = (uint32_t)yb_val;
        uint32_t yb_hi = (uint32_t)(yb_val >> 32);

        // Accumulate against ALL outputs (bold_t coalesced from global)
        #pragma unroll
        for (int o = 0; o < NUM_OUTPUTS; o++) {
            size_t bt_idx = o * D * poly_len + d_idx * poly_len + z;
            uint64_t t_val  = __ldg(&d_bold_t[bt_idx]);
            uint64_t tb_val = __ldg(&d_bold_t_bar[bt_idx]);

            acc_lo[o] += (uint64_t)y_lo  * (uint32_t)t_val
                       + (uint64_t)yb_lo * (uint32_t)tb_val;
            acc_hi[o] += (uint64_t)y_hi  * (uint32_t)(t_val >> 32)
                       + (uint64_t)yb_hi * (uint32_t)(tb_val >> 32);
        }

        num_added += 2;
        if (num_added >= 64) {
            #pragma unroll
            for (int o = 0; o < NUM_OUTPUTS; o++) {
                acc_lo[o] = barrett_raw_u64(acc_lo[o], bcr0, mod0);
                acc_hi[o] = barrett_raw_u64(acc_hi[o], bcr1, mod1);
            }
            num_added = 0;
        }
    }

    // Final Barrett
    #pragma unroll
    for (int o = 0; o < NUM_OUTPUTS; o++) {
        acc_lo[o] = barrett_raw_u64(acc_lo[o], bcr0, mod0);
        acc_hi[o] = barrett_raw_u64(acc_hi[o], bcr1, mod1);
    }

    // Add z_body x bold_t_hat
    #pragma unroll
    for (int o = 0; o < NUM_OUTPUTS; o++) {
        uint64_t fin_lo = 0, fin_hi = 0;
        for (size_t k = 0; k < t_exp_left; k++) {
            uint64_t z_val  = __ldg(&my_z_body[k * poly_len + z]);
            uint64_t th_val = __ldg(&d_bold_t_hat[o * t_exp_left * poly_len + k * poly_len + z]);
            fin_lo += (z_val & 0xFFFFFFFF) * (th_val & 0xFFFFFFFF);
            fin_hi += (z_val >> 32)         * (th_val >> 32);
        }
        fin_lo = barrett_raw_u64(fin_lo, bcr0, mod0);
        fin_hi = barrett_raw_u64(fin_hi, bcr1, mod1);

        acc_lo[o] += fin_lo;
        if (acc_lo[o] >= mod0) acc_lo[o] -= mod0;
        acc_hi[o] += fin_hi;
        if (acc_hi[o] >= mod1) acc_hi[o] -= mod1;

        uint64_t* temp_ntt = my_scratch + o * inspir_scratch_per_output;
        temp_ntt[z]            = acc_lo[o];
        temp_ntt[poly_len + z] = acc_hi[o];
    }
}

// Dispatch helper for multi-output kernel.
static void inspir_launch_multi_output_kernel(
    size_t num_outputs,
    size_t chunk_size,
    size_t poly_len,
    cudaStream_t stream,
    uint64_t* scratch, const uint64_t* bold_t, const uint64_t* bold_t_bar,
    const uint64_t* bold_t_hat, const uint64_t* y_body, const uint64_t* z_body,
    const uint32_t* tables, const uint32_t* gen_pows,
    size_t batch_size, size_t D, size_t t_exp_left,
    size_t inspir_spo, size_t ybs, size_t zbs, size_t ss, NTTParams params)
{
    size_t batch_per_block;
    if      (num_outputs <= 4)  batch_per_block = 8;
    else if (num_outputs <= 8)  batch_per_block = 4;
    else if (num_outputs <= 16) batch_per_block = 2;
    else                        batch_per_block = 1;
    if (batch_per_block > chunk_size) batch_per_block = chunk_size;

    dim3 block(32, (unsigned)batch_per_block);
    dim3 grid((poly_len + 31) / 32,
              (chunk_size + batch_per_block - 1) / batch_per_block);

    #define INSPIR_LAUNCH_MO(N) \
        inspir_fused_multi_output<N><<<grid, block, 0, stream>>>( \
            scratch, bold_t, bold_t_bar, bold_t_hat, y_body, z_body, \
            tables, gen_pows, batch_size, D, poly_len, t_exp_left, \
            inspir_spo, ybs, zbs, ss, params)

    switch (num_outputs) {
        case 1:  INSPIR_LAUNCH_MO(1);  break;
        case 2:  INSPIR_LAUNCH_MO(2);  break;
        case 3:  INSPIR_LAUNCH_MO(3);  break;
        case 4:  INSPIR_LAUNCH_MO(4);  break;
        case 6:  INSPIR_LAUNCH_MO(6);  break;
        case 8:  INSPIR_LAUNCH_MO(8);  break;
        case 12: INSPIR_LAUNCH_MO(12); break;
        case 16: INSPIR_LAUNCH_MO(16); break;
        case 18: INSPIR_LAUNCH_MO(18); break;
        case 24: INSPIR_LAUNCH_MO(24); break;
        case 32: INSPIR_LAUNCH_MO(32); break;
        default:
            fprintf(stderr, "ERROR: unsupported num_outputs=%zu for multi-output kernel\n", num_outputs);
            abort();
    }
    #undef INSPIR_LAUNCH_MO
}

// ---------- InspiRING post-process kernel (BATCHED) ----------
// Same as inspir_post_process but processes ALL clients in a single launch.
// Grid: dim3(num_outputs, batch_size)
// Block: 1024 threads
// scratch_stride: num_outputs * inspir_scratch_per_output (per-client scratch spacing)
// inter_stride: num_outputs * poly_len (per-client intermediate spacing)

static __global__ void __launch_bounds__(1024)
inspir_post_process_batched(
    uint8_t* d_response_out,
    const uint64_t* d_intermediate,
    const uint64_t* d_a_hat,
    uint64_t* d_scratch,
    size_t num_outputs,
    uint64_t rlwe_q_prime_1,
    uint64_t rlwe_q_prime_2,
    size_t response_bytes_per_output,
    size_t inspir_scratch_per_output,
    size_t scratch_stride,              // per-client scratch spacing (elements)
    size_t inter_stride,                // per-client intermediate spacing (elements)
    NTTParams params)
{
    size_t output_idx = blockIdx.x;
    size_t batch_idx  = blockIdx.y;
    size_t tid = threadIdx.x;
    size_t poly_len = params.poly_len;
    uint64_t modulus = params.modulus;

    size_t crt_count = params.crt_count;
    size_t threads_per_crt = blockDim.x / crt_count;
    size_t my_crt = tid / threads_per_crt;
    size_t local_tid = tid % threads_per_crt;

    // Per-client, per-output offsets
    uint64_t* my_scratch = d_scratch + batch_idx * scratch_stride + output_idx * inspir_scratch_per_output;
    uint64_t* temp_ntt = my_scratch;
    uint64_t* temp_raw = temp_ntt + 2 * poly_len;

    const uint64_t* my_a_hat  = d_a_hat + output_idx * poly_len;
    const uint64_t* b_values  = d_intermediate + batch_idx * inter_stride + output_idx * poly_len;

    size_t resp_per_batch = num_outputs * response_bytes_per_output;
    uint8_t* my_response = d_response_out + batch_idx * resp_per_batch + output_idx * response_bytes_per_output;

    // INTT both CRT components
    if (my_crt < crt_count)
        ntt_inverse_kernel_parallel(temp_ntt + my_crt * poly_len, &params, my_crt, local_tid, threads_per_crt);
    __syncthreads();

    // CRT compose + add b_values -> raw domain
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t x     = temp_ntt[z];
        uint64_t y_val = temp_ntt[poly_len + z];
        uint64_t sum_raw = crt_compose_2(x, y_val, &params);

        uint64_t final_b = sum_raw + b_values[z];
        final_b -= modulus * (final_b >= modulus);

        temp_raw[poly_len + z] = final_b;
        temp_raw[z]            = my_a_hat[z];
    }
    __syncthreads();

    // Modswitch and bit-pack
    size_t q_1_bits = inspir_ceil_log2_u64(rlwe_q_prime_2);
    size_t q_2_bits = inspir_ceil_log2_u64(rlwe_q_prime_1);

    for (size_t i = tid; i < response_bytes_per_output; i += blockDim.x)
        my_response[i] = 0;
    __syncthreads();

    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t val = temp_raw[z];
        double d_val = (double)val;
        uint64_t val_rescaled = (uint64_t)((d_val * (double)rlwe_q_prime_2) / (double)modulus + 0.5);
        inspir_write_arbitrary_bits(my_response, val_rescaled, z * q_1_bits, q_1_bits);
    }
    __syncthreads();

    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t val = temp_raw[poly_len + z];
        double d_val = (double)val;
        uint64_t val_rescaled = (uint64_t)((d_val * (double)rlwe_q_prime_1) / (double)modulus + 0.5);
        inspir_write_arbitrary_bits(my_response, val_rescaled, poly_len * q_1_bits + z * q_2_bits, q_2_bits);
    }
}

// ---------- InspiRING post-process kernel ----------
// INTT + CRT compose + add b_values + modswitch + bitpack.
// Reads temp_ntt from scratch (written by inspir_fused_expand_ip or inspir_fused_multi_output).
// Requires b_values from Step1 matmul (d_intermediate).
//
// Grid:  num_outputs blocks
// Block: 1024 threads (needed for cooperative INTT)

static __global__ void __launch_bounds__(1024)
inspir_post_process(
    uint8_t* d_response_out,
    const uint64_t* d_intermediate,     // b_values from Step1
    const uint64_t* d_a_hat,            // per-output, raw domain
    uint64_t* d_scratch,                // per-stream scratch (temp_ntt written by inner product)
    size_t num_outputs,
    uint64_t rlwe_q_prime_1,
    uint64_t rlwe_q_prime_2,
    size_t response_bytes_per_output,
    size_t inspir_scratch_per_output,
    NTTParams params)
{
    size_t output_idx = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t poly_len = params.poly_len;
    uint64_t modulus = params.modulus;

    size_t crt_count = params.crt_count;
    size_t threads_per_crt = blockDim.x / crt_count;
    size_t my_crt = tid / threads_per_crt;
    size_t local_tid = tid % threads_per_crt;

    uint64_t* my_scratch = d_scratch + output_idx * inspir_scratch_per_output;
    uint64_t* temp_ntt = my_scratch;
    uint64_t* temp_raw = temp_ntt + 2 * poly_len;

    const uint64_t* my_a_hat  = d_a_hat + output_idx * poly_len;
    const uint64_t* b_values  = d_intermediate + output_idx * poly_len;
    uint8_t* my_response = d_response_out + output_idx * response_bytes_per_output;

    // INTT both CRT components
    if (my_crt < crt_count)
        ntt_inverse_kernel_parallel(temp_ntt + my_crt * poly_len, &params, my_crt, local_tid, threads_per_crt);
    __syncthreads();

    // CRT compose + add b_values -> raw domain
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t x     = temp_ntt[z];
        uint64_t y_val = temp_ntt[poly_len + z];
        uint64_t sum_raw = crt_compose_2(x, y_val, &params);

        uint64_t final_b = sum_raw + b_values[z];
        final_b -= modulus * (final_b >= modulus);

        temp_raw[poly_len + z] = final_b;
        temp_raw[z]            = my_a_hat[z];
    }
    __syncthreads();

    // Modswitch and bit-pack
    size_t q_1_bits = inspir_ceil_log2_u64(rlwe_q_prime_2);
    size_t q_2_bits = inspir_ceil_log2_u64(rlwe_q_prime_1);

    for (size_t i = tid; i < response_bytes_per_output; i += blockDim.x)
        my_response[i] = 0;
    __syncthreads();

    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t val = temp_raw[z];
        double d_val = (double)val;
        uint64_t val_rescaled = (uint64_t)((d_val * (double)rlwe_q_prime_2) / (double)modulus + 0.5);
        inspir_write_arbitrary_bits(my_response, val_rescaled, z * q_1_bits, q_1_bits);
    }
    __syncthreads();

    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t val = temp_raw[poly_len + z];
        double d_val = (double)val;
        uint64_t val_rescaled = (uint64_t)((d_val * (double)rlwe_q_prime_1) / (double)modulus + 0.5);
        inspir_write_arbitrary_bits(my_response, val_rescaled, poly_len * q_1_bits + z * q_2_bits, q_2_bits);
    }
}

// ═══════════════════════════════════════════════════════════════
// SandwichPIR Post-Process (single modulus — no CRT compose)
//
// The RLWE mask polynomial â_o is deterministic from the CRS seed and
// the database (it's the Phase-1 InspiRING precomp output), so its
// rescaled+bitpacked byte form is identical for every query and every
// client. We therefore compute it ONCE at init time via
// `sw_inspir_fill_mask_template` into a compact host-resident buffer,
// and the online post-process kernel below writes ONLY body bytes into
// a body-sized `d_response_body`. The full wire-format response is
// assembled host-side by concatenating the static mask template with
// the per-client body bytes (the wrapper layer decides when to do that;
// body-only clients never pay for the assembly).
// ═══════════════════════════════════════════════════════════════

// One-time kernel: fill a compact per-output mask template buffer.
// Layout: `d_mask_template_out[o * mask_bytes_per_output .. +mask_bytes_per_output)`
// contains the rescaled+bitpacked mask for output `o`. No batch dim —
// the same template is shared across all queries/clients/batches.
// Grid: (num_outputs, 1). Block: 1024 threads.
__global__ void
sw_inspir_fill_mask_template(
    uint8_t* d_mask_template_out,   // [num_outputs][mask_bytes_per_output]
    const uint64_t* d_a_hat,        // [num_outputs][poly_len]
    uint64_t rlwe_q_prime_2,        // q_1 = 2^18 (mask modulus, large)
    size_t mask_bytes_per_output,   // = (poly_len * q_1_bits + 7) / 8
    uint64_t modulus,               // Q
    size_t poly_len)
{
    size_t output_idx = blockIdx.x;
    size_t tid = threadIdx.x;
    const uint64_t* my_a_hat = d_a_hat + output_idx * poly_len;
    uint8_t* my_mask = d_mask_template_out + output_idx * mask_bytes_per_output;

    size_t q_1_bits = inspir_ceil_log2_u64(rlwe_q_prime_2);

    // Zero the mask byte range (write_arbitrary_bits is atomicOr-only,
    // so targets must start zeroed).
    for (size_t i = tid; i < mask_bytes_per_output; i += blockDim.x)
        my_mask[i] = 0;
    __syncthreads();

    // Rescale + bitpack mask coefficients into [0, poly_len * q_1_bits) bits.
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        double d_val = (double)my_a_hat[z];
        uint64_t val_rescaled =
            (uint64_t)((d_val * (double)rlwe_q_prime_2) / (double)modulus + 0.5);
        inspir_write_arbitrary_bits(my_mask, val_rescaled, z * q_1_bits, q_1_bits);
    }
}

// Online post-process. Writes BODY-ONLY bytes per (client, output) into
// `d_response_body_out` at a compact layout:
//     d_response_body_out[batch_idx * num_outputs * body_bytes_per_output
//                          + output_idx * body_bytes_per_output + ...]
// Mask bytes live in a separate `d_mask_template` buffer and are never
// touched here. The host assembles the full wire format by concatenating
// mask template + body when the client needs the full response.
__global__ void
sw_inspir_post_process(
    uint8_t* d_response_body_out,   // [max_B][num_outputs][body_bytes_per_output]
    const uint64_t* d_intermediate,
    uint64_t* d_scratch,
    size_t num_outputs,
    uint64_t rlwe_q_prime_1,        // q_2 = 2^10 (body modulus, small)
    size_t body_bytes_per_output,
    size_t inspir_scratch_per_output,
    size_t scratch_stride,
    size_t inter_stride,
    NTTParams params)
{
    size_t output_idx = blockIdx.x;
    size_t batch_idx  = blockIdx.y;
    size_t tid = threadIdx.x;
    size_t poly_len = params.poly_len;
    uint64_t modulus = params.moduli[0];

    uint64_t* my_scratch = d_scratch + batch_idx * scratch_stride + output_idx * inspir_scratch_per_output;
    uint64_t* temp_ntt = my_scratch;
    uint64_t* temp_raw = temp_ntt + 2 * poly_len;

    const uint64_t* b_values = d_intermediate + batch_idx * inter_stride + output_idx * poly_len;

    size_t body_per_batch = num_outputs * body_bytes_per_output;
    uint8_t* my_body = d_response_body_out
        + batch_idx * body_per_batch
        + output_idx * body_bytes_per_output;

    // Single-modulus INTT (alt path for large modulus ≈ 2^32)
    ntt_inverse_alt_parallel(temp_ntt, &params, 0, tid, blockDim.x);
    __syncthreads();

    // Final body polynomial: INTT result + matmul body values (no CRT compose).
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t sum_raw = temp_ntt[z];
        uint64_t final_b = sum_raw + b_values[z];
        final_b -= modulus * (final_b >= modulus);
        temp_raw[poly_len + z] = final_b;
    }
    __syncthreads();

    size_t q_2_bits = inspir_ceil_log2_u64(rlwe_q_prime_1);

    // Zero ONLY the body byte range (atomicOr needs zeroed targets).
    for (size_t i = tid; i < body_bytes_per_output; i += blockDim.x)
        my_body[i] = 0;
    __syncthreads();

    // Rescale + bitpack body into [0, poly_len * q_2_bits) bits of `my_body`.
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        double d_val = (double)temp_raw[poly_len + z];
        uint64_t val_rescaled = (uint64_t)((d_val * (double)rlwe_q_prime_1) / (double)modulus + 0.5);
        inspir_write_arbitrary_bits(my_body, val_rescaled, z * q_2_bits, q_2_bits);
    }
}
