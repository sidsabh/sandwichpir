#ifndef NTT_CUH
#define NTT_CUH

#include <stdint.h>
#include <stdio.h>

#define CUDA_ASSERT(stmt) do { \
    cudaError_t err = (stmt);  \
    if (err != cudaSuccess) {  \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        abort();               \
    }                          \
} while (0)

struct GpuTimer {
  cudaEvent_t start, stop;
  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }
  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void tic(cudaStream_t s = 0) { cudaEventRecord(start, s); }
  float toc_ms(cudaStream_t s = 0) {
    cudaEventRecord(stop, s);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
  }
};


#define CUDA_ALLOC_AND_COPY(dest, src, size) do { \
    CUDA_ASSERT(cudaMalloc(&(dest), (size))); \
    CUDA_ASSERT(cudaMemcpy((dest), (src), (size), cudaMemcpyHostToDevice)); \
} while(0)


// NTT parameters
struct NTTParams {
    uint32_t poly_len = 0;
    uint32_t log2_poly_len = 0;
    uint32_t crt_count = 0;
    uint64_t* moduli = nullptr;
    uint64_t* barrett_cr = nullptr;  // Barrett constants (cr_1) for 64-bit reduction
    uint64_t* forward_table = nullptr;
    uint64_t* forward_prime_table = nullptr;
    uint64_t* inverse_table = nullptr;
    uint64_t* inverse_prime_table = nullptr;
    uint64_t mod0_inv_mod1 = 0;
    uint64_t mod1_inv_mod0 = 0;
    uint64_t barrett_cr_0_modulus = 0;  // cr_0 for 128-bit CRT reduction
    uint64_t barrett_cr_1_modulus = 0;  // cr_1 for 128-bit CRT reduction
    uint64_t modulus = 0;  // Product of all CRT moduli
};

// Barrett Reduction (matches spiral-rs arith.rs)

// 64-bit Barrett reduction: barrett_raw_u64
// Matches spiral-rs: src/arith.rs::barrett_raw_u64
__device__ __forceinline__ uint64_t barrett_raw_u64(
    uint64_t input,
    uint64_t const_ratio_1,
    uint64_t modulus
) {
    uint64_t tmp = __umul64hi(input, const_ratio_1);
    uint64_t res = input - tmp * modulus;
    return res >= modulus ? res - modulus : res;
}

// 128-bit Barrett reduction: barrett_raw_u128
// Matches spiral-rs: src/arith.rs::barrett_raw_u128
__device__ __forceinline__ uint64_t barrett_raw_u128(
    uint64_t val_lo,
    uint64_t val_hi,
    uint64_t cr0,
    uint64_t cr1,
    uint64_t modulus
) {
    uint64_t zx = val_lo;
    uint64_t zy = val_hi;

    uint64_t tmp1, tmp2x, tmp2y, tmp3, carry;

    // First: zx * cr0, take high part
    uint64_t prody = __umul64hi(zx, cr0);
    carry = prody;

    // zx * cr1
    tmp2x = zx * cr1;
    tmp2y = __umul64hi(zx, cr1);
    // tmp1 = tmp2x + carry (wrapping), overflow1 = 1 if overflowed
    tmp1 = tmp2x + carry;
    uint64_t overflow1 = tmp1 < tmp2x ? 1 : 0;
    tmp3 = tmp2y + overflow1;

    // zy * cr0
    tmp2x = zy * cr0;
    tmp2y = __umul64hi(zy, cr0);
    uint64_t old_tmp1 = tmp1;
    tmp1 = old_tmp1 + tmp2x;
    carry = tmp2y + (tmp1 < old_tmp1 ? 1 : 0);
    tmp1 = zy * cr1 + tmp3 + carry;

    tmp3 = zx - tmp1 * modulus;
    return tmp3;
}

// 128-bit Barrett reduction with final reduction step
// Matches spiral-rs: src/arith.rs::barrett_reduction_u128_raw
__device__ __forceinline__ uint64_t barrett_reduction_u128(
    uint64_t val_lo,
    uint64_t val_hi,
    uint64_t modulus,
    uint64_t cr0,
    uint64_t cr1
) {
    uint64_t reduced = barrett_raw_u128(val_lo, val_hi, cr0, cr1, modulus);
    return reduced >= modulus ? reduced - modulus : reduced;
}

// Forward NTT (parallelized across all threads)

__device__ __forceinline__ void ntt_forward_kernel_parallel(
    uint64_t* operand,
    const NTTParams* params,
    uint32_t coeff_mod,
    uint32_t tid,
    uint32_t block_size
) {
    const uint32_t n = params->poly_len;
    const uint32_t log_n = params->log2_poly_len;
    const uint64_t* forward_table = params->forward_table + coeff_mod * n;
    const uint64_t* forward_prime_table = params->forward_prime_table + coeff_mod * n;
    const uint32_t modulus_small = params->moduli[coeff_mod];
    const uint32_t two_times_modulus_small = 2 * modulus_small;

    // Main butterfly loops (unrolled for register optimization across stages)
    #pragma unroll
    for (uint32_t mm = 0; mm < log_n; mm++) {
        const uint32_t m = 1 << mm;
        const uint32_t t = n >> (mm + 1);
        const uint32_t total_butterflies = m * t;

        for (uint32_t butterfly_idx = tid; butterfly_idx < total_butterflies; butterfly_idx += block_size) {
            const uint32_t i = butterfly_idx / t;
            const uint32_t j = butterfly_idx % t;

            const uint64_t w = forward_table[m + i];
            const uint64_t w_prime = forward_prime_table[m + i];

            const uint32_t idx1 = i * 2 * t + j;
            const uint32_t idx2 = idx1 + t;

            const uint64_t x = operand[idx1];
            const uint64_t y = operand[idx2];

            uint64_t curr_x = x;
            if (x >= two_times_modulus_small) {
                curr_x -= two_times_modulus_small;
            }

            // Harvey/Shoup reduction for w * y (CRT sub-moduli < 2^31)
            const uint64_t q_tmp = (y * w_prime) >> 32;
            const uint64_t q_new = w * y - q_tmp * modulus_small;

            operand[idx1] = curr_x + q_new;
            operand[idx2] = curr_x + (two_times_modulus_small - q_new);
        }
        __syncthreads();
    }

    // Final reduction (parallelized)
    for (uint32_t i = tid; i < n; i += block_size) {
        uint64_t val = operand[i];
        if (val >= two_times_modulus_small) {
            val -= two_times_modulus_small;
        }
        if (val >= modulus_small) {
            val -= modulus_small;
        }
        operand[i] = val;
    }
    __syncthreads();
}

// Inverse NTT (parallelized across all threads)

__device__ __forceinline__ void ntt_inverse_kernel_parallel(
    uint64_t* operand,
    const NTTParams* params,
    uint32_t coeff_mod,
    uint32_t tid,
    uint32_t block_size
) {
    const uint32_t n = params->poly_len;
    const uint64_t* inverse_table = params->inverse_table + coeff_mod * n;
    const uint64_t* inverse_table_prime = params->inverse_prime_table + coeff_mod * n;
    const uint64_t modulus = params->moduli[coeff_mod];
    const uint64_t two_times_modulus = 2 * modulus;

    // Reverse butterfly loops (unrolled for register optimization)
    #pragma unroll
    for (int mm = params->log2_poly_len - 1; mm >= 0; mm--) {
        const uint32_t h = 1 << mm;
        const uint32_t t = n >> (mm + 1);
        const uint32_t total_butterflies = h * t;

        // Parallelize across butterflies
        for (uint32_t butterfly_idx = tid; butterfly_idx < total_butterflies; butterfly_idx += block_size) {
            const uint32_t i = butterfly_idx / t;
            const uint32_t j = butterfly_idx % t;

            const uint64_t w = inverse_table[h + i];
            const uint64_t w_prime = inverse_table_prime[h + i];

            const uint32_t idx1 = i * 2 * t + j;
            const uint32_t idx2 = idx1 + t;

            const uint64_t x = operand[idx1];
            const uint64_t y = operand[idx2];

            const uint64_t t_tmp = two_times_modulus - y + x;
            const uint64_t curr_x = x + y - (two_times_modulus * (((x << 1) >= t_tmp) ? 1 : 0));
            const uint64_t h_tmp = (t_tmp * w_prime) >> 32;

            const uint64_t res_x = (curr_x + (modulus * (t_tmp & 1))) >> 1;
            const uint64_t res_y = w * t_tmp - h_tmp * modulus;

            operand[idx1] = res_x;
            operand[idx2] = res_y;
        }
        __syncthreads();
    }

    // Final reduction (parallelized)
    for (uint32_t i = tid; i < n; i += block_size) {
        uint64_t val = operand[i];
        if (val >= two_times_modulus) {
            val -= two_times_modulus;
        }
        if (val >= modulus) {
            val -= modulus;
        }
        operand[i] = val;
    }
    __syncthreads();
}

// ═══════════════════════════════════════════════════════════════
// Alt NTT for large moduli (crt_count=1, modulus ≈ 2^32)
// Matches spiral-rs ntt/alt.rs — uses __umul64hi for 128-bit Barrett
// ═══════════════════════════════════════════════════════════════

__device__ __forceinline__ void ntt_forward_alt_parallel(
    uint64_t* operand,
    const NTTParams* params,
    uint32_t coeff_mod,
    uint32_t tid,
    uint32_t block_size
) {
    const uint32_t n = params->poly_len;
    const uint32_t log_n = params->log2_poly_len;
    const uint64_t* forward_table = params->forward_table + coeff_mod * n;
    const uint64_t* forward_prime_table = params->forward_prime_table + coeff_mod * n;
    const uint64_t modulus_small = params->moduli[coeff_mod];
    const uint64_t two_times_modulus_small = 2 * modulus_small;

    #pragma unroll
    for (uint32_t mm = 0; mm < log_n; mm++) {
        const uint32_t m = 1 << mm;
        const uint32_t t = n >> (mm + 1);
        const uint32_t total_butterflies = m * t;

        for (uint32_t butterfly_idx = tid; butterfly_idx < total_butterflies; butterfly_idx += block_size) {
            const uint32_t i = butterfly_idx / t;
            const uint32_t j = butterfly_idx % t;
            const uint64_t w = forward_table[m + i];
            const uint64_t w_prime = forward_prime_table[m + i];
            const uint32_t idx1 = i * 2 * t + j;
            const uint32_t idx2 = idx1 + t;
            const uint64_t x = operand[idx1];
            const uint64_t y = operand[idx2];

            uint64_t curr_x = x - (two_times_modulus_small * ((x >= two_times_modulus_small) ? 1ULL : 0ULL));
            // q_tmp = ((y as u128) * (w_prime as u128)) >> 64
            const uint64_t q_tmp = __umul64hi(y, w_prime);
            // q_new = ((w as u128) * (y as u128) - q_tmp * (mod as u128)) % mod
            uint64_t wy_lo = w * y;
            uint64_t wy_hi = __umul64hi(w, y);
            uint64_t qt_lo = q_tmp * modulus_small;
            uint64_t qt_hi = __umul64hi(q_tmp, modulus_small);
            // 128-bit sub: (wy_hi:wy_lo) - (qt_hi:qt_lo)
            uint64_t diff_lo = wy_lo - qt_lo;
            uint64_t borrow = (wy_lo < qt_lo) ? 1ULL : 0ULL;
            uint64_t diff_hi = wy_hi - qt_hi - borrow;
            // diff = (diff_hi:diff_lo) which should be in [0, 2*mod)
            // Since diff < 2*mod < 2^34 and diff_hi should be 0:
            uint64_t q_new = diff_lo;
            if (q_new >= modulus_small) q_new -= modulus_small;

            operand[idx1] = curr_x + q_new;
            operand[idx2] = curr_x + (two_times_modulus_small - q_new);
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < n; i += block_size) {
        uint64_t val = operand[i];
        if (val >= two_times_modulus_small) val -= two_times_modulus_small;
        if (val >= modulus_small) val -= modulus_small;
        operand[i] = val;
    }
    __syncthreads();
}

__device__ __forceinline__ void ntt_inverse_alt_parallel(
    uint64_t* operand,
    const NTTParams* params,
    uint32_t coeff_mod,
    uint32_t tid,
    uint32_t block_size
) {
    const uint32_t n = params->poly_len;
    const uint64_t* inverse_table = params->inverse_table + coeff_mod * n;
    const uint64_t* inverse_table_prime = params->inverse_prime_table + coeff_mod * n;
    const uint64_t modulus = params->moduli[coeff_mod];
    const uint64_t two_times_modulus = 2 * modulus;

    for (int mm = params->log2_poly_len - 1; mm >= 0; mm--) {
        const uint32_t h = 1 << mm;
        const uint32_t t = n >> (mm + 1);
        const uint32_t total_butterflies = h * t;

        for (uint32_t butterfly_idx = tid; butterfly_idx < total_butterflies; butterfly_idx += block_size) {
            const uint32_t i = butterfly_idx / t;
            const uint32_t j = butterfly_idx % t;
            const uint64_t w = inverse_table[h + i];
            const uint64_t w_prime = inverse_table_prime[h + i];
            const uint32_t idx1 = i * 2 * t + j;
            const uint32_t idx2 = idx1 + t;
            const uint64_t x = operand[idx1];
            const uint64_t y = operand[idx2];

            const uint64_t t_tmp = two_times_modulus - y + x;
            const uint64_t curr_x = x + y - (two_times_modulus * (((x << 1) >= t_tmp) ? 1ULL : 0ULL));
            // u128 Barrett: h_tmp = (t_tmp * w_prime) >> 64
            const uint64_t h_tmp = __umul64hi(t_tmp, w_prime);
            const uint64_t res_x = (curr_x + (modulus * (t_tmp & 1))) >> 1;
            // Exact match of Rust alt.rs inverse butterfly:
            // res_y = ((w as u128) * (t_tmp as u128) - (h_tmp * mod as u128)) % mod
            uint64_t wt_lo = w * t_tmp;
            uint64_t wt_hi = __umul64hi(w, t_tmp);
            uint64_t hm_lo = h_tmp * modulus;
            uint64_t hm_hi = __umul64hi(h_tmp, modulus);
            uint64_t d_lo = wt_lo - hm_lo;
            uint64_t d_borrow = (wt_lo < hm_lo) ? 1ULL : 0ULL;
            uint64_t d_hi = wt_hi - hm_hi - d_borrow;
            uint64_t res_y = d_lo;
            if (res_y >= modulus) res_y -= modulus;

            operand[idx1] = res_x;
            operand[idx2] = res_y;
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < n; i += block_size) {
        uint64_t val = operand[i];
        if (val >= two_times_modulus) val -= two_times_modulus;
        if (val >= modulus) val -= modulus;
        operand[i] = val;
    }
    __syncthreads();
}

// CRT Reconstruction (matches spiral-rs params.rs::crt_compose_2)

__device__ __forceinline__ uint64_t crt_compose_2(
    uint64_t x,  // First CRT residue
    uint64_t y,  // Second CRT residue
    const NTTParams* params
) {
    // val = x * mod1_inv_mod0 + y * mod0_inv_mod1  (128-bit)
    // Compute x * mod1_inv_mod0
    uint64_t prod1_lo = x * params->mod1_inv_mod0;
    uint64_t prod1_hi = __umul64hi(x, params->mod1_inv_mod0);

    // Compute y * mod0_inv_mod1
    uint64_t prod2_lo = y * params->mod0_inv_mod1;
    uint64_t prod2_hi = __umul64hi(y, params->mod0_inv_mod1);

    // Add: val = prod1 + prod2
    uint64_t val_lo = prod1_lo + prod2_lo;
    uint64_t carry = val_lo < prod1_lo ? 1 : 0;
    uint64_t val_hi = prod1_hi + prod2_hi + carry;

    // Barrett reduction for 128-bit value
    return barrett_reduction_u128(
        val_lo, val_hi,
        params->modulus,
        params->barrett_cr_0_modulus,
        params->barrett_cr_1_modulus
    );
}

// Pointwise multiplication in NTT domain

__device__ __forceinline__ void pointwise_multiply(
    uint64_t* result,
    const uint64_t* a,
    const uint64_t* b,
    uint32_t poly_len,
    uint64_t modulus,
    uint64_t barrett_cr
) {
    for (uint32_t i = 0; i < poly_len; i++) {
        uint64_t prod = a[i] * b[i];
        result[i] = barrett_raw_u64(prod, barrett_cr, modulus);
    }
}

#endif
