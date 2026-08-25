// Standalone NTT benchmark: SandwichPIR's spiral-rs derived alt-form NTT
// vs GPU-NTT (Ozcan, Apache 2.0).
//
// Parameters: d=2048, Q=4294955009, negacyclic (X^N+1).
//
// Both NTTs use the same Q and equivalent root tables (each generated in
// the format that NTT expects). Round-trip correctness is checked per NTT
// implementation independently — cross-implementation match is not expected
// because spiral-rs alt path uses a non-standard butterfly indexing.
//
// Build:
//   nvcc -O3 -arch=sm_80 \
//     -I src/cuda -I ../GPU-NTT/src/include \
//     benches/ntt_bench.cu \
//     ../GPU-NTT/build/src/libntt-1.0.a \
//     -o benches/build/ntt_bench
//
// Run:  ./benches/build/ntt_bench [max_batch] [iters]

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>

#include "common/ntt.cuh"

#include "gpuntt/ntt_merge/ntt.cuh"
#include "gpuntt/common/nttparameters.cuh"

using namespace gpuntt;

// ───────────────────────── Host-side number theory ─────────────────────────

static uint64_t mod_mul_u64(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)((__uint128_t)a * b % m);
}

static uint64_t mod_pow_u64(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t res = 1; base %= mod;
    while (exp) {
        if (exp & 1) res = mod_mul_u64(res, base, mod);
        base = mod_mul_u64(base, base, mod);
        exp >>= 1;
    }
    return res;
}

// floor(2^128 / Q), high 64 bits — matches spiral-rs `get_barrett_crs`.
static uint64_t get_barrett_cr1(uint64_t Q) {
    __uint128_t hi_bit = (__uint128_t)1 << 127;
    __uint128_t q1 = hi_bit / Q;
    __uint128_t r1 = hi_bit - q1 * Q;
    __uint128_t q_total = 2 * q1 + ((2 * r1) >= Q ? 1 : 0);
    return (uint64_t)(q_total >> 64);
}

// Find a primitive (2N)-th root of unity mod Q for X^N+1 (negacyclic).
// Returns ψ such that ψ^N ≡ -1 (mod Q).
static uint64_t find_primitive_2n_root(uint64_t Q, uint64_t two_n) {
    if ((Q - 1) % two_n != 0) { fprintf(stderr, "2N does not divide Q-1\n"); exit(1); }
    uint64_t exp = (Q - 1) / two_n;
    for (uint64_t g = 2; g < 1000000; g++) {
        uint64_t r = mod_pow_u64(g, exp, Q);
        // r has order dividing 2N. For r to be a primitive 2N-th root,
        // r^N must equal Q-1 (≡ -1).
        if (mod_pow_u64(r, two_n / 2, Q) == Q - 1) return r;
    }
    fprintf(stderr, "no primitive 2N-th root found\n"); exit(1);
}

static uint64_t mod_inverse(uint64_t a, uint64_t m) {
    // Extended GCD via Fermat's little (m must be prime).
    return mod_pow_u64(a, m - 2, m);
}

static uint64_t reverse_bits_u64(uint64_t v, int bits) {
    uint64_t r = 0;
    for (int i = 0; i < bits; i++) { r = (r << 1) | (v & 1); v >>= 1; }
    return r;
}

// Spiral-rs alt-form: root_powers[bitrev(i, log2_n)] = root^i  for i in [1, n).
// root_powers[0] = 1.
static std::vector<uint64_t> powers_of_primitive_root(uint64_t root, uint64_t Q, int log2_n) {
    int n = 1 << log2_n;
    std::vector<uint64_t> out(n, 0);
    uint64_t power = root;
    for (int i = 1; i < n; i++) {
        int idx = (int)reverse_bits_u64(i, log2_n);
        out[idx] = power;
        power = mod_mul_u64(power, root, Q);
    }
    out[0] = 1;
    return out;
}

// Harvey/Shoup-style: scaled[i] = floor((root_powers[i] << 64) / Q).
static std::vector<uint64_t> scale_powers_u64(uint64_t Q, const std::vector<uint64_t>& in) {
    std::vector<uint64_t> out(in.size(), 0);
    for (size_t i = 0; i < in.size(); i++) {
        __uint128_t wide = (__uint128_t)in[i] << 64;
        out[i] = (uint64_t)(wide / Q);
    }
    return out;
}

// (a / 2) mod m, where m is odd.
static uint64_t div2_uint_mod(uint64_t a, uint64_t m) {
    if (a & 1) return (a + m) >> 1;
    return a >> 1;
}

// ───────────────────────── Our NTT (host setup) ─────────────────────────

// Allocate device-side NTTParams (spiral-rs alt path) for one modulus Q, poly_len N.
// The struct is returned by value and contains device pointers.
static NTTParams build_our_ntt_params(uint64_t Q, int log2_N) {
    int N = 1 << log2_N;
    NTTParams p{};
    p.poly_len = (uint32_t)N;
    p.log2_poly_len = (uint32_t)log2_N;
    p.crt_count = 1;
    p.modulus = Q;

    // Build forward + inverse tables in spiral-rs alt format.
    uint64_t psi = find_primitive_2n_root(Q, (uint64_t)(2 * N));
    uint64_t inv_psi = mod_inverse(psi, Q);
    auto fwd = powers_of_primitive_root(psi, Q, log2_N);
    auto fwd_p = scale_powers_u64(Q, fwd);
    auto inv = powers_of_primitive_root(inv_psi, Q, log2_N);
    // spiral-rs build_ntt_tables_alt halves the inverse powers
    for (int i = 0; i < N; i++) inv[i] = div2_uint_mod(inv[i], Q);
    auto inv_p = scale_powers_u64(Q, inv);

    uint64_t cr1 = get_barrett_cr1(Q);
    uint64_t mods[1] = {Q};
    uint64_t crs[1]  = {cr1};

    auto upload = [](uint64_t** dst, const void* src, size_t bytes) {
        cudaMalloc((void**)dst, bytes);
        cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice);
    };
    upload(&p.moduli,               mods,  sizeof(uint64_t));
    upload(&p.barrett_cr,           crs,   sizeof(uint64_t));
    upload(&p.forward_table,        fwd.data(),   N * sizeof(uint64_t));
    upload(&p.forward_prime_table,  fwd_p.data(), N * sizeof(uint64_t));
    upload(&p.inverse_table,        inv.data(),   N * sizeof(uint64_t));
    upload(&p.inverse_prime_table,  inv_p.data(), N * sizeof(uint64_t));
    return p;
}

static void free_our_ntt_params(NTTParams& p) {
    cudaFree(p.moduli); cudaFree(p.barrett_cr);
    cudaFree(p.forward_table);  cudaFree(p.forward_prime_table);
    cudaFree(p.inverse_table);  cudaFree(p.inverse_prime_table);
}

// One block per polynomial, 1024 threads.  Operand stays in shared memory.
__global__ void our_ntt_forward_kernel(uint64_t* batch_data, NTTParams params, int batch) {
    int b = blockIdx.x; if (b >= batch) return;
    int tid = threadIdx.x;
    int n = (int)params.poly_len;
    extern __shared__ uint64_t buf[];
    for (int i = tid; i < n; i += blockDim.x) buf[i] = batch_data[(size_t)b * n + i];
    __syncthreads();
    ntt_forward_alt_parallel(buf, &params, 0, tid, blockDim.x);
    for (int i = tid; i < n; i += blockDim.x) batch_data[(size_t)b * n + i] = buf[i];
}

__global__ void our_ntt_inverse_kernel(uint64_t* batch_data, NTTParams params, int batch) {
    int b = blockIdx.x; if (b >= batch) return;
    int tid = threadIdx.x;
    int n = (int)params.poly_len;
    extern __shared__ uint64_t buf[];
    for (int i = tid; i < n; i += blockDim.x) buf[i] = batch_data[(size_t)b * n + i];
    __syncthreads();
    ntt_inverse_alt_parallel(buf, &params, 0, tid, blockDim.x);
    for (int i = tid; i < n; i += blockDim.x) batch_data[(size_t)b * n + i] = buf[i];
}

// ───────────────────────── Bench helpers ─────────────────────────

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

static void fill_random(std::vector<uint64_t>& v, uint64_t Q) {
    uint64_t s = 0xC0FFEE0123456789ULL;
    for (auto& x : v) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        x = s % Q;
    }
}

template <typename Fn>
static float time_loop(int iters, Fn fn) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    // warm-up
    fn(); CK(cudaDeviceSynchronize());
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) fn();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / iters;
}

// ───────────────────────── Main ─────────────────────────

int main(int argc, char** argv) {
    int max_batch = 32768;
    int iters     = 5;
    if (argc >= 2) max_batch = atoi(argv[1]);
    if (argc >= 3) iters     = atoi(argv[2]);

    constexpr int LOGN = 11;
    constexpr int N    = 1 << LOGN;
    constexpr uint64_t Q = 4294955009ULL;

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    printf("Device %d: %s, CC %d.%d\n", dev, prop.name, prop.major, prop.minor);
    printf("Params: N=%d, Q=%llu, X^N+1, batches up to %d, %d iters/measure\n\n",
           N, (unsigned long long)Q, max_batch, iters);

    // ── Our NTT setup ──
    NTTParams our_params = build_our_ntt_params(Q, LOGN);
    // ── GPU-NTT setup (using same Q via NTTFactors) ──
    uint64_t psi = find_primitive_2n_root(Q, (uint64_t)(2 * N));
    uint64_t omega = mod_mul_u64(psi, psi, Q);  // primitive N-th root (unused for X^N+1, but required by NTTFactors)
    NTTFactors<Data64> factors(Modulus64(Q), omega, psi);
    NTTParameters<Data64> gpu_params(LOGN, factors, ReductionPolynomial::X_N_plus);

    // GPU-NTT root tables (forward + inverse), bit-reversed
    auto gpu_fwd = gpu_params.gpu_root_of_unity_table_generator(gpu_params.forward_root_of_unity_table);
    auto gpu_inv = gpu_params.gpu_root_of_unity_table_generator(gpu_params.inverse_root_of_unity_table);
    Root<Data64> *gpu_fwd_dev, *gpu_inv_dev;
    CK(cudaMalloc(&gpu_fwd_dev, gpu_fwd.size() * sizeof(Root<Data64>)));
    CK(cudaMalloc(&gpu_inv_dev, gpu_inv.size() * sizeof(Root<Data64>)));
    CK(cudaMemcpy(gpu_fwd_dev, gpu_fwd.data(), gpu_fwd.size() * sizeof(Root<Data64>), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gpu_inv_dev, gpu_inv.data(), gpu_inv.size() * sizeof(Root<Data64>), cudaMemcpyHostToDevice));

    Modulus<Data64>*   gpu_mod_dev;
    Ninverse<Data64>*  gpu_ninv_dev;
    Modulus<Data64>    gpu_mod_h[1] = { Modulus64(Q) };
    Ninverse<Data64>   gpu_ninv_h[1] = { gpu_params.n_inv };
    CK(cudaMalloc(&gpu_mod_dev,  sizeof(Modulus<Data64>)));
    CK(cudaMalloc(&gpu_ninv_dev, sizeof(Ninverse<Data64>)));
    CK(cudaMemcpy(gpu_mod_dev,  gpu_mod_h,  sizeof(Modulus<Data64>),  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gpu_ninv_dev, gpu_ninv_h, sizeof(Ninverse<Data64>), cudaMemcpyHostToDevice));

    ntt_rns_configuration<Data64> cfg_fwd = {
        .n_power = LOGN,
        .ntt_type = FORWARD,
        .ntt_layout = PerPolynomial,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = nullptr,
        .stream = 0,
    };
    ntt_rns_configuration<Data64> cfg_inv = {
        .n_power = LOGN,
        .ntt_type = INVERSE,
        .ntt_layout = PerPolynomial,
        .reduction_poly = ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = gpu_ninv_dev,
        .stream = 0,
    };

    // ── Buffers ──
    size_t max_elems = (size_t)max_batch * N;
    uint64_t* d_ours; CK(cudaMalloc(&d_ours, max_elems * sizeof(uint64_t)));
    Data64*   d_gpu;  CK(cudaMalloc(&d_gpu,  max_elems * sizeof(Data64)));

    std::vector<uint64_t> h_input(max_elems);
    fill_random(h_input, Q);
    std::vector<uint64_t> h_orig = h_input;

    // ── Bench loop ──
    int batches[] = {1, 1024, max_batch};
    int n_batches = (max_batch == 1024 || max_batch == 1) ? 2 : 3;
    if (max_batch == 1) n_batches = 1;

    //  fwd / inv : total wall-time (ms) of one kernel launch processing the
    //              entire batch (forward / inverse NTT respectively).
    //  per/poly  : amortized round-trip latency per polynomial in microseconds:
    //              (fwd + inv) * 1000 / batch.  Lower = higher throughput.
    //  round-trip: forward∘inverse equality check (each impl independently).
    printf("         |        Ours (spiral-rs alt, lazy)      |     GPU-NTT (Ozcan, strict Barrett)    | round-trip\n");
    printf(" batch   |  fwd ms     inv ms     per/poly us     |  fwd ms     inv ms     per/poly us     | ours/ gpu\n");
    printf("---------+----------------------------------------+----------------------------------------+-----------\n");

    for (int bi = 0; bi < n_batches; bi++) {
        int B = batches[bi];
        if (B > max_batch) continue;
        size_t elems = (size_t)B * N;

        // Reload input for our NTT
        CK(cudaMemcpy(d_ours, h_input.data(), elems * sizeof(uint64_t), cudaMemcpyHostToDevice));

        size_t smem = N * sizeof(uint64_t);
        float ours_fwd = time_loop(iters, [&]() {
            our_ntt_forward_kernel<<<B, 1024, smem>>>(d_ours, our_params, B);
        });
        float ours_inv = time_loop(iters, [&]() {
            our_ntt_inverse_kernel<<<B, 1024, smem>>>(d_ours, our_params, B);
        });

        // Round-trip check for our NTT (forward then inverse on fresh input)
        CK(cudaMemcpy(d_ours, h_input.data(), elems * sizeof(uint64_t), cudaMemcpyHostToDevice));
        our_ntt_forward_kernel<<<B, 1024, smem>>>(d_ours, our_params, B);
        our_ntt_inverse_kernel<<<B, 1024, smem>>>(d_ours, our_params, B);
        CK(cudaDeviceSynchronize());
        std::vector<uint64_t> h_back(elems);
        CK(cudaMemcpy(h_back.data(), d_ours, elems * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        bool ours_ok = true;
        for (size_t i = 0; i < elems && ours_ok; i++) if (h_back[i] != h_input[i]) ours_ok = false;

        // Reload input for GPU-NTT
        CK(cudaMemcpy(d_gpu, h_input.data(), elems * sizeof(Data64), cudaMemcpyHostToDevice));

        float gpu_fwd_ms = time_loop(iters, [&]() {
            GPU_NTT_Inplace(d_gpu, gpu_fwd_dev, gpu_mod_dev, cfg_fwd, B, 1);
        });
        float gpu_inv_ms = time_loop(iters, [&]() {
            GPU_INTT_Inplace(d_gpu, gpu_inv_dev, gpu_mod_dev, cfg_inv, B, 1);
        });

        // Round-trip check for GPU-NTT
        CK(cudaMemcpy(d_gpu, h_input.data(), elems * sizeof(Data64), cudaMemcpyHostToDevice));
        GPU_NTT_Inplace(d_gpu, gpu_fwd_dev, gpu_mod_dev, cfg_fwd, B, 1);
        GPU_INTT_Inplace(d_gpu, gpu_inv_dev, gpu_mod_dev, cfg_inv, B, 1);
        CK(cudaDeviceSynchronize());
        std::vector<Data64> g_back(elems);
        CK(cudaMemcpy(g_back.data(), d_gpu, elems * sizeof(Data64), cudaMemcpyDeviceToHost));
        bool gpu_ok = true;
        for (size_t i = 0; i < elems && gpu_ok; i++) if (g_back[i] != h_input[i]) gpu_ok = false;

        float ours_per_ntt_us = (ours_fwd + ours_inv) * 1000.0f / B;
        float gpu_per_ntt_us  = (gpu_fwd_ms + gpu_inv_ms) * 1000.0f / B;

        printf(" %-7d | %9.2f  %9.2f  %15.2f      | %9.2f  %9.2f  %15.2f      | %4s/%4s\n",
               B, ours_fwd, ours_inv, ours_per_ntt_us,
               gpu_fwd_ms, gpu_inv_ms, gpu_per_ntt_us,
               ours_ok ? "OK" : "FAIL", gpu_ok ? "OK" : "FAIL");
    }

    cudaFree(d_ours); cudaFree(d_gpu);
    cudaFree(gpu_fwd_dev); cudaFree(gpu_inv_dev);
    cudaFree(gpu_mod_dev); cudaFree(gpu_ninv_dev);
    free_our_ntt_params(our_params);
    return 0;
}
