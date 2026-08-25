/**
 * SandwichPIR Tensor Core Packing — Single Modulus
 *
 * Adapted from inspiring/tc_packing.cu for crt_count=1.
 * M-reformulation GEMM: result = Y · M, no permutation gathering.
 *
 * Key differences from inspiring/ version:
 * - Single modulus Q ≈ 2^32 (no CRT, no moduli[1])
 * - 4 byte slices per operand (not 8)
 * - 16 CUTLASS calls (not 32)
 * - No CRT compose in finalize/post-process
 *
 * GEMM shape: [B × tN] × [tN × ρN] → [B × ρN]
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

#include "common/ntt.cuh"
#include "common/log.cuh"
#include "sandwich/tc_packing.cuh"
#include "sandwich/packing.cuh"

// ═══════════════════════════════════════════════════════════════
// CUTLASS GEMM types (same as inspiring/tc_packing.cu)
// ═══════════════════════════════════════════════════════════════

using SwTcGemm_Sm80 = cutlass::gemm::device::Gemm<
    uint8_t, cutlass::layout::RowMajor,
    uint8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 3
>;

using SwTcGemm_Sm75 = cutlass::gemm::device::Gemm<
    uint8_t, cutlass::layout::RowMajor,
    uint8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 2
>;

using SwSimtGemm_Sm50 = cutlass::gemm::device::Gemm<
    uint8_t, cutlass::layout::RowMajor,
    uint8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 1, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 2
>;

static cutlass::Status sw_packing_gemm(int gpu_tier,
    int M, int N, int K,
    const uint8_t* A, int lda, const uint8_t* B, int ldb,
    int32_t* C, int ldc, int32_t alpha, int32_t beta,
    cudaStream_t stream = 0)
{
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    if (gpu_tier >= 2) {
        SwTcGemm_Sm80::Arguments args{problem_size, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta}, 1};
        SwTcGemm_Sm80 op; auto s = op.initialize(args, nullptr, stream); if (s != cutlass::Status::kSuccess) return s; return op(stream);
    } else if (gpu_tier == 1) {
        SwTcGemm_Sm75::Arguments args{problem_size, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta}, 1};
        SwTcGemm_Sm75 op; auto s = op.initialize(args, nullptr, stream); if (s != cutlass::Status::kSuccess) return s; return op(stream);
    } else {
        SwSimtGemm_Sm50::Arguments args{problem_size, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta}, 1};
        SwSimtGemm_Sm50 op; auto s = op.initialize(args, nullptr, stream); if (s != cutlass::Status::kSuccess) return s; return op(stream);
    }
}

// ═══════════════════════════════════════════════════════════════
// GPU Kernels (single CRT)
// ═══════════════════════════════════════════════════════════════

__global__ void sw_build_M_tile_kernel(
    uint64_t* __restrict__ d_tile,
    const uint64_t* __restrict__ d_bold_t,
    const uint64_t* __restrict__ d_bold_t_bar,
    const uint32_t* __restrict__ d_tables,
    const uint32_t* __restrict__ d_gen_pows,
    size_t num_iter, size_t t_exp_left, size_t poly_len,
    size_t output_idx, size_t K_gemm)
{
    size_t z = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= poly_len) return;
    size_t k = blockIdx.z, o = output_idx, D = num_iter * t_exp_left;

    for (size_t i = 0; i < num_iter; i++) {
        uint32_t gp = d_gen_pows[i];
        uint32_t z_prime     = d_tables[((gp - 1) / 2) * poly_len + z];
        uint32_t z_prime_bar = d_tables[((2 * (uint32_t)poly_len - gp - 1) / 2) * poly_len + z];

        size_t bold_t_idx = o * D * poly_len + (i * t_exp_left + k) * poly_len + z;
        uint32_t t_val  = (uint32_t)(d_bold_t[bold_t_idx] & 0xFFFFFFFF);
        uint32_t tb_val = (uint32_t)(d_bold_t_bar[bold_t_idx] & 0xFFFFFFFF);

        atomicAdd((unsigned long long*)&d_tile[(k * poly_len + z_prime) + z * K_gemm], (unsigned long long)t_val);
        atomicAdd((unsigned long long*)&d_tile[(k * poly_len + z_prime_bar) + z * K_gemm], (unsigned long long)tb_val);
    }
}

__global__ void sw_reduce_byte_decompose_tile(
    uint8_t* __restrict__ out, const uint64_t* __restrict__ in,
    size_t tile_elems, size_t full_elems, size_t tile_offset,
    uint64_t mod, uint64_t barrett_cr)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tile_elems) return;
    uint32_t v = (uint32_t)barrett_raw_u64(in[idx], barrett_cr, mod);
    size_t out_idx = tile_offset + idx;
    out[0 * full_elems + out_idx] = (uint8_t)(v & 0xFF);
    out[1 * full_elems + out_idx] = (uint8_t)((v >> 8) & 0xFF);
    out[2 * full_elems + out_idx] = (uint8_t)((v >> 16) & 0xFF);
    out[3 * full_elems + out_idx] = (uint8_t)((v >> 24) & 0xFF);
}

// Byte-decompose with padded output stride: input is B×K, output has Md×K stride per slice.
// Handles Md > B alignment for batched GEMM.
__global__ void sw_byte_decompose_padded_kernel(
    uint8_t* __restrict__ out, const uint64_t* __restrict__ in,
    size_t B, size_t K, size_t Md)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * K) return;
    size_t row = idx / K;  // batch index (0..B-1)
    size_t col = idx % K;  // position within K
    uint32_t v = (uint32_t)(in[idx] & 0xFFFFFFFF);
    size_t stride = Md * K;  // padded stride per byte slice
    size_t out_idx = row * K + col;
    out[0 * stride + out_idx] = (uint8_t)(v & 0xFF);
    out[1 * stride + out_idx] = (uint8_t)((v >> 8) & 0xFF);
    out[2 * stride + out_idx] = (uint8_t)((v >> 16) & 0xFF);
    out[3 * stride + out_idx] = (uint8_t)((v >> 24) & 0xFF);
}

__global__ void sw_byte_decompose_kernel(
    uint8_t* __restrict__ out, const uint64_t* __restrict__ in, size_t count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    uint32_t v = (uint32_t)(in[idx] & 0xFFFFFFFF);
    out[0 * count + idx] = (uint8_t)(v & 0xFF);
    out[1 * count + idx] = (uint8_t)((v >> 8) & 0xFF);
    out[2 * count + idx] = (uint8_t)((v >> 16) & 0xFF);
    out[3 * count + idx] = (uint8_t)((v >> 24) & 0xFF);
}

// Batched accumulate: reads 4×4 panels from the big GEMM output G_big (4*Md × 4*Ng),
// applies shift sm[i+j] for panel (i,j), accumulates into result (Md × Ng).
// G_big is ColumnMajor with ldc=4*Md. Panel (i,j) starts at row i*Md, col j*Ng.
__global__ void sw_accumulate_batched_kernel(
    uint64_t* __restrict__ result, const int32_t* __restrict__ G_big,
    size_t Md, size_t Ng, size_t ldc,
    uint64_t sm0, uint64_t sm1, uint64_t sm2, uint64_t sm3,
    uint64_t sm4, uint64_t sm5, uint64_t sm6)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Md * Ng) return;
    size_t row = idx % Md;   // batch index within panel
    size_t col = idx / Md;   // output position

    uint64_t acc = 0;
    // 16 panels: (i,j) for i,j in 0..3, shift = sm[i+j]
    const uint64_t sm[7] = {sm0, sm1, sm2, sm3, sm4, sm5, sm6};
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int32_t g = G_big[(i * Md + row) + (size_t)(j * Ng + col) * ldc];
            acc += (uint64_t)(uint32_t)g * sm[i + j];
        }
    }
    result[idx] = acc;
}

__global__ void sw_accumulate_shift_kernel(
    uint64_t* __restrict__ result, const int32_t* __restrict__ G,
    uint64_t shift_const, size_t count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    result[idx] += (uint64_t)(uint32_t)G[idx] * shift_const;
}

__global__ void sw_finalize_to_scratch_kernel(
    uint64_t* __restrict__ d_scratch,
    const uint64_t* __restrict__ d_result,
    const uint64_t* __restrict__ d_bold_t_hat,
    const uint64_t* __restrict__ d_z_body,
    size_t result_stride, size_t num_outputs, size_t poly_len,
    size_t t_exp_left, size_t N_gemm,
    size_t scratch_stride, size_t inspir_spo,
    uint64_t mod0, uint64_t barrett_cr0)
{
    size_t z = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= poly_len) return;
    size_t o = blockIdx.y, c = blockIdx.z;

    // Barrett reduce GEMM result first (keep acc < mod for safe accumulation)
    uint64_t acc = barrett_raw_u64(
        d_result[c + (o * poly_len + z) * result_stride], barrett_cr0, mod0);
    size_t z_body_stride = t_exp_left * poly_len;
    for (size_t k = 0; k < t_exp_left; k++) {
        uint32_t t_v = (uint32_t)d_bold_t_hat[o * t_exp_left * poly_len + k * poly_len + z];
        uint32_t z_v = (uint32_t)d_z_body[c * z_body_stride + k * poly_len + z];
        // Reduce product individually (u32×u32 can be ~2^64 for Q≈2^32)
        acc += barrett_raw_u64((uint64_t)t_v * z_v, barrett_cr0, mod0);
        if (acc >= mod0) acc -= mod0;
    }

    d_scratch[c * scratch_stride + o * inspir_spo + z] = acc;
}

// ═══════════════════════════════════════════════════════════════
// Cross-product table (4×4 byte decomposition, 16 products)
// ═══════════════════════════════════════════════════════════════

struct SwCP { int i, j; };
static const SwCP SW_CP[16] = {
    {0,0}, {0,1},{1,0}, {0,2},{1,1},{2,0}, {0,3},{1,2},{2,1},{3,0},
    {1,3},{2,2},{3,1}, {2,3},{3,2}, {3,3}
};
static const int SW_SS[8] = {0, 1, 3, 6, 10, 13, 15, 16};

// ═══════════════════════════════════════════════════════════════
// Extern "C" API
// ═══════════════════════════════════════════════════════════════

extern "C" {

void* sw_tc_packing_init(
    uint64_t* d_bold_t, uint64_t* d_bold_t_bar,
    const uint64_t* d_bold_t_hat, const uint64_t* d_a_hat,
    const uint32_t* d_tables, const uint32_t* d_gen_pows,
    size_t num_iter, size_t t_exp_left, size_t poly_len,
    size_t num_outputs, size_t max_batch_size,
    uint64_t rlwe_q_prime_1, uint64_t rlwe_q_prime_2,
    size_t response_bytes_per_output, NTTParams ntt_params, int gpu_tier)
{
    auto* ctx = new SwTcPackingContext();
    ctx->poly_len = poly_len; ctx->t_exp_left = t_exp_left;
    ctx->num_iter = num_iter; ctx->num_outputs = num_outputs;
    ctx->max_batch_size = max_batch_size;
    ctx->K_gemm = t_exp_left * poly_len; ctx->N_gemm = num_outputs * poly_len;
    ctx->gpu_tier = gpu_tier;
    ctx->rlwe_q_prime_1 = rlwe_q_prime_1; ctx->rlwe_q_prime_2 = rlwe_q_prime_2;
    ctx->response_bytes_per_output = response_bytes_per_output;
    {
        // q_1 is the mask modulus (rlwe_q_prime_2, large), q_2 is the body
        // modulus (rlwe_q_prime_1, small). Naming is historical.
        size_t q_1_bits = inspir_ceil_log2_u64(rlwe_q_prime_2);
        size_t q_2_bits = inspir_ceil_log2_u64(rlwe_q_prime_1);
        ctx->mask_bytes_per_output = (poly_len * q_1_bits + 7) / 8;
        ctx->body_bytes_per_output = (poly_len * q_2_bits + 7) / 8;
    }
    ctx->ntt_params = ntt_params;

    CUDA_ASSERT(cudaMemcpy(&ctx->mod0, ntt_params.moduli, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(&ctx->barrett_cr0, ntt_params.barrett_cr, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    size_t K = ctx->K_gemm, N = ctx->N_gemm, M_elems = K * N;

    SW_LOG("SwTcPacking init: M [%zu × %zu] = %.1f MiB\n", K, N, 4.0*M_elems/(double)(1ULL << 20));
    GpuTimer timer; timer.tic();

    uint8_t* d_M_combined;
    CUDA_ASSERT(cudaMalloc(&d_M_combined, 4 * M_elems));

    uint64_t* d_tile; size_t tile_elems = K * poly_len;
    CUDA_ASSERT(cudaMalloc(&d_tile, tile_elems * sizeof(uint64_t)));

    int thr = 256, blk = (poly_len + thr - 1) / thr;
    dim3 bg(blk, 1, (int)t_exp_left);
    int db = (tile_elems + thr - 1) / thr;

    for (size_t o = 0; o < num_outputs; o++) {
        CUDA_ASSERT(cudaMemset(d_tile, 0, tile_elems * sizeof(uint64_t)));
        sw_build_M_tile_kernel<<<bg, thr>>>(d_tile, d_bold_t, d_bold_t_bar,
            d_tables, d_gen_pows, num_iter, t_exp_left, poly_len, o, K);
        CUDA_ASSERT(cudaGetLastError());
        sw_reduce_byte_decompose_tile<<<db, thr>>>(d_M_combined, d_tile, tile_elems,
            M_elems, o * poly_len * K, ctx->mod0, ctx->barrett_cr0);
        CUDA_ASSERT(cudaGetLastError());
    }
    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaFree(d_tile));
    if (d_bold_t) CUDA_ASSERT(cudaFree(d_bold_t));
    if (d_bold_t_bar) CUDA_ASSERT(cudaFree(d_bold_t_bar));

    for (int b = 0; b < 4; b++) ctx->d_M_bytes[b] = d_M_combined + (size_t)b * M_elems;

    size_t th = num_outputs * t_exp_left * poly_len;
    CUDA_ASSERT(cudaMalloc(&ctx->d_bold_t_hat, th * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMemcpy(ctx->d_bold_t_hat, d_bold_t_hat, th * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
    size_t ah = num_outputs * poly_len;
    CUDA_ASSERT(cudaMalloc(&ctx->d_a_hat, ah * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMemcpy(ctx->d_a_hat, d_a_hat, ah * sizeof(uint64_t), cudaMemcpyDeviceToDevice));

    int align = (gpu_tier >= 1) ? 4 : 1;
    size_t pb = ((max_batch_size + align - 1) / align) * align;
    CUDA_ASSERT(cudaMalloc(&ctx->d_A_bytes_buf, 4 * pb * K));
    // Big GEMM output: (4*pb × 4*N) for the single batched GEMM
    CUDA_ASSERT(cudaMalloc(&ctx->d_G, 4 * pb * 4 * N * sizeof(int32_t)));
    CUDA_ASSERT(cudaMalloc(&ctx->d_result, pb * N * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMalloc(&ctx->d_scratch, max_batch_size * num_outputs * 4 * poly_len * sizeof(uint64_t)));

    SW_LOG("SwTcPacking init done in %.1f ms\n", timer.toc_ms());
    return ctx;
}

void sw_tc_packing_gemms(void* context, cudaStream_t stream,
    const uint64_t* d_y_body, const uint64_t* d_z_body, size_t batch_size)
{
    auto* ctx = (SwTcPackingContext*)context;
    if (!ctx || batch_size == 0) return;
    size_t N=ctx->poly_len, K=ctx->K_gemm, Ng=ctx->N_gemm, B=batch_size, rho=ctx->num_outputs, t=ctx->t_exp_left;
    int align=(ctx->gpu_tier>=1)?4:1, Md=((int)B+align-1)/align*align;
    size_t BN=(size_t)Md*Ng;

    // Verbose sub-phase events
    bool verbose = sw_verbose();
    ctx->has_events = verbose;
    if (verbose) {
        CUDA_ASSERT(cudaEventCreate(&ctx->ev_decomp));
        CUDA_ASSERT(cudaEventCreate(&ctx->ev_gemm));
        CUDA_ASSERT(cudaEventCreate(&ctx->ev_accum));
        CUDA_ASSERT(cudaEventCreate(&ctx->ev_finalize));
    }

    // Byte-decompose Y with Md stride (zero-pad for alignment)
    if ((size_t)Md != B) {
        CUDA_ASSERT(cudaMemsetAsync(ctx->d_A_bytes_buf, 0, 4 * (size_t)Md * K, stream));
    }
    { size_t A_elems = B * K;
      int thr=256, blk=(A_elems+thr-1)/thr;
      sw_byte_decompose_padded_kernel<<<blk,thr,0,stream>>>(ctx->d_A_bytes_buf, d_y_body, B, K, Md);
      CUDA_ASSERT(cudaGetLastError()); }
    if (verbose) CUDA_ASSERT(cudaEventRecord(ctx->ev_decomp, stream));

    // Single big GEMM: (4*Md × 4*Ng × K)
    {
        auto s = sw_packing_gemm(ctx->gpu_tier,
            4*Md, 4*(int)Ng, (int)K,
            ctx->d_A_bytes_buf, (int)K,
            ctx->d_M_bytes[0], (int)K,
            ctx->d_G, 4*Md,
            1, 0, stream);
        if (s != cutlass::Status::kSuccess) {
            fprintf(stderr, "SW batched packing GEMM failed: %d\n", (int)s);
            abort();
        }
    }
    if (verbose) CUDA_ASSERT(cudaEventRecord(ctx->ev_gemm, stream));

    // Accumulate 4x4 byte-product panels into final u64 result.
    //
    // The GEMM output has a 4x4 grid of panels: panel (i,j) holds the product of
    // byte-slice i of the input (A) with byte-slice j of the M-matrix.
    // To reconstruct the full u32*u32 product mod Q, panel (i,j) needs to be
    // shifted left by 8*(i+j) bits, i.e. multiplied by 2^(8*(i+j)) mod Q.
    // The 7 distinct shift values correspond to i+j = 0,1,...,6.
    uint64_t sm[7];
    for (int s = 0; s < 7; s++) {
        uint64_t power = 1;
        for (int i = 0; i < 8 * s; i++)
            power = (power * 2) % ctx->mod0;
        sm[s] = power;  // sm[s] = 2^(8*s) mod Q
    }

    {
        int thr = 256;
        int blk = (BN + thr - 1) / thr;
        sw_accumulate_batched_kernel<<<blk, thr, 0, stream>>>(
            ctx->d_result, ctx->d_G,
            (size_t)Md, Ng, (size_t)(4 * Md),
            sm[0], sm[1], sm[2], sm[3], sm[4], sm[5], sm[6]);
        CUDA_ASSERT(cudaGetLastError());
    }
    if (verbose) CUDA_ASSERT(cudaEventRecord(ctx->ev_accum, stream));

    // Finalize: Barrett-reduce GEMM result and add bold_t_hat * z_body key-switch term.
    // Each output o gets 4*poly_len scratch slots (scratch_per_output), and per-client
    // scratch is laid out as rho contiguous blocks of scratch_per_output (scratch_stride).
    {
        size_t scratch_per_output = 4 * N;
        size_t scratch_stride = rho * scratch_per_output;
        int thr = 256;
        int bz = (N + thr - 1) / thr;
        dim3 grid(bz, (int)rho, (int)B);
        sw_finalize_to_scratch_kernel<<<grid, thr, 0, stream>>>(
            ctx->d_scratch, ctx->d_result,
            ctx->d_bold_t_hat, d_z_body,
            (size_t)Md, rho, N, t, Ng,
            scratch_stride, scratch_per_output,
            ctx->mod0, ctx->barrett_cr0);
        CUDA_ASSERT(cudaGetLastError());
    }
    if (verbose) CUDA_ASSERT(cudaEventRecord(ctx->ev_finalize, stream));
}

void sw_tc_packing_post_process(void* context, cudaStream_t stream,
    const uint64_t* d_intermediate, uint8_t* d_response_body_out, size_t batch_size)
{
    auto* ctx = (SwTcPackingContext*)context;
    if (!ctx || batch_size == 0) return;
    size_t N=ctx->poly_len, rho=ctx->num_outputs;
    dim3 pp((int)rho,(int)batch_size);
    sw_inspir_post_process<<<pp,1024,0,stream>>>(
        d_response_body_out, d_intermediate, ctx->d_scratch,
        rho, ctx->rlwe_q_prime_1, ctx->body_bytes_per_output,
        4*N, rho*4*N, rho*N, ctx->ntt_params);
    CUDA_ASSERT(cudaGetLastError());
}

// One-time: fill d_mask_template_out [num_outputs × mask_bytes_per_output]
// with the rescaled+bitpacked mask template derived from d_a_hat. Call
// ONCE at init time from the online orchestrator, then download the
// resulting buffer to host and persist it on the server struct.
void sw_tc_packing_fill_mask_template(void* context, cudaStream_t stream,
    uint8_t* d_mask_template_out)
{
    auto* ctx = (SwTcPackingContext*)context;
    if (!ctx || d_mask_template_out == nullptr) return;
    size_t N = ctx->poly_len, rho = ctx->num_outputs;
    dim3 grid((int)rho, 1);
    sw_inspir_fill_mask_template<<<grid, 1024, 0, stream>>>(
        d_mask_template_out, ctx->d_a_hat,
        ctx->rlwe_q_prime_2, ctx->mask_bytes_per_output,
        ctx->mod0, N);
    CUDA_ASSERT(cudaGetLastError());
}

void sw_tc_packing_free(void* context)
{
    auto* ctx = (SwTcPackingContext*)context;
    if (!ctx) return;
    if (ctx->d_M_bytes[0]) cudaFree(ctx->d_M_bytes[0]);
    if (ctx->d_bold_t_hat) cudaFree(ctx->d_bold_t_hat);
    if (ctx->d_a_hat) cudaFree(ctx->d_a_hat);
    if (ctx->d_A_bytes_buf) cudaFree(ctx->d_A_bytes_buf);
    if (ctx->d_G) cudaFree(ctx->d_G);
    if (ctx->d_result) cudaFree(ctx->d_result);
    if (ctx->d_scratch) cudaFree(ctx->d_scratch);
    delete ctx;
}

} // extern "C"
