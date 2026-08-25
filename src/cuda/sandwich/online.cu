/**
 * SandwichPIR GPU Online — Orchestrator
 *
 * Coordinates the full online pipeline on GPU: matmul + TC packing + post-process.
 *   - Matmul: u32 queries × u8 DB via stacked byte-decomposed CUTLASS GEMM, then modswitch.
 *   - TC packing (tc_packing.cu): InspiRING packing GEMM + finalize into scratch.
 *   - Post-process (tc_packing.cu): INTT + add body values + modswitch + bitpack.
 *
 * Init uploads DB and allocates all online scratch buffers.
 * Compute orchestrates two-stream overlap: copy_stream for H→D uploads,
 * compute_stream for serialized GEMMs and post-processing.
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

#define SW_ASSERT(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    fprintf(stderr, "SW online CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); abort(); } } while(0)

// ═══════════════════════════════════════════════════════════════
// CUTLASS GEMM for matmul (same types as tc_packing)
// ═══════════════════════════════════════════════════════════════

using SwMmGemm_Sm80 = cutlass::gemm::device::Gemm<
    uint8_t, cutlass::layout::RowMajor, uint8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor, int32_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 3>;

using SwMmGemm_Sm75 = cutlass::gemm::device::Gemm<
    uint8_t, cutlass::layout::RowMajor, uint8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor, int32_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 2>;

using SwMmGemm_Sm50 = cutlass::gemm::device::Gemm<
    uint8_t, cutlass::layout::RowMajor, uint8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor, int32_t,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 1, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 2>;

static cutlass::Status sw_matmul_gemm(int tier, int M, int N, int K,
    const uint8_t* A, int lda, const uint8_t* B, int ldb,
    int32_t* C, int ldc, int32_t alpha, int32_t beta,
    cudaStream_t stream = 0) {
    cutlass::gemm::GemmCoord ps(M,N,K);
    if (tier>=2) { SwMmGemm_Sm80 op; SwMmGemm_Sm80::Arguments a{ps,{A,lda},{B,ldb},{C,ldc},{C,ldc},{alpha,beta},1}; auto s=op.initialize(a,nullptr,stream); if(s!=cutlass::Status::kSuccess)return s; return op(stream); }
    if (tier==1) { SwMmGemm_Sm75 op; SwMmGemm_Sm75::Arguments a{ps,{A,lda},{B,ldb},{C,ldc},{C,ldc},{alpha,beta},1}; auto s=op.initialize(a,nullptr,stream); if(s!=cutlass::Status::kSuccess)return s; return op(stream); }
    SwMmGemm_Sm50 op; SwMmGemm_Sm50::Arguments a{ps,{A,lda},{B,ldb},{C,ldc},{C,ldc},{alpha,beta},1}; auto s=op.initialize(a,nullptr,stream); if(s!=cutlass::Status::kSuccess)return s; return op(stream);
}

// ═══════════════════════════════════════════════════════════════
// Matmul Kernels
// ═══════════════════════════════════════════════════════════════

// Extract low byte of u16 DB into [db_cols × db_rows_padded] u8 RowMajor.
// p=256 so all values fit in u8 — high byte is always 0, no stacking needed.
// DB is col-major: db[col * db_rows_padded + row]. Output keeps same layout.
__global__ void sw_decompose_db(uint8_t* out, const uint16_t* db, size_t count) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    out[idx] = (uint8_t)(db[idx] & 0xFF);
}

// Byte-decompose u32 queries (stored as u64) into 4 horizontally-stacked slices
__global__ void sw_decompose_query(uint8_t* out, const uint64_t* q, size_t K, size_t N) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * N) return;
    uint32_t v = (uint32_t)q[idx];
    out[idx + 0*K*N] = (uint8_t)(v);
    out[idx + 1*K*N] = (uint8_t)(v >> 8);
    out[idx + 2*K*N] = (uint8_t)(v >> 16);
    out[idx + 3*K*N] = (uint8_t)(v >> 24);
}

// Accumulate (M)×(4N) i32 GEMM → M×N u32 in Z_Q, modswitch W=2^32 → Q.
// DB is 1 byte (p=256), query is 4 bytes → 4 byte-slice products with shifts 0,8,16,24.
// Result is round(acc * Q / 2^32), computed via __umulhi (32×32→64 in 1 instruction).
// Output fits in u32 (Q < 2^32) but stored as u64 for downstream compatibility.
__global__ void sw_accum_modswitch(
    uint64_t* inter, const int32_t* gemm, size_t M, size_t N, uint32_t Q) {
    size_t n = blockIdx.y;
    if (n >= N) return;
    const int32_t* col0 = gemm + (0*N + n) * M;
    const int32_t* col1 = gemm + (1*N + n) * M;
    const int32_t* col2 = gemm + (2*N + n) * M;
    const int32_t* col3 = gemm + (3*N + n) * M;
    uint64_t* out = inter + n * M;
    for (size_t m = (size_t)blockIdx.x * blockDim.x + threadIdx.x; m < M; m += (size_t)gridDim.x * blockDim.x) {
        uint32_t acc = (uint32_t)col0[m];
        acc += (uint32_t)col1[m] << 8;
        acc += (uint32_t)col2[m] << 16;
        acc += (uint32_t)col3[m] << 24;
        // round(acc * Q / 2^32) via 32-bit intrinsics (3 instructions, not 16)
        uint32_t lo = acc * Q;
        uint32_t hi = __umulhi(acc, Q);
        out[m] = (uint64_t)(hi + (lo >= 0x80000000u));
    }
}

// ═══════════════════════════════════════════════════════════════
// Context
// ═══════════════════════════════════════════════════════════════

struct SandwichOnlineContext {
    uint8_t* d_db_stacked;       // (2M) × K_pad u8
    uint64_t* d_query_raw;       // K × max_B u64
    uint8_t*  d_query_stacked;   // K × (4*max_B) u8
    int32_t*  d_gemm_out;        // (2M) × (4*max_B) i32
    uint64_t* d_intermediate;    // M × max_B u64 (Z_Q)
    uint8_t*  d_response;        // max_B × rho × body_bytes_per_output
    uint8_t*  d_mask_template;   // rho × mask_bytes_per_output (populated ONCE at init)

    // TC packing context (owned)
    void* tc_ctx;

    // Packing key scratch (for upload)
    uint64_t* d_y_body;
    uint64_t* d_z_body;

    size_t db_rows, db_rows_padded, db_cols, poly_len;
    size_t num_outputs, max_batch_size, t_exp_left;
    size_t response_bytes_per_output;    // full wire format (mask + body), kept for metadata
    size_t body_bytes_per_output;        // = (poly_len * q_2_bits + 7) / 8
    size_t mask_bytes_per_output;        // = (poly_len * q_1_bits + 7) / 8
    uint64_t Q, q_prime_1, q_prime_2;
    int gpu_tier;

    NTTParams ntt_params;  // device pointers
    cudaStream_t compute_stream;  // all GEMMs serialized here (full HBM each)
    cudaStream_t copy_stream;     // H→D uploads (overlaps with compute via events)
};

// ═══════════════════════════════════════════════════════════════
// Extern "C" API
// ═══════════════════════════════════════════════════════════════

// Helper: alloc + copy host→device
static void* alloc_copy(const void* h, size_t bytes) {
    void* d; SW_ASSERT(cudaMalloc(&d, bytes));
    SW_ASSERT(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
    return d;
}

extern "C" {

// Per-thread cudaSetDevice wrapper. CUDA's current device is thread-local;
// the Rust sharded entry point spawns one worker thread per GPU and calls
// this once at the top of each thread so all subsequent FFI calls in that
// thread route to the right device. Returns 0 on success, cudaError_t otherwise.
int sandwich_set_device(int dev) {
    return (int)cudaSetDevice(dev);
}

// Number of CUDA-capable devices visible to this process.
int sandwich_device_count(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

// Allocate page-locked (pinned) host memory. Buffers allocated here can be
// used as the source/destination of `cudaMemcpyAsync` to get TRUE async DMA
// directly to/from GPU HBM, bypassing the internal staging copy that
// pageable host memory requires. The host thread is not blocked during the
// transfer, and multiple GPUs can DMA from the same pinned source in
// parallel. Critical for multi-GPU sharded uploads.
//
// Returns NULL on failure. Caller must release with sandwich_free_pinned.
void* sandwich_alloc_pinned(size_t bytes) {
    void* ptr = nullptr;
    cudaError_t e = cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable);
    if (e != cudaSuccess) {
        fprintf(stderr, "sandwich_alloc_pinned(%zu): cudaHostAlloc failed: %s\n",
                bytes, cudaGetErrorString(e));
        return nullptr;
    }
    return ptr;
}

void sandwich_free_pinned(void* ptr) {
    if (ptr) cudaFreeHost(ptr);
}

void* sandwich_online_init(
    const uint8_t* h_db,
    size_t db_rows, size_t db_rows_padded, size_t db_cols,
    size_t poly_len, size_t t_exp_left,
    size_t num_outputs, size_t max_batch_size,
    uint64_t Q, uint64_t q_prime_1, uint64_t q_prime_2,
    size_t response_bytes_per_output,
    // NTT params as individual arrays (single CRT)
    const uint64_t* h_moduli, const uint64_t* h_barrett_cr,
    const uint64_t* h_fwd, const uint64_t* h_fwd_p,
    const uint64_t* h_inv, const uint64_t* h_inv_p,
    uint64_t modulus,
    int gpu_tier)
{
    auto* ctx = new SandwichOnlineContext();
    ctx->db_rows = db_rows; ctx->db_rows_padded = db_rows_padded;
    ctx->db_cols = db_cols; ctx->poly_len = poly_len;
    ctx->t_exp_left = t_exp_left; ctx->num_outputs = num_outputs;
    ctx->max_batch_size = max_batch_size;
    ctx->response_bytes_per_output = response_bytes_per_output;
    {
        // Mask = q_1 = q_prime_2 (large, 2^18); body = q_2 = q_prime_1 (small, 2^10).
        size_t q_1_bits = 0, q_2_bits = 0;
        { uint64_t y = q_prime_2 - 1; while (y) { q_1_bits++; y >>= 1; } }
        { uint64_t y = q_prime_1 - 1; while (y) { q_2_bits++; y >>= 1; } }
        ctx->mask_bytes_per_output = (poly_len * q_1_bits + 7) / 8;
        ctx->body_bytes_per_output = (poly_len * q_2_bits + 7) / 8;
    }
    ctx->Q = Q; ctx->q_prime_1 = q_prime_1; ctx->q_prime_2 = q_prime_2;
    ctx->gpu_tier = gpu_tier;
    ctx->tc_ctx = nullptr;
    ctx->d_mask_template = nullptr;

    // Build NTTParams with device pointers
    size_t tbl = poly_len * sizeof(uint64_t);  // single CRT
    ctx->ntt_params.poly_len = (uint32_t)poly_len;
    ctx->ntt_params.log2_poly_len = 31 - __builtin_clz((uint32_t)poly_len);
    ctx->ntt_params.crt_count = 1;
    ctx->ntt_params.modulus = modulus;
    ctx->ntt_params.moduli = (uint64_t*)alloc_copy(h_moduli, sizeof(uint64_t));
    ctx->ntt_params.barrett_cr = (uint64_t*)alloc_copy(h_barrett_cr, sizeof(uint64_t));
    ctx->ntt_params.forward_table = (uint64_t*)alloc_copy(h_fwd, tbl);
    ctx->ntt_params.forward_prime_table = (uint64_t*)alloc_copy(h_fwd_p, tbl);
    ctx->ntt_params.inverse_table = (uint64_t*)alloc_copy(h_inv, tbl);
    ctx->ntt_params.inverse_prime_table = (uint64_t*)alloc_copy(h_inv_p, tbl);

    // DB and matmul scratch are allocated lazily via sandwich_online_upload_db
    // to allow M-matrix build (which frees bold_t) before DB upload on tight GPUs.
    ctx->d_db_stacked = nullptr;
    ctx->d_query_raw = nullptr;
    ctx->d_query_stacked = nullptr;
    ctx->d_gemm_out = nullptr;
    ctx->d_intermediate = nullptr;
    ctx->d_response = nullptr;
    ctx->d_y_body = nullptr;
    ctx->d_z_body = nullptr;

    // Upload DB + allocate scratch if h_db is provided
    if (h_db) {
        size_t db_elems = db_cols * db_rows_padded;
        SW_ASSERT(cudaMalloc(&ctx->d_db_stacked, db_elems));
        SW_ASSERT(cudaMemcpy(ctx->d_db_stacked, h_db, db_elems, cudaMemcpyHostToDevice));

        int align = (gpu_tier >= 1) ? 4 : 1;
        size_t pb = ((max_batch_size + align - 1) / align) * align;
        SW_ASSERT(cudaMalloc(&ctx->d_query_raw, db_rows * pb * sizeof(uint64_t)));
        SW_ASSERT(cudaMalloc(&ctx->d_query_stacked, db_rows * 4 * pb));
        SW_ASSERT(cudaMalloc(&ctx->d_gemm_out, db_cols * 4 * pb * sizeof(int32_t)));
        SW_ASSERT(cudaMalloc(&ctx->d_intermediate, db_cols * pb * sizeof(uint64_t)));

        // Body-only response buffer (mask bytes live in d_mask_template, not here).
        size_t resp_body = pb * num_outputs * ctx->body_bytes_per_output;
        SW_ASSERT(cudaMalloc(&ctx->d_response, resp_body));

        // Static mask template: one copy, shared across all batches/clients.
        SW_ASSERT(cudaMalloc(&ctx->d_mask_template, num_outputs * ctx->mask_bytes_per_output));

        size_t key_elems = pb * t_exp_left * poly_len;
        SW_ASSERT(cudaMalloc(&ctx->d_y_body, key_elems * sizeof(uint64_t)));
        SW_ASSERT(cudaMalloc(&ctx->d_z_body, key_elems * sizeof(uint64_t)));
    }

    SW_ASSERT(cudaStreamCreate(&ctx->compute_stream));
    SW_ASSERT(cudaStreamCreate(&ctx->copy_stream));

    SW_LOG("SandwichPIR online init: db=%zux%zu, batch=%zu, tier=%d\n",
           db_rows, db_cols, max_batch_size, gpu_tier);

    return ctx;
}

// Deferred DB upload + scratch allocation (for staged memory management).
void sandwich_online_upload_db(void* context, const uint8_t* h_db) {
    auto* ctx = (SandwichOnlineContext*)context;
    if (!ctx || !h_db || ctx->d_db_stacked) return; // already uploaded

    size_t db_elems = ctx->db_cols * ctx->db_rows_padded;
    SW_ASSERT(cudaMalloc(&ctx->d_db_stacked, db_elems));
    SW_ASSERT(cudaMemcpy(ctx->d_db_stacked, h_db, db_elems, cudaMemcpyHostToDevice));

    int align = (ctx->gpu_tier >= 1) ? 4 : 1;
    size_t pb = ((ctx->max_batch_size + align - 1) / align) * align;
    SW_ASSERT(cudaMalloc(&ctx->d_query_raw, ctx->db_rows * pb * sizeof(uint64_t)));
    SW_ASSERT(cudaMalloc(&ctx->d_query_stacked, ctx->db_rows * 4 * pb));
    SW_ASSERT(cudaMalloc(&ctx->d_gemm_out, ctx->db_cols * 4 * pb * sizeof(int32_t)));
    SW_ASSERT(cudaMalloc(&ctx->d_intermediate, ctx->db_cols * pb * sizeof(uint64_t)));

    // Body-only response + static mask template (see non-deferred init path above).
    size_t resp_body = pb * ctx->num_outputs * ctx->body_bytes_per_output;
    SW_ASSERT(cudaMalloc(&ctx->d_response, resp_body));
    SW_ASSERT(cudaMalloc(&ctx->d_mask_template, ctx->num_outputs * ctx->mask_bytes_per_output));

    size_t key_elems = pb * ctx->t_exp_left * ctx->poly_len;
    SW_ASSERT(cudaMalloc(&ctx->d_y_body, key_elems * sizeof(uint64_t)));
    SW_ASSERT(cudaMalloc(&ctx->d_z_body, key_elems * sizeof(uint64_t)));

    // In the staged path init_packing ran before upload_db, so d_mask_template
    // did not yet exist when init_packing wanted to populate it. Fire the
    // fill here now that both tc_ctx AND d_mask_template are ready.
    if (ctx->tc_ctx != nullptr) {
        sw_tc_packing_fill_mask_template(ctx->tc_ctx, 0, ctx->d_mask_template);
        SW_ASSERT(cudaDeviceSynchronize());
    }
}

void sandwich_online_init_packing(
    void* context,
    uint64_t* d_bold_t, uint64_t* d_bold_t_bar,
    const uint64_t* d_bold_t_hat, const uint64_t* d_a_hat,
    const uint32_t* d_tables, const uint32_t* d_gen_pows,
    size_t num_iter,
    size_t tables_count, size_t gen_pows_count)
{
    auto* ctx = (SandwichOnlineContext*)context;
    ctx->tc_ctx = sw_tc_packing_init(
        d_bold_t, d_bold_t_bar, d_bold_t_hat, d_a_hat,
        d_tables, d_gen_pows,
        num_iter, ctx->t_exp_left, ctx->poly_len,
        ctx->num_outputs, ctx->max_batch_size,
        ctx->q_prime_1, ctx->q_prime_2,
        ctx->response_bytes_per_output,
        ctx->ntt_params, ctx->gpu_tier);

    // Fill d_mask_template from d_a_hat once, now that the tc_ctx holds
    // the precomputed mask polynomials. The resulting bytes persist for
    // the lifetime of this context; callers download them once via
    // sandwich_online_download_mask_template and keep a host-side copy.
    // (In the staged path d_mask_template may not be allocated yet —
    // upload_db will call the fill afterwards.)
    if (ctx->d_mask_template) {
        sw_tc_packing_fill_mask_template(ctx->tc_ctx, 0, ctx->d_mask_template);
        SW_ASSERT(cudaDeviceSynchronize());
    }
    SW_LOG("SandwichPIR packing initialized\n");
}

// Download body bytes directly into a pre-tiled pinned host response buffer
// via cudaMemcpy3D.  The destination is strided: within each client slot,
// body_bpo bytes must land at offset `mask_bpo` after every slot start;
// across clients, the slice stride is `dst_row_pitch * dst_slice_ysize`
// (one full client slot).  The caller pre-tiled the mask bytes into the
// buffer at offline init, so the GPU never touches mask bytes during online
// and no host-side scatter is ever required.
//
// dst_byte_offset: byte offset within h_response_buf where THIS shard's
//   first body slot begins — = (this_shard_output_offset * dst_row_pitch)
//   + mask_bpo.  For single-shard mode, output_offset = 0 so this is
//   just mask_bpo.
// dst_row_pitch: full per-output stride in the host buffer, i.e.
//   response_bytes_per_output = mask_bpo + body_bpo.
// dst_slice_ysize: total number of outputs per client in the host buffer
//   (num_outputs across all shards); combines with dst_row_pitch to give
//   the slice (= client) stride dst_row_pitch * dst_slice_ysize.
// copy_height: number of outputs THIS shard is contributing per client
//   (= num_outputs_local); rows per slice we actually copy.
// body_bpo: row width in bytes (= body_bytes_per_output).
void sandwich_online_compute(
    void* context,
    const uint64_t* h_queries, size_t batch_size,
    const uint64_t* h_y_body, const uint64_t* h_z_body,
    uint8_t* h_response_buf,
    size_t dst_byte_offset,
    size_t dst_row_pitch,
    size_t dst_slice_ysize,
    size_t copy_height,
    size_t body_bpo)
{
    auto* ctx = (SandwichOnlineContext*)context;
    if (!ctx || batch_size == 0 || !ctx->tc_ctx) return;

    // ══════════════════════════════════════════════════════════════════════
    // Stream orchestration: serialized GEMMs with overlapped H→D uploads
    // ══════════════════════════════════════════════════════════════════════
    //
    // Why packing runs first:
    //   Packing keys (y_body, z_body) are small (~100 KB per client) and upload
    //   quickly, whereas queries are large (~8 MB per client) and block the CPU
    //   for ~2.9 ms with unpinned memory. By launching the packing GEMM after
    //   the small key upload, the GPU executes packing while the CPU is blocked
    //   inside the large query cudaMemcpyAsync. This hides the query transfer.
    //
    // Why GEMMs are serialized (not concurrent):
    //   Both the packing GEMM and the matmul GEMM are HBM-bandwidth-bound.
    //   Running them concurrently would halve available memory bandwidth for
    //   each, making both slower than sequential execution. Serializing on a
    //   single compute_stream gives each GEMM full HBM bandwidth.
    //
    // Event dependency pattern:
    //   copy_stream handles all H→D transfers. Two events synchronize with
    //   compute_stream: ev_keys_ready gates the packing GEMM launch, and
    //   ev_queries_ready gates the matmul launch. This ensures each GEMM
    //   starts only after its inputs are resident on the GPU.
    //
    //   copy_stream:    [upload keys] ─── [upload queries (2.9 ms, blocks CPU)] ──
    //                        │ ev_keys_ready       │ ev_queries_ready
    //   compute_stream:      └──► packing GEMM ────┴──► matmul ──► post-process

    size_t K = ctx->db_rows, M = ctx->db_cols, N = batch_size;
    int align = (ctx->gpu_tier >= 1) ? 4 : 1;
    int Nd = ((int)N + align - 1) / align * align;
    cudaStream_t cs = ctx->compute_stream;
    cudaStream_t xs = ctx->copy_stream;
    size_t key_sz = N * ctx->t_exp_left * ctx->poly_len * sizeof(uint64_t);

    bool verbose = sw_verbose();
    auto wtime = []() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6; };
    cudaEvent_t ev_keys_ready, ev_queries_ready;
    cudaEvent_t ev_pk, ev_mm_gemm, ev_mm, ev_fin;
    SW_ASSERT(cudaEventCreate(&ev_keys_ready));
    SW_ASSERT(cudaEventCreate(&ev_queries_ready));
    if (verbose) {
        SW_ASSERT(cudaEventCreate(&ev_pk));
        SW_ASSERT(cudaEventCreate(&ev_mm_gemm));
        SW_ASSERT(cudaEventCreate(&ev_mm));
        SW_ASSERT(cudaEventCreate(&ev_fin));
    }
    double t0 = verbose ? wtime() : 0;

    // ── Upload keys (small), launch packing, THEN upload queries (large, blocks CPU) ──
    // With unpinned host memory, cudaMemcpyAsync blocks the CPU. We must
    // launch the packing GEMM BEFORE the large query upload so the GPU
    // runs the GEMM while the CPU is blocked in the query copy.
    SW_ASSERT(cudaMemcpyAsync(ctx->d_y_body, h_y_body, key_sz, cudaMemcpyHostToDevice, xs));
    SW_ASSERT(cudaMemcpyAsync(ctx->d_z_body, h_z_body, key_sz, cudaMemcpyHostToDevice, xs));
    SW_ASSERT(cudaEventRecord(ev_keys_ready, xs));

    // ── Phase 1: Launch packing GEMM (GPU starts while CPU does query upload) ──
    SW_ASSERT(cudaStreamWaitEvent(cs, ev_keys_ready, 0));
    sw_tc_packing_gemms(ctx->tc_ctx, cs, ctx->d_y_body, ctx->d_z_body, batch_size);
    if (verbose) SW_ASSERT(cudaEventRecord(ev_pk, cs));

    // ── Upload queries (may block CPU ~2.9 ms if unpinned — packing GEMM already running) ──
    size_t qe = K * N;
    SW_ASSERT(cudaMemcpyAsync(ctx->d_query_raw, h_queries, qe * sizeof(uint64_t), cudaMemcpyHostToDevice, xs));
    SW_ASSERT(cudaEventRecord(ev_queries_ready, xs));

    // ── Phase 2: Matmul (waits for queries — already uploaded during packing) ──
    SW_ASSERT(cudaStreamWaitEvent(cs, ev_queries_ready, 0));
    { int t=256, b=(qe+t-1)/t;
      sw_decompose_query<<<b,t,0,cs>>>(ctx->d_query_stacked, ctx->d_query_raw, K, N);
      SW_ASSERT(cudaGetLastError()); }
    { auto st = sw_matmul_gemm(ctx->gpu_tier, (int)M, 4*Nd, (int)K,
          ctx->d_db_stacked, (int)ctx->db_rows_padded,
          ctx->d_query_stacked, (int)K,
          ctx->d_gemm_out, (int)M, 1, 0, cs);
      if (st != cutlass::Status::kSuccess) { fprintf(stderr, "SW matmul GEMM failed\n"); abort(); }
    }
    if (verbose) SW_ASSERT(cudaEventRecord(ev_mm_gemm, cs));
    { int thr=256, bx=((int)M+thr-1)/thr; dim3 grid(bx,(int)N);
      sw_accum_modswitch<<<grid,thr,0,cs>>>(ctx->d_intermediate, ctx->d_gemm_out, M, N, (uint32_t)ctx->Q);
      SW_ASSERT(cudaGetLastError()); }
    if (verbose) SW_ASSERT(cudaEventRecord(ev_mm, cs));

    // ── Phase 3: Post-process (needs both d_scratch from packing and d_intermediate from matmul) ──
    sw_tc_packing_post_process(ctx->tc_ctx, cs,
        ctx->d_intermediate, ctx->d_response, batch_size);
    if (verbose) SW_ASSERT(cudaEventRecord(ev_fin, cs));
    SW_ASSERT(cudaStreamSynchronize(cs));
    double t1 = verbose ? wtime() : 0;

    // ── Phase 4: Direct 3D download into pinned pre-tiled response buffer ──
    // One cudaMemcpy3D writes body bytes from the flat GPU buffer
    // (contiguous per-client, per-output) directly into the strided body
    // slots of the caller's pre-tiled host response buffer.  The mask
    // bytes in that buffer are untouched (pre-filled at offline init),
    // so after this call the buffer contains the complete interleaved
    // `[mask | body | mask | body | ...]` wire format for every client
    // slot this shard contributes to.  With a pinned destination the
    // DMA goes directly from GPU to host RAM with no internal staging
    // copy and no host-side scatter loop.
    {
        cudaMemcpy3DParms p = {0};
        p.srcPtr = make_cudaPitchedPtr(
            (void*)ctx->d_response,
            body_bpo,               // src row pitch (contiguous body rows on GPU)
            body_bpo,               // logical width = body_bpo
            copy_height);           // src ysize = num_outputs_local per client
        p.dstPtr = make_cudaPitchedPtr(
            (void*)(h_response_buf + dst_byte_offset),
            dst_row_pitch,          // row pitch = response_bytes_per_output
            body_bpo,               // logical width = body_bpo
            dst_slice_ysize);       // dst ysize = rho_total (slice pitch = resp_bpo × rho_total)
        p.extent = make_cudaExtent(body_bpo, copy_height, N);
        p.kind = cudaMemcpyDeviceToHost;
        SW_ASSERT(cudaMemcpy3D(&p));
    }
    double t2 = verbose ? wtime() : 0;

    if (verbose) {
        float pk_ms, mm_gemm_ms, mm_accum_ms, fin_ms;
        cudaEventElapsedTime(&pk_ms, ev_keys_ready, ev_pk);
        cudaEventElapsedTime(&mm_gemm_ms, ev_pk, ev_mm_gemm);
        cudaEventElapsedTime(&mm_accum_ms, ev_mm_gemm, ev_mm);
        cudaEventElapsedTime(&fin_ms, ev_mm, ev_fin);

        // Packing sub-phases (from tc_packing context)
        auto* tc = (SwTcPackingContext*)ctx->tc_ctx;
        float pk_decomp=0, pk_gemm=0, pk_accum=0, pk_z=0;
        if (tc->has_events) {
            cudaEventElapsedTime(&pk_decomp, ev_keys_ready, tc->ev_decomp);
            cudaEventElapsedTime(&pk_gemm, tc->ev_decomp, tc->ev_gemm);
            cudaEventElapsedTime(&pk_accum, tc->ev_gemm, tc->ev_accum);
            cudaEventElapsedTime(&pk_z, tc->ev_accum, tc->ev_finalize);
            cudaEventDestroy(tc->ev_decomp);
            cudaEventDestroy(tc->ev_gemm);
            cudaEventDestroy(tc->ev_accum);
            cudaEventDestroy(tc->ev_finalize);
            tc->has_events = false;
        }

        SW_LOG("SW online breakdown (%zu clients):\n", batch_size);
        SW_LOG("  packing(Ydecomp=%.1f gemm=%.1f accum=%.1f Zfinal=%.1f)=%.1f\n",
               pk_decomp, pk_gemm, pk_accum, pk_z, pk_ms);
        SW_LOG("  matmul(gemm=%.1f accum=%.1f)=%.1f  post=%.1f ms\n",
               mm_gemm_ms, mm_accum_ms, mm_gemm_ms + mm_accum_ms, fin_ms);
        SW_LOG("  wall: compute=%.1f download=%.1f total=%.1f ms\n",
               t1-t0, t2-t1, t2-t0);
        cudaEventDestroy(ev_pk);
        cudaEventDestroy(ev_mm_gemm);
        cudaEventDestroy(ev_mm);
        cudaEventDestroy(ev_fin);
    }
    cudaEventDestroy(ev_keys_ready);
    cudaEventDestroy(ev_queries_ready);
}

// Matmul-only: used for TC GEMM hint (A × DB in Z_{2^32} → modswitch → Z_Q).
// Same GEMM as the online path, but skips TC packing. Downloads intermediate to host.
void sandwich_online_hint_gemm(
    void* context,
    const uint64_t* h_a_rows, size_t batch_size,
    uint64_t* h_result_out)
{
    auto* ctx = (SandwichOnlineContext*)context;
    if (!ctx || batch_size == 0) return;
    size_t K = ctx->db_rows, M = ctx->db_cols, N = batch_size;
    int align = (ctx->gpu_tier >= 1) ? 4 : 1;
    int Nd = ((int)N + align - 1) / align * align;

    SW_ASSERT(cudaMemcpy(ctx->d_query_raw, h_a_rows, K * N * sizeof(uint64_t), cudaMemcpyHostToDevice));
    { int t=256, b=((int)(K*N)+t-1)/t;
      sw_decompose_query<<<b,t>>>(ctx->d_query_stacked, ctx->d_query_raw, K, N); SW_ASSERT(cudaGetLastError()); }
    { auto s = sw_matmul_gemm(ctx->gpu_tier, (int)M, 4*Nd, (int)K,
          ctx->d_db_stacked, (int)ctx->db_rows_padded, ctx->d_query_stacked, (int)K,
          ctx->d_gemm_out, (int)M, 1, 0);
      if (s != cutlass::Status::kSuccess) { fprintf(stderr, "SW hint GEMM failed\n"); abort(); } }
    { int thr=256, bx=((int)M+thr-1)/thr; dim3 grid(bx,(int)N);
      sw_accum_modswitch<<<grid,thr>>>(ctx->d_intermediate, ctx->d_gemm_out, M, N, (uint32_t)ctx->Q); SW_ASSERT(cudaGetLastError()); }
    SW_ASSERT(cudaMemcpy(h_result_out, ctx->d_intermediate, M * N * sizeof(uint64_t), cudaMemcpyDeviceToHost));
}

// Same as hint_gemm but keeps result on device. Caller takes d_intermediate via take function.
void sandwich_online_hint_gemm_on_device(
    void* context,
    const uint64_t* h_a_rows, size_t batch_size)
{
    auto* ctx = (SandwichOnlineContext*)context;
    if (!ctx || batch_size == 0) return;
    size_t K = ctx->db_rows, M = ctx->db_cols, N = batch_size;
    int align = (ctx->gpu_tier >= 1) ? 4 : 1;
    int Nd = ((int)N + align - 1) / align * align;

    bool verbose = sw_verbose();
    cudaEvent_t ev_start, ev_gemm, ev_accum;
    if (verbose) {
        SW_ASSERT(cudaEventCreate(&ev_start));
        SW_ASSERT(cudaEventCreate(&ev_gemm));
        SW_ASSERT(cudaEventCreate(&ev_accum));
        SW_ASSERT(cudaEventRecord(ev_start));
    }

    SW_ASSERT(cudaMemcpy(ctx->d_query_raw, h_a_rows, K * N * sizeof(uint64_t), cudaMemcpyHostToDevice));
    { int t=256, b=((int)(K*N)+t-1)/t;
      sw_decompose_query<<<b,t>>>(ctx->d_query_stacked, ctx->d_query_raw, K, N); SW_ASSERT(cudaGetLastError()); }
    { auto s = sw_matmul_gemm(ctx->gpu_tier, (int)M, 4*Nd, (int)K,
          ctx->d_db_stacked, (int)ctx->db_rows_padded, ctx->d_query_stacked, (int)K,
          ctx->d_gemm_out, (int)M, 1, 0);
      if (s != cutlass::Status::kSuccess) { fprintf(stderr, "SW hint GEMM failed\n"); abort(); } }
    if (verbose) SW_ASSERT(cudaEventRecord(ev_gemm));
    { int thr=256, bx=((int)M+thr-1)/thr; dim3 grid(bx,(int)N);
      sw_accum_modswitch<<<grid,thr>>>(ctx->d_intermediate, ctx->d_gemm_out, M, N, (uint32_t)ctx->Q); SW_ASSERT(cudaGetLastError()); }
    if (verbose) SW_ASSERT(cudaEventRecord(ev_accum));
    SW_ASSERT(cudaDeviceSynchronize());

    if (verbose) {
        float gemm_ms, accum_ms;
        cudaEventElapsedTime(&gemm_ms, ev_start, ev_gemm);
        cudaEventElapsedTime(&accum_ms, ev_gemm, ev_accum);
        SW_LOG("SW GEMM hint: gemm=%.1f accum=%.1f total=%.1f ms (%zu polys)\n",
               gemm_ms, accum_ms, gemm_ms + accum_ms, N);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_gemm);
        cudaEventDestroy(ev_accum);
    }
}

// Take d_intermediate pointer (nulls it so free won't touch it)
uint64_t* sandwich_online_take_intermediate(void* context) {
    auto* ctx = (SandwichOnlineContext*)context;
    if (!ctx) return nullptr;
    uint64_t* ptr = ctx->d_intermediate;
    ctx->d_intermediate = nullptr;
    return ptr;
}

// Download the static mask template bytes from device. The destination
// must be at least `num_outputs * mask_bytes_per_output` bytes. Called
// ONCE per shard from server_gpu.rs::SandwichGpuServer::new after the
// offline pipeline finishes; the bytes are then held host-side in the
// server struct and reused for every query assembly.
void sandwich_online_download_mask_template(void* context, uint8_t* h_out) {
    auto* ctx = (SandwichOnlineContext*)context;
    if (!ctx || !ctx->d_mask_template || !h_out) return;
    size_t bytes = ctx->num_outputs * ctx->mask_bytes_per_output;
    SW_ASSERT(cudaMemcpy(h_out, ctx->d_mask_template, bytes, cudaMemcpyDeviceToHost));
}

void sandwich_online_free(void* context) {
    auto* ctx = (SandwichOnlineContext*)context;
    if (!ctx) return;
    if (ctx->compute_stream) cudaStreamDestroy(ctx->compute_stream);
    if (ctx->copy_stream) cudaStreamDestroy(ctx->copy_stream);
    if (ctx->d_db_stacked) cudaFree(ctx->d_db_stacked);
    if (ctx->d_query_raw) cudaFree(ctx->d_query_raw);
    if (ctx->d_query_stacked) cudaFree(ctx->d_query_stacked);
    if (ctx->d_gemm_out) cudaFree(ctx->d_gemm_out);
    if (ctx->d_intermediate) cudaFree(ctx->d_intermediate);
    if (ctx->d_response) cudaFree(ctx->d_response);
    if (ctx->d_mask_template) cudaFree(ctx->d_mask_template);
    if (ctx->d_y_body) cudaFree(ctx->d_y_body);
    if (ctx->d_z_body) cudaFree(ctx->d_z_body);
    if (ctx->tc_ctx) sw_tc_packing_free(ctx->tc_ctx);
    if (ctx->ntt_params.moduli) cudaFree(ctx->ntt_params.moduli);
    if (ctx->ntt_params.barrett_cr) cudaFree(ctx->ntt_params.barrett_cr);
    if (ctx->ntt_params.forward_table) cudaFree(ctx->ntt_params.forward_table);
    if (ctx->ntt_params.forward_prime_table) cudaFree(ctx->ntt_params.forward_prime_table);
    if (ctx->ntt_params.inverse_table) cudaFree(ctx->ntt_params.inverse_table);
    if (ctx->ntt_params.inverse_prime_table) cudaFree(ctx->ntt_params.inverse_prime_table);
    delete ctx;
}

} // extern "C"
