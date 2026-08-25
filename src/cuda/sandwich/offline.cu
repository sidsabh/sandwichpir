/**
 * SandwichPIR GPU Offline Pipeline — Single modulus NTT hint + InspiRING precomp
 *
 * Part 1: NTT Hint — compute H = A · DB via polynomial convolution in R_Q
 *   For each DB column: NTT(db_col) * A_ntt[row] → accumulate → INTT → hint
 *   Single modulus (no CRT), so convd_len = poly_len (not 2*poly_len)
 *
 * Part 2: InspiRING Precomp — build rotation matrices for TC packing
 *   Generates bold_t, bold_t_bar, bold_t_hat, a_hat on GPU.
 *   These feed into sandwich_online_init_packing() to build the M-matrix.
 *
 * Adapted from simplepir/offline.cu and inspiring/precomp.cu for single CRT.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "common/ntt.cuh"
#include "common/log.cuh"

#define SW_CUDA_ASSERT(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    fprintf(stderr, "SandwichPIR offline CUDA error %d at %s:%d: %s\n", err, __FILE__, __LINE__, cudaGetErrorString(err)); \
    abort(); } } while(0)

// ═══════════════════════════════════════════════════════════════
// Part 1: NTT Hint Kernel (single modulus)
// ═══════════════════════════════════════════════════════════════

// One block per DB column. All 1024 threads work on a single CRT component.
// For each row polynomial: NTT(db_elem), pointwise multiply with A_ntt, accumulate.
// Accumulator stays in global memory so a separate finalize kernel can INTT it.
__global__ void sw_compute_hint_loop_kernel(
    const uint8_t* __restrict__ db,
    const uint64_t* __restrict__ query_ntt,  // [db_rows_poly × poly_len] (single CRT)
    uint64_t* __restrict__ accum_global,
    NTTParams params,
    size_t db_cols, size_t db_rows_padded, size_t db_rows_poly)
{
    size_t col = blockIdx.x;
    if (col >= db_cols) return;

    size_t tid = threadIdx.x;
    uint32_t poly_len = params.poly_len;

    extern __shared__ uint64_t workspace[];  // [poly_len]

    // Global accumulator for this column: [poly_len]
    uint64_t* accum = accum_global + col * poly_len;

    // Initialize
    for (size_t i = tid; i < poly_len; i += blockDim.x)
        accum[i] = 0;
    __syncthreads();

    uint64_t mod0 = params.moduli[0];
    uint64_t bcr0 = params.barrett_cr[0];

    // Local accumulator in shared memory (avoid global writes per iteration)
    // We reuse workspace[] for NTT, then accumulate into accum_global at the end.
    // Strategy: use a separate shared-memory accumulator.
    // But we only have one shared array (workspace). So we accumulate in global
    // but reduce the Barrett overhead by skipping unnecessary reductions.

    for (size_t row = 0; row < db_rows_poly; row++) {
        // Load DB elements into workspace — DB values are u8 (0..255), already < Q.
        // No Barrett reduction needed on load.
        for (size_t z = tid; z < poly_len; z += blockDim.x) {
            size_t db_idx = col * db_rows_padded + row * poly_len + z;
            workspace[z] = (uint64_t)db[db_idx];
        }
        __syncthreads();

        // Forward NTT (single modulus, all threads)
        ntt_forward_alt_parallel(workspace, &params, 0, tid, blockDim.x);
        __syncthreads();

        // Pointwise multiply with query_ntt[row] and accumulate into global
        const uint64_t* query_row = &query_ntt[row * poly_len];
        for (size_t z = tid; z < poly_len; z += blockDim.x) {
            uint64_t p = barrett_raw_u64(query_row[z] * workspace[z], bcr0, mod0);
            accum[z] += p;
        }
        __syncthreads();

        // Periodic Barrett reduction of accumulator (every 256 rows)
        if ((row & 255) == 255 || row == db_rows_poly - 1) {
            for (size_t z = tid; z < poly_len; z += blockDim.x)
                accum[z] = barrett_raw_u64(accum[z], bcr0, mod0);
            __syncthreads();
        }
    }
}

// Final Barrett reduction + INTT + write.  Runs after sw_compute_hint_loop_kernel
// so host can place a CUDA event between the two phases.
__global__ void sw_compute_hint_finalize_kernel(
    uint64_t* __restrict__ hint_out,
    uint64_t* __restrict__ accum_global,
    NTTParams params,
    size_t db_cols)
{
    size_t col = blockIdx.x;
    if (col >= db_cols) return;

    size_t tid = threadIdx.x;
    uint32_t poly_len = params.poly_len;

    extern __shared__ uint64_t workspace[];  // [poly_len]

    uint64_t* accum = accum_global + col * poly_len;

    uint64_t mod0 = params.moduli[0];
    uint64_t bcr0 = params.barrett_cr[0];

    // Final Barrett reduction
    for (size_t z = tid; z < poly_len; z += blockDim.x)
        accum[z] = barrett_raw_u64(accum[z], bcr0, mod0);
    __syncthreads();

    // Copy to shared for INTT
    for (size_t z = tid; z < poly_len; z += blockDim.x)
        workspace[z] = accum[z];
    __syncthreads();

    // Inverse NTT (single modulus)
    ntt_inverse_alt_parallel(workspace, &params, 0, tid, blockDim.x);
    __syncthreads();

    // Write to output in row-major (poly_len × db_cols): hint[z * db_cols + col]
    // Precomp expects this layout for negacyclic permutation indexing.
    for (size_t z = tid; z < poly_len; z += blockDim.x)
        hint_out[z * db_cols + col] = workspace[z];
}

// ═══════════════════════════════════════════════════════════════
// Offline Context
// ═══════════════════════════════════════════════════════════════

struct SandwichOfflineContext {
    uint8_t* d_db;
    uint64_t* d_query_ntt;   // [db_rows_poly × poly_len]
    uint64_t* d_hint_0;      // [db_cols × poly_len]
    uint64_t* d_accum;       // [db_cols × poly_len]

    NTTParams ntt_params;
    size_t db_rows, db_rows_padded, db_cols, db_rows_poly;
    size_t poly_len;
};

// ═══════════════════════════════════════════════════════════════
// Extern "C" API
// ═══════════════════════════════════════════════════════════════

extern "C" {

void* sandwich_offline_init(
    const uint8_t* h_db,
    size_t db_rows, size_t db_rows_padded, size_t db_cols,
    const uint64_t* h_query_ntt,  // [db_rows_poly × poly_len] (host)
    uint32_t poly_len,
    const uint64_t* h_moduli,       // [1]
    const uint64_t* h_barrett_cr,   // [1]
    const uint64_t* h_forward_table,
    const uint64_t* h_forward_prime_table,
    const uint64_t* h_inverse_table,
    const uint64_t* h_inverse_prime_table,
    uint64_t modulus)
{
    auto* ctx = new SandwichOfflineContext();
    ctx->db_rows = db_rows;
    ctx->db_rows_padded = db_rows_padded;
    ctx->db_cols = db_cols;
    ctx->db_rows_poly = db_rows / poly_len;
    ctx->poly_len = poly_len;

    // Set up NTT params (single modulus)
    ctx->ntt_params.poly_len = poly_len;
    ctx->ntt_params.log2_poly_len = 31 - __builtin_clz(poly_len);
    ctx->ntt_params.crt_count = 1;
    ctx->ntt_params.modulus = modulus;
    ctx->ntt_params.mod0_inv_mod1 = 0;
    ctx->ntt_params.mod1_inv_mod0 = 0;
    ctx->ntt_params.barrett_cr_0_modulus = 0;
    ctx->ntt_params.barrett_cr_1_modulus = 0;

    size_t table_size = poly_len * sizeof(uint64_t);  // single CRT
    size_t mod_size = sizeof(uint64_t);

    // Allocate + copy NTT tables to device
    auto alloc_copy = [](auto*& d_ptr, const auto* h_ptr, size_t bytes) {
        SW_CUDA_ASSERT(cudaMalloc(&d_ptr, bytes));
        SW_CUDA_ASSERT(cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice));
    };

    alloc_copy(ctx->ntt_params.moduli, h_moduli, mod_size);
    alloc_copy(ctx->ntt_params.barrett_cr, h_barrett_cr, mod_size);
    alloc_copy(ctx->ntt_params.forward_table, h_forward_table, table_size);
    alloc_copy(ctx->ntt_params.forward_prime_table, h_forward_prime_table, table_size);
    alloc_copy(ctx->ntt_params.inverse_table, h_inverse_table, table_size);
    alloc_copy(ctx->ntt_params.inverse_prime_table, h_inverse_prime_table, table_size);

    // Also need n_inv_mod for INTT normalization
    // This is computed by the NTT infrastructure in ntt.cuh via ntt_inverse_alt_parallel

    // Upload DB (u8, p=256 — 1 byte per entry)
    size_t db_size = db_cols * db_rows_padded;
    alloc_copy(ctx->d_db, h_db, db_size);

    // Upload query NTT
    size_t query_size = ctx->db_rows_poly * poly_len * sizeof(uint64_t);
    alloc_copy(ctx->d_query_ntt, h_query_ntt, query_size);

    // Allocate output + accumulator
    size_t hint_size = db_cols * poly_len * sizeof(uint64_t);
    SW_CUDA_ASSERT(cudaMalloc(&ctx->d_hint_0, hint_size));
    SW_CUDA_ASSERT(cudaMalloc(&ctx->d_accum, hint_size));

    SW_LOG("SandwichPIR offline init: db=%zux%zu, db_rows_poly=%zu\n",
           db_rows, db_cols, ctx->db_rows_poly);

    return ctx;
}

int sandwich_offline_compute_hint(void* context) {
    auto* ctx = (SandwichOfflineContext*)context;
    if (!ctx) return -1;

    size_t poly_len = ctx->poly_len;
    size_t smem_size = poly_len * sizeof(uint64_t);
    bool verbose = sw_verbose();

    cudaEvent_t ev_start, ev_loop_done, ev_fin;
    if (verbose) {
        SW_CUDA_ASSERT(cudaEventCreate(&ev_start));
        SW_CUDA_ASSERT(cudaEventCreate(&ev_loop_done));
        SW_CUDA_ASSERT(cudaEventCreate(&ev_fin));
        SW_CUDA_ASSERT(cudaEventRecord(ev_start));
    }

    // Phase 1: NTT + pointwise multiply + accumulate (main loop)
    sw_compute_hint_loop_kernel<<<ctx->db_cols, 1024, smem_size>>>(
        ctx->d_db, ctx->d_query_ntt, ctx->d_accum,
        ctx->ntt_params, ctx->db_cols, ctx->db_rows_padded, ctx->db_rows_poly);
    SW_CUDA_ASSERT(cudaGetLastError());

    if (verbose)
        SW_CUDA_ASSERT(cudaEventRecord(ev_loop_done));

    // Phase 2: final Barrett reduction + INTT + write
    sw_compute_hint_finalize_kernel<<<ctx->db_cols, 1024, smem_size>>>(
        ctx->d_hint_0, ctx->d_accum,
        ctx->ntt_params, ctx->db_cols);
    SW_CUDA_ASSERT(cudaGetLastError());

    if (verbose)
        SW_CUDA_ASSERT(cudaEventRecord(ev_fin));

    SW_CUDA_ASSERT(cudaDeviceSynchronize());

    if (verbose) {
        float loop_ms, fin_ms, total_ms;
        cudaEventElapsedTime(&loop_ms, ev_start, ev_loop_done);
        cudaEventElapsedTime(&fin_ms, ev_loop_done, ev_fin);
        cudaEventElapsedTime(&total_ms, ev_start, ev_fin);
        SW_LOG("SW hint breakdown (%zu cols, %zu row-polys):\n",
               ctx->db_cols, ctx->db_rows_poly);
        SW_LOG("  ntt+mul+acc=%.2f  barrett+intt=%.2f  total=%.2f ms\n",
               loop_ms, fin_ms, total_ms);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_loop_done);
        cudaEventDestroy(ev_fin);
    }

    return 0;
}

// Download hint to host (poly_len × db_cols, already row-major on device)
int sandwich_offline_get_hint(void* context, uint64_t* h_hint_out) {
    auto* ctx = (SandwichOfflineContext*)context;
    if (!ctx) return -1;

    size_t hint_elems = ctx->db_cols * ctx->poly_len;
    SW_CUDA_ASSERT(cudaMemcpy(h_hint_out, ctx->d_hint_0,
                               hint_elems * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    return 0;
}

// Get device pointer to hint (row-major: poly_len × db_cols).
uint64_t* sandwich_offline_get_hint_device_ptr(void* context) {
    auto* ctx = (SandwichOfflineContext*)context;
    return ctx ? ctx->d_hint_0 : nullptr;
}

// Take ownership of d_hint_0: returns pointer and nulls it so free() won't touch it.
uint64_t* sandwich_offline_take_hint_device_ptr(void* context) {
    auto* ctx = (SandwichOfflineContext*)context;
    if (!ctx) return nullptr;
    uint64_t* ptr = ctx->d_hint_0;
    ctx->d_hint_0 = nullptr;
    return ptr;
}

void sandwich_offline_free(void* context) {
    auto* ctx = (SandwichOfflineContext*)context;
    if (!ctx) return;
    if (ctx->d_db) cudaFree(ctx->d_db);
    if (ctx->d_query_ntt) cudaFree(ctx->d_query_ntt);
    if (ctx->d_hint_0) cudaFree(ctx->d_hint_0);
    if (ctx->d_accum) cudaFree(ctx->d_accum);
    // NTT table device pointers
    if (ctx->ntt_params.moduli) cudaFree(ctx->ntt_params.moduli);
    if (ctx->ntt_params.barrett_cr) cudaFree(ctx->ntt_params.barrett_cr);
    if (ctx->ntt_params.forward_table) cudaFree(ctx->ntt_params.forward_table);
    if (ctx->ntt_params.forward_prime_table) cudaFree(ctx->ntt_params.forward_prime_table);
    if (ctx->ntt_params.inverse_table) cudaFree(ctx->ntt_params.inverse_table);
    if (ctx->ntt_params.inverse_prime_table) cudaFree(ctx->ntt_params.inverse_prime_table);
    delete ctx;
}

} // extern "C"
