/**
 * PIR GEMM Benchmark
 *
 * Measures throughput of DB × queries for SimplePIR-style computation.
 *
 * Modes:
 *   db32_q32  — uint32 DB × uint32 queries  (baseline)
 *   db32_q64  — uint32 DB × uint64 queries
 *   db16_q32  — uint16 DB × uint32 queries  (half DB size)
 *   db16_q64  — uint16 DB × uint64 queries
 *   db16_crt     — CRT: two (uint16 DB × uint32 queries), combined time
 *   db8_q32      — uint8 DB × uint32 queries → uint32 accum (DoublePIR Step 1, SIMT)
 *   db32_crt_i64 — CRT: two (int32 DB × int32 queries → int64 accum), matches actual CUTLASS kernel
 *   db8_q32_tc   — Tensor core: 4x int8 GEMM with alpha/beta folding (DoublePIR Step 1, cuBLAS)
 *   db16_q64_tc  — Tensor core: 15x int8 GEMM (uint16 DB x uint64 query, SimplePIR offline)
 *   db8_q32_tc_cutlass  — CUTLASS uint8 TC: 1x wide GEMM (uint8 DB x uint32 query, SM80+)
 *   db16_q64_tc_cutlass — CUTLASS uint8 TC: 2x wide GEMM (uint16 DB x uint64 query, SM80+)
 *
 * Usage:
 *   ./pir_bench [--mode db32_q32|...|db16_q64_tc_cutlass|all]
 *               [--batches 1,2,4,...] [--log_dim N] [--warmup N] [--iters N]
 *
 * Requires CUTLASS headers (clone https://github.com/NVIDIA/cutlass)
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <sstream>

#include <cublas_v2.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

// ---------- Error checking ----------

#define CUDA_CHECK(status)                                           \
    {                                                                \
        cudaError_t error = status;                                  \
        if (error != cudaSuccess) {                                  \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__       \
                      << std::endl;                                  \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    }

#define CUTLASS_CHECK(status)                                        \
    {                                                                \
        cutlass::Status error = status;                              \
        if (error != cutlass::Status::kSuccess) {                    \
            std::cerr << "CUTLASS error: "                           \
                      << cutlassGetStatusString(error)               \
                      << " at " << __FILE__ << ":" << __LINE__       \
                      << std::endl;                                  \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    }

#define CUBLAS_CHECK(status)                                          \
    {                                                                \
        cublasStatus_t error = status;                               \
        if (error != CUBLAS_STATUS_SUCCESS) {                        \
            std::cerr << "cuBLAS error: " << (int)error              \
                      << " at " << __FILE__ << ":" << __LINE__       \
                      << std::endl;                                  \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    }

// ---------- Kernel: decompose int32 query into 4 int8 byte slices ----------

__global__ void decompose_query_bytes_kernel(
    int8_t* __restrict__ q0,
    int8_t* __restrict__ q1,
    int8_t* __restrict__ q2,
    int8_t* __restrict__ q3,
    const int32_t* __restrict__ query,
    size_t count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int32_t v = query[idx];
        q0[idx] = (int8_t)(v & 0xFF);
        q1[idx] = (int8_t)((v >> 8) & 0xFF);
        q2[idx] = (int8_t)((v >> 16) & 0xFF);
        q3[idx] = (int8_t)((v >> 24) & 0xFF);
    }
}

// ---------- Kernel: decompose uint16 into 2 int8 byte slices ----------

__global__ void decompose_i16_bytes_kernel(
    int8_t* __restrict__ b0,
    int8_t* __restrict__ b1,
    const uint16_t* __restrict__ data,
    size_t count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint16_t v = data[idx];
        b0[idx] = (int8_t)(v & 0xFF);
        b1[idx] = (int8_t)((v >> 8) & 0xFF);
    }
}

// ---------- Kernel: decompose uint64 into 8 int8 byte slices ----------

__global__ void decompose_i64_bytes_kernel(
    int8_t* __restrict__ b0, int8_t* __restrict__ b1,
    int8_t* __restrict__ b2, int8_t* __restrict__ b3,
    int8_t* __restrict__ b4, int8_t* __restrict__ b5,
    int8_t* __restrict__ b6, int8_t* __restrict__ b7,
    const uint64_t* __restrict__ data,
    size_t count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint64_t v = data[idx];
        b0[idx] = (int8_t)(v & 0xFF);
        b1[idx] = (int8_t)((v >> 8) & 0xFF);
        b2[idx] = (int8_t)((v >> 16) & 0xFF);
        b3[idx] = (int8_t)((v >> 24) & 0xFF);
        b4[idx] = (int8_t)((v >> 32) & 0xFF);
        b5[idx] = (int8_t)((v >> 40) & 0xFF);
        b6[idx] = (int8_t)((v >> 48) & 0xFF);
        b7[idx] = (int8_t)((v >> 56) & 0xFF);
    }
}

// ---------- Kernels for CUTLASS uint8 TC path ----------

// Decompose uint16 DB into 2 uint8 byte slices
__global__ void decompose_u16_bytes_kernel(
    uint8_t* __restrict__ b0,
    uint8_t* __restrict__ b1,
    const uint16_t* __restrict__ data,
    size_t count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint16_t v = data[idx];
        b0[idx] = (uint8_t)(v & 0xFF);
        b1[idx] = (uint8_t)((v >> 8) & 0xFF);
    }
}

// Decompose uint64 queries into packed uint8 layout: K × (8*N) contiguous
// Byte slice i occupies columns [i*N, (i+1)*N) in col-major with stride K
__global__ void decompose_u64_bytes_packed_kernel(
    uint8_t* __restrict__ out,  // K * 8 * N contiguous
    const uint64_t* __restrict__ data,  // K * N col-major
    size_t K, size_t N)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * N) return;
    uint64_t v = data[idx];
    size_t stride = K * N;
    out[idx + 0 * stride] = (uint8_t)(v & 0xFF);
    out[idx + 1 * stride] = (uint8_t)((v >> 8) & 0xFF);
    out[idx + 2 * stride] = (uint8_t)((v >> 16) & 0xFF);
    out[idx + 3 * stride] = (uint8_t)((v >> 24) & 0xFF);
    out[idx + 4 * stride] = (uint8_t)((v >> 32) & 0xFF);
    out[idx + 5 * stride] = (uint8_t)((v >> 40) & 0xFF);
    out[idx + 6 * stride] = (uint8_t)((v >> 48) & 0xFF);
    out[idx + 7 * stride] = (uint8_t)((v >> 56) & 0xFF);
}

// Accumulate GEMM 0 (db_b=0): 8 byte products → uint64
// gemm_out is col-major M × (8*N), stride M
__global__ void accumulate_db0_kernel(
    uint64_t* __restrict__ accum,          // M × N col-major
    const int32_t* __restrict__ gemm_out,  // M × (8*N) col-major
    size_t M, size_t N)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    size_t m = idx % M;
    size_t n = idx / M;

    uint64_t acc = 0;
    #pragma unroll
    for (int q_b = 0; q_b < 8; q_b++) {
        uint64_t val = (uint64_t)(uint32_t)gemm_out[m + (q_b * N + n) * M];
        acc += val << (8 * q_b);
    }
    accum[idx] = acc;
}

// Accumulate GEMM 1 (db_b=1): 7 byte products → add to uint64
// Skip q_b=7 (power 8 → 0 mod 2^64)
__global__ void accumulate_db1_kernel(
    uint64_t* __restrict__ accum,          // M × N col-major (read-modify-write)
    const int32_t* __restrict__ gemm_out,  // M × (8*N) col-major
    size_t M, size_t N)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    size_t m = idx % M;
    size_t n = idx / M;

    uint64_t acc = 0;
    #pragma unroll
    for (int q_b = 0; q_b < 7; q_b++) {
        uint64_t val = (uint64_t)(uint32_t)gemm_out[m + (q_b * N + n) * M];
        acc += val << (8 * (q_b + 1));
    }
    accum[idx] += acc;
}

// Decompose uint16 DB into vertically stacked (2M)×K row-major uint8:
// First M rows = low bytes, next M rows = high bytes
__global__ void decompose_u16_stacked_kernel(
    uint8_t* __restrict__ out,  // (2M)×K row-major
    const uint16_t* __restrict__ data,  // M×K row-major
    size_t count)  // M*K
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint16_t v = data[idx];
        out[idx] = (uint8_t)(v & 0xFF);
        out[count + idx] = (uint8_t)((v >> 8) & 0xFF);
    }
}

// Accumulate stacked GEMM output (2M)×(8N) → M×N uint64
// gemm_out is col-major (2M)×(8N), stride 2M
// Rows [0,M) = db_b0 products, rows [M,2M) = db_b1 products
__global__ void accumulate_db16_q64_stacked_kernel(
    uint64_t* __restrict__ accum,          // M × N col-major
    const int32_t* __restrict__ gemm_out,  // (2M) × (8*N) col-major, stride 2M
    size_t M, size_t N)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    size_t m = idx % M;
    size_t n = idx / M;
    size_t stride = 2 * M;

    uint64_t acc = 0;
    // db_b0 contributions: 8 terms, shifts 0,8,16,...,56
    #pragma unroll
    for (int q_b = 0; q_b < 8; q_b++) {
        uint64_t val = (uint64_t)(uint32_t)gemm_out[m + (q_b * N + n) * stride];
        acc += val << (8 * q_b);
    }
    // db_b1 contributions: 7 terms, shifts 8,16,...,56 (skip q_b=7 → shift 64 ≡ 0)
    #pragma unroll
    for (int q_b = 0; q_b < 7; q_b++) {
        uint64_t val = (uint64_t)(uint32_t)gemm_out[M + m + (q_b * N + n) * stride];
        acc += val << (8 * (q_b + 1));
    }
    accum[idx] = acc;
}

// Decompose uint32 queries into packed uint8 layout: K × (4*N) contiguous
// Byte slice i occupies columns [i*N, (i+1)*N) in col-major with stride K
__global__ void decompose_u32_bytes_packed_kernel(
    uint8_t* __restrict__ out,  // K * 4 * N contiguous
    const uint32_t* __restrict__ data,  // K * N col-major
    size_t K, size_t N)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * N) return;
    uint32_t v = data[idx];
    size_t stride = K * N;
    out[idx + 0 * stride] = (uint8_t)(v & 0xFF);
    out[idx + 1 * stride] = (uint8_t)((v >> 8) & 0xFF);
    out[idx + 2 * stride] = (uint8_t)((v >> 16) & 0xFF);
    out[idx + 3 * stride] = (uint8_t)((v >> 24) & 0xFF);
}

// Accumulate 1 GEMM (uint8 DB × 4 query bytes) → uint32
// gemm_out is col-major M × (4*N), stride M
__global__ void accumulate_db8_q32_kernel(
    uint32_t* __restrict__ accum,          // M × N col-major
    const int32_t* __restrict__ gemm_out,  // M × (4*N) col-major
    size_t M, size_t N)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    size_t m = idx % M;
    size_t n = idx / M;

    uint32_t acc = 0;
    #pragma unroll
    for (int q_b = 0; q_b < 4; q_b++) {
        uint32_t val = (uint32_t)gemm_out[m + (q_b * N + n) * M];
        acc += val << (8 * q_b);
    }
    accum[idx] = acc;
}

// ---------- Benchmark kernel ----------

template <typename DbT, typename QueryT>
struct PirBench {
    using Rows = cutlass::layout::RowMajor;
    using Cols = cutlass::layout::ColumnMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        DbT,    Rows,   // A (database)
        QueryT, Cols,   // B (queries)
        QueryT, Rows    // C/D (output)
    >;

    static void print_header(uint64_t M, uint64_t K, int warmup, int iters, const std::string& label) {
        const double GiB = 1024.0 * 1024.0 * 1024.0;
        std::cout << "\n=== " << label << " ===" << std::endl;
        std::cout << "DB: " << M << " x " << K << " uint" << (sizeof(DbT) * 8)
                  << " (" << std::fixed << std::setprecision(1)
                  << (double(M) * K * sizeof(DbT)) / GiB << " GiB)" << std::endl;
        std::cout << "Query/Output type: uint" << (sizeof(QueryT) * 8) << std::endl;

        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        std::cout << "GPU memory: " << std::fixed << std::setprecision(1)
                  << total_mem / GiB << " GiB total, "
                  << free_mem / GiB << " GiB free" << std::endl;

        cudaDeviceProp props;
        CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
        std::cout << "GPU: " << props.name << std::endl;
        std::cout << "HBM BW (theoretical): ~"
                  << (2.0 * props.memoryClockRate * (props.memoryBusWidth / 8.0) / 1.0e6)
                  << " GB/s" << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        std::cout << std::setw(8) << "Batch"
                  << std::setw(12) << "Comp (ms)"
                  << std::setw(12) << "E2E (ms)"
                  << std::setw(12) << "E2E QPS"
                  << std::setw(18) << "Eff Tput (GB/s)"
                  << std::setw(18) << "HW BW (GB/s)"
                  << std::endl;
        std::cout << std::string(80, '-') << std::endl;
    }

    static void run(uint64_t M, uint64_t K, const std::vector<int>& batch_sizes,
                    int warmup, int iters, const std::string& label) {
        print_header(M, K, warmup, iters, label);

        // Allocate DB once
        cutlass::HostTensor<DbT, Rows> tensor_A({int(M), int(K)});
        cutlass::reference::host::TensorFill(tensor_A.host_view());
        tensor_A.sync_device();

        for (int N : batch_sizes) {
            cutlass::gemm::GemmCoord problem_size(M, N, K);

            cutlass::HostTensor<QueryT, Cols> tensor_B(problem_size.kn());
            cutlass::HostTensor<QueryT, Rows> tensor_C(problem_size.mn());
            cutlass::HostTensor<QueryT, Rows> tensor_D(problem_size.mn());

            cutlass::reference::host::TensorFill(tensor_B.host_view());
            cutlass::reference::host::TensorFill(tensor_C.host_view());
            cutlass::reference::host::TensorFill(tensor_D.host_view());

            tensor_B.sync_device();
            tensor_C.sync_device();
            tensor_D.sync_device();

            QueryT alpha = QueryT(1);
            QueryT beta  = QueryT(0);

            typename Gemm::Arguments arguments{
                problem_size,
                tensor_A.device_ref(),
                tensor_B.device_ref(),
                tensor_C.device_ref(),
                tensor_D.device_ref(),
                {alpha, beta},
                1
            };

            Gemm gemm_op;
            cutlass::Status status = gemm_op.can_implement(arguments);
            if (status != cutlass::Status::kSuccess) {
                std::cerr << "Cannot implement GEMM for batch=" << N
                          << ": " << cutlassGetStatusString(status) << std::endl;
                continue;
            }

            size_t workspace_size = Gemm::get_workspace_size(arguments);
            cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
            status = gemm_op.initialize(arguments, workspace.get());
            CUTLASS_CHECK(status);

            for (int i = 0; i < warmup; i++) {
                status = gemm_op();
                CUTLASS_CHECK(status);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            // Compute-only timing (GPU events)
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < iters; i++) {
                status = gemm_op();
                CUTLASS_CHECK(status);
            }
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float elapsed_ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            double comp_ms = elapsed_ms / iters;

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));

            // E2E timing (wall clock: H->D query + compute + D->H result)
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; i++) {
                tensor_B.sync_device();
                status = gemm_op();
                CUTLASS_CHECK(status);
                tensor_D.sync_host();
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double e2e_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

            double db_bytes = double(M) * K * sizeof(DbT);
            double e2e_qps = N / (e2e_ms / 1000.0);
            double eff_tput_gbs = (db_bytes * N) / (e2e_ms / 1000.0) / 1.0e9;

            double query_bytes = double(K) * N * sizeof(QueryT);
            double output_bytes = double(M) * N * sizeof(QueryT);
            double hw_bw_gbs = (db_bytes + query_bytes + output_bytes) / (comp_ms / 1000.0) / 1.0e9;

            std::cout << std::setw(8) << N
                      << std::setw(12) << std::fixed << std::setprecision(2) << comp_ms
                      << std::setw(12) << std::fixed << std::setprecision(2) << e2e_ms
                      << std::setw(12) << std::fixed << std::setprecision(0) << e2e_qps
                      << std::setw(18) << std::fixed << std::setprecision(1) << eff_tput_gbs
                      << std::setw(18) << std::fixed << std::setprecision(1) << hw_bw_gbs
                      << std::endl;
        }

        std::cout << std::string(80, '-') << std::endl;
        std::cout << "Comp    = GPU compute only (no PCIe)" << std::endl;
        std::cout << "E2E     = H->D query upload + compute + D->H result download" << std::endl;
        std::cout << "Eff Tput = (DB_size * batch) / E2E_time  [amortized]" << std::endl;
        std::cout << "HW BW   = (DB + queries + output) / comp_time" << std::endl;
    }
};

// ---------- CRT benchmark: two uint16 x uint32 GEMMs ----------

void run_crt(uint64_t M, uint64_t K, const std::vector<int>& batch_sizes,
             int warmup, int iters) {
    using Rows = cutlass::layout::RowMajor;
    using Cols = cutlass::layout::ColumnMajor;
    using Gemm = cutlass::gemm::device::Gemm<uint16_t, Rows, uint32_t, Cols, uint32_t, Rows>;

    const double GiB = 1024.0 * 1024.0 * 1024.0;
    std::cout << "\n=== CRT: 2x (uint16 DB x uint32 query) ===" << std::endl;
    std::cout << "DB: " << M << " x " << K << " uint16 ("
              << std::fixed << std::setprecision(1)
              << (double(M) * K * sizeof(uint16_t)) / GiB << " GiB)" << std::endl;
    std::cout << "Each GEMM: uint16 x uint32 -> uint32, combined time for both" << std::endl;

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU memory: " << std::fixed << std::setprecision(1)
              << total_mem / GiB << " GiB total, "
              << free_mem / GiB << " GiB free" << std::endl;

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    std::cout << std::setw(8) << "Batch"
              << std::setw(12) << "Comp (ms)"
              << std::setw(12) << "E2E (ms)"
              << std::setw(12) << "E2E QPS"
              << std::setw(18) << "Eff Tput (GB/s)"
              << std::setw(18) << "HW BW (GB/s)"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Allocate DB once
    cutlass::HostTensor<uint16_t, Rows> tensor_A({int(M), int(K)});
    cutlass::reference::host::TensorFill(tensor_A.host_view());
    tensor_A.sync_device();

    for (int N : batch_sizes) {
        cutlass::gemm::GemmCoord problem_size(M, N, K);

        // Two sets of queries (CRT residues)
        cutlass::HostTensor<uint32_t, Cols> tensor_B1(problem_size.kn());
        cutlass::HostTensor<uint32_t, Rows> tensor_C1(problem_size.mn());
        cutlass::HostTensor<uint32_t, Rows> tensor_D1(problem_size.mn());
        cutlass::HostTensor<uint32_t, Cols> tensor_B2(problem_size.kn());
        cutlass::HostTensor<uint32_t, Rows> tensor_C2(problem_size.mn());
        cutlass::HostTensor<uint32_t, Rows> tensor_D2(problem_size.mn());

        cutlass::reference::host::TensorFill(tensor_B1.host_view());
        cutlass::reference::host::TensorFill(tensor_C1.host_view());
        cutlass::reference::host::TensorFill(tensor_D1.host_view());
        cutlass::reference::host::TensorFill(tensor_B2.host_view());
        cutlass::reference::host::TensorFill(tensor_C2.host_view());
        cutlass::reference::host::TensorFill(tensor_D2.host_view());

        tensor_B1.sync_device(); tensor_C1.sync_device(); tensor_D1.sync_device();
        tensor_B2.sync_device(); tensor_C2.sync_device(); tensor_D2.sync_device();

        uint32_t alpha = 1, beta = 0;

        typename Gemm::Arguments args1{
            problem_size, tensor_A.device_ref(), tensor_B1.device_ref(),
            tensor_C1.device_ref(), tensor_D1.device_ref(), {alpha, beta}, 1
        };
        typename Gemm::Arguments args2{
            problem_size, tensor_A.device_ref(), tensor_B2.device_ref(),
            tensor_C2.device_ref(), tensor_D2.device_ref(), {alpha, beta}, 1
        };

        Gemm gemm_op1, gemm_op2;
        cutlass::Status s1 = gemm_op1.can_implement(args1);
        cutlass::Status s2 = gemm_op2.can_implement(args2);
        if (s1 != cutlass::Status::kSuccess || s2 != cutlass::Status::kSuccess) {
            std::cerr << "Cannot implement CRT GEMM for batch=" << N << std::endl;
            continue;
        }

        size_t ws1 = Gemm::get_workspace_size(args1);
        size_t ws2 = Gemm::get_workspace_size(args2);
        cutlass::device_memory::allocation<uint8_t> workspace1(ws1), workspace2(ws2);
        CUTLASS_CHECK(gemm_op1.initialize(args1, workspace1.get()));
        CUTLASS_CHECK(gemm_op2.initialize(args2, workspace2.get()));

        for (int i = 0; i < warmup; i++) {
            CUTLASS_CHECK(gemm_op1());
            CUTLASS_CHECK(gemm_op2());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compute-only timing (GPU events)
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            CUTLASS_CHECK(gemm_op1());
            CUTLASS_CHECK(gemm_op2());
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        double comp_ms = elapsed_ms / iters;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // E2E timing (wall clock: H->D queries + compute + D->H results)
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            tensor_B1.sync_device(); tensor_B2.sync_device();
            CUTLASS_CHECK(gemm_op1());
            CUTLASS_CHECK(gemm_op2());
            tensor_D1.sync_host(); tensor_D2.sync_host();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double e2e_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

        double db_bytes = double(M) * K * sizeof(uint16_t);
        double e2e_qps = N / (e2e_ms / 1000.0);
        double eff_tput_gbs = (db_bytes * N) / (e2e_ms / 1000.0) / 1.0e9;

        // HW BW: read DB twice + read 2x queries + write 2x outputs
        double query_bytes = 2.0 * double(K) * N * sizeof(uint32_t);
        double output_bytes = 2.0 * double(M) * N * sizeof(uint32_t);
        double hw_bw_gbs = (2.0 * db_bytes + query_bytes + output_bytes) / (comp_ms / 1000.0) / 1.0e9;

        std::cout << std::setw(8) << N
                  << std::setw(12) << std::fixed << std::setprecision(2) << comp_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << e2e_ms
                  << std::setw(12) << std::fixed << std::setprecision(0) << e2e_qps
                  << std::setw(18) << std::fixed << std::setprecision(1) << eff_tput_gbs
                  << std::setw(18) << std::fixed << std::setprecision(1) << hw_bw_gbs
                  << std::endl;
    }

    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Comp    = GPU compute only (no PCIe)" << std::endl;
    std::cout << "E2E     = H->D query upload + compute + D->H result download" << std::endl;
    std::cout << "Eff Tput = (DB_size * batch) / E2E_time  [amortized]" << std::endl;
    std::cout << "HW BW   = (2*DB + 2*queries + 2*output) / comp_time  [both GEMMs]" << std::endl;
}

// ---------- Mixed-precision: uint8 DB x uint32 query -> uint32 accum ----------
// Matches DoublePIR Step 1: DB elements are uint8 (unpacked), queries are uint32

void run_db8_q32(uint64_t M, uint64_t K, const std::vector<int>& batch_sizes,
                 int warmup, int iters) {
    using Rows = cutlass::layout::RowMajor;
    using Cols = cutlass::layout::ColumnMajor;
    using Gemm = cutlass::gemm::device::Gemm<
        uint8_t,                                // ElementA (DB)
        Rows,                                   // LayoutA
        uint32_t,                               // ElementB (Query)
        Cols,                                   // LayoutB
        uint32_t,                               // ElementC (Output)
        Cols,                                   // LayoutC
        uint32_t,                               // ElementAccumulator
        cutlass::arch::OpClassSimt,             // OperatorClass
        cutlass::arch::Sm50,                    // ArchTag
        cutlass::gemm::GemmShape<64, 64, 8>,    // ThreadblockShape
        cutlass::gemm::GemmShape<32, 32, 8>,    // WarpShape
        cutlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape
        cutlass::epilogue::thread::LinearCombination<uint32_t, 1, uint32_t, uint32_t>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
        2                                        // Stages
    >;

    const double GiB = 1024.0 * 1024.0 * 1024.0;
    std::cout << "\n=== uint8 DB x uint32 query -> uint32 (DoublePIR Step 1) ===" << std::endl;
    std::cout << "DB: " << M << " x " << K << " uint8 ("
              << std::fixed << std::setprecision(1)
              << (double(M) * K * sizeof(uint8_t)) / GiB << " GiB)" << std::endl;

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU memory: " << std::fixed << std::setprecision(1)
              << total_mem / GiB << " GiB total, "
              << free_mem / GiB << " GiB free" << std::endl;

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "HBM BW (theoretical): ~"
              << (2.0 * props.memoryClockRate * (props.memoryBusWidth / 8.0) / 1.0e6)
              << " GB/s" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    std::cout << std::setw(8) << "Batch"
              << std::setw(12) << "Comp (ms)"
              << std::setw(12) << "E2E (ms)"
              << std::setw(12) << "E2E QPS"
              << std::setw(18) << "Eff Tput (GB/s)"
              << std::setw(18) << "HW BW (GB/s)"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    cutlass::HostTensor<uint8_t, Rows> tensor_A({int(M), int(K)});
    cutlass::reference::host::TensorFill(tensor_A.host_view());
    tensor_A.sync_device();

    for (int N : batch_sizes) {
        cutlass::gemm::GemmCoord problem_size(M, N, K);

        cutlass::HostTensor<uint32_t, Cols> tensor_B(problem_size.kn());
        cutlass::HostTensor<uint32_t, Cols> tensor_C(problem_size.mn());
        cutlass::HostTensor<uint32_t, Cols> tensor_D(problem_size.mn());

        cutlass::reference::host::TensorFill(tensor_B.host_view());
        cutlass::reference::host::TensorFill(tensor_C.host_view());
        cutlass::reference::host::TensorFill(tensor_D.host_view());

        tensor_B.sync_device(); tensor_C.sync_device(); tensor_D.sync_device();

        uint32_t alpha = 1, beta = 0;

        typename Gemm::Arguments arguments{
            problem_size,
            tensor_A.device_ref(),
            tensor_B.device_ref(),
            tensor_C.device_ref(),
            tensor_D.device_ref(),
            {alpha, beta},
            1
        };

        Gemm gemm_op;
        cutlass::Status status = gemm_op.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "Cannot implement db8_q32 GEMM for batch=" << N
                      << ": " << cutlassGetStatusString(status) << std::endl;
            continue;
        }

        size_t workspace_size = Gemm::get_workspace_size(arguments);
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
        status = gemm_op.initialize(arguments, workspace.get());
        CUTLASS_CHECK(status);

        for (int i = 0; i < warmup; i++) {
            status = gemm_op();
            CUTLASS_CHECK(status);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compute-only timing (GPU events)
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            status = gemm_op();
            CUTLASS_CHECK(status);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        double comp_ms = elapsed_ms / iters;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // E2E timing (wall clock: H->D query + compute + D->H result)
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            tensor_B.sync_device();
            status = gemm_op();
            CUTLASS_CHECK(status);
            tensor_D.sync_host();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double e2e_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

        double db_bytes = double(M) * K * sizeof(uint8_t);
        double e2e_qps = N / (e2e_ms / 1000.0);
        double eff_tput_gbs = (db_bytes * N) / (e2e_ms / 1000.0) / 1.0e9;

        double query_bytes = double(K) * N * sizeof(uint32_t);
        double output_bytes = double(M) * N * sizeof(uint32_t);
        double hw_bw_gbs = (db_bytes + query_bytes + output_bytes) / (comp_ms / 1000.0) / 1.0e9;

        std::cout << std::setw(8) << N
                  << std::setw(12) << std::fixed << std::setprecision(2) << comp_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << e2e_ms
                  << std::setw(12) << std::fixed << std::setprecision(0) << e2e_qps
                  << std::setw(18) << std::fixed << std::setprecision(1) << eff_tput_gbs
                  << std::setw(18) << std::fixed << std::setprecision(1) << hw_bw_gbs
                  << std::endl;
    }

    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Comp    = GPU compute only (no PCIe)" << std::endl;
    std::cout << "E2E     = H->D query upload + compute + D->H result download" << std::endl;
    std::cout << "Eff Tput = (DB_uint8_size * batch) / E2E_time  [amortized]" << std::endl;
    std::cout << "HW BW   = (DB + queries + output) / comp_time" << std::endl;
}

// ---------- CRT benchmark: 2x (int32 DB x int32 query -> int64 accum) ----------
// This matches the actual CUTLASS kernel used in online_kernel_sp_cutlass.cu

void run_crt_i64(uint64_t M, uint64_t K, const std::vector<int>& batch_sizes,
                 int warmup, int iters) {
    using Rows = cutlass::layout::RowMajor;
    using Cols = cutlass::layout::ColumnMajor;
    using Gemm = cutlass::gemm::device::Gemm<
        int32_t, Rows,
        int32_t, Cols,
        int64_t, Cols,
        int64_t,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        cutlass::gemm::GemmShape<64, 64, 8>,
        cutlass::gemm::GemmShape<32, 32, 8>,
        cutlass::gemm::GemmShape<1, 1, 1>,
        cutlass::epilogue::thread::LinearCombination<int64_t, 1, int64_t, int64_t>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
        2
    >;

    const double GiB = 1024.0 * 1024.0 * 1024.0;
    std::cout << "\n=== CRT-i64: 2x (int32 DB x int32 query -> int64 accum) ===" << std::endl;
    std::cout << "DB: " << M << " x " << K << " int32 (widened from uint16, "
              << std::fixed << std::setprecision(1)
              << (double(M) * K * sizeof(int32_t)) / GiB << " GiB)" << std::endl;
    std::cout << "Each GEMM: int32 x int32 -> int64, combined time for both" << std::endl;

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU memory: " << std::fixed << std::setprecision(1)
              << total_mem / GiB << " GiB total, "
              << free_mem / GiB << " GiB free" << std::endl;

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "HBM BW (theoretical): ~"
              << (2.0 * props.memoryClockRate * (props.memoryBusWidth / 8.0) / 1.0e6)
              << " GB/s" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    std::cout << std::setw(8) << "Batch"
              << std::setw(12) << "Comp (ms)"
              << std::setw(12) << "E2E (ms)"
              << std::setw(12) << "E2E QPS"
              << std::setw(18) << "Eff Tput (GB/s)"
              << std::setw(18) << "HW BW (GB/s)"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Allocate widened DB (int32, simulating uint16 -> int32 widen)
    cutlass::HostTensor<int32_t, Rows> tensor_A({int(M), int(K)});
    cutlass::reference::host::TensorFill(tensor_A.host_view());
    tensor_A.sync_device();

    for (int N : batch_sizes) {
        cutlass::gemm::GemmCoord problem_size(M, N, K);

        cutlass::HostTensor<int32_t, Cols> tensor_B1(problem_size.kn());
        cutlass::HostTensor<int64_t, Cols> tensor_C1(problem_size.mn());
        cutlass::HostTensor<int64_t, Cols> tensor_D1(problem_size.mn());
        cutlass::HostTensor<int32_t, Cols> tensor_B2(problem_size.kn());
        cutlass::HostTensor<int64_t, Cols> tensor_C2(problem_size.mn());
        cutlass::HostTensor<int64_t, Cols> tensor_D2(problem_size.mn());

        cutlass::reference::host::TensorFill(tensor_B1.host_view());
        cutlass::reference::host::TensorFill(tensor_C1.host_view());
        cutlass::reference::host::TensorFill(tensor_D1.host_view());
        cutlass::reference::host::TensorFill(tensor_B2.host_view());
        cutlass::reference::host::TensorFill(tensor_C2.host_view());
        cutlass::reference::host::TensorFill(tensor_D2.host_view());

        tensor_B1.sync_device(); tensor_C1.sync_device(); tensor_D1.sync_device();
        tensor_B2.sync_device(); tensor_C2.sync_device(); tensor_D2.sync_device();

        int64_t alpha = 1, beta = 0;

        typename Gemm::Arguments args1{
            problem_size, tensor_A.device_ref(), tensor_B1.device_ref(),
            tensor_C1.device_ref(), tensor_D1.device_ref(), {alpha, beta}, 1
        };
        typename Gemm::Arguments args2{
            problem_size, tensor_A.device_ref(), tensor_B2.device_ref(),
            tensor_C2.device_ref(), tensor_D2.device_ref(), {alpha, beta}, 1
        };

        Gemm gemm_op1, gemm_op2;
        cutlass::Status s1 = gemm_op1.can_implement(args1);
        cutlass::Status s2 = gemm_op2.can_implement(args2);
        if (s1 != cutlass::Status::kSuccess || s2 != cutlass::Status::kSuccess) {
            std::cerr << "Cannot implement CRT-i64 GEMM for batch=" << N
                      << ": " << cutlassGetStatusString(s1) << " / "
                      << cutlassGetStatusString(s2) << std::endl;
            continue;
        }

        size_t ws1 = Gemm::get_workspace_size(args1);
        size_t ws2 = Gemm::get_workspace_size(args2);
        cutlass::device_memory::allocation<uint8_t> workspace1(ws1), workspace2(ws2);
        CUTLASS_CHECK(gemm_op1.initialize(args1, workspace1.get()));
        CUTLASS_CHECK(gemm_op2.initialize(args2, workspace2.get()));

        for (int i = 0; i < warmup; i++) {
            CUTLASS_CHECK(gemm_op1());
            CUTLASS_CHECK(gemm_op2());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compute-only timing (GPU events)
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            CUTLASS_CHECK(gemm_op1());
            CUTLASS_CHECK(gemm_op2());
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        double comp_ms = elapsed_ms / iters;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // E2E timing (wall clock: H->D queries + compute + D->H results)
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            tensor_B1.sync_device(); tensor_B2.sync_device();
            CUTLASS_CHECK(gemm_op1());
            CUTLASS_CHECK(gemm_op2());
            tensor_D1.sync_host(); tensor_D2.sync_host();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double e2e_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

        // Effective throughput uses original DB size (uint16, since that's the real data)
        double db_bytes_real = double(M) * K * sizeof(uint16_t);
        double e2e_qps = N / (e2e_ms / 1000.0);
        double eff_tput_gbs = (db_bytes_real * N) / (e2e_ms / 1000.0) / 1.0e9;

        // HW BW: read int32 DB twice + read 2x int32 queries + write 2x int64 outputs
        double db_bytes_gpu = double(M) * K * sizeof(int32_t);
        double query_bytes = 2.0 * double(K) * N * sizeof(int32_t);
        double output_bytes = 2.0 * double(M) * N * sizeof(int64_t);
        double hw_bw_gbs = (2.0 * db_bytes_gpu + query_bytes + output_bytes) / (comp_ms / 1000.0) / 1.0e9;

        std::cout << std::setw(8) << N
                  << std::setw(12) << std::fixed << std::setprecision(2) << comp_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << e2e_ms
                  << std::setw(12) << std::fixed << std::setprecision(0) << e2e_qps
                  << std::setw(18) << std::fixed << std::setprecision(1) << eff_tput_gbs
                  << std::setw(18) << std::fixed << std::setprecision(1) << hw_bw_gbs
                  << std::endl;
    }

    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Comp    = GPU compute only (no PCIe)" << std::endl;
    std::cout << "E2E     = H->D query upload + compute + D->H result download" << std::endl;
    std::cout << "Eff Tput = (DB_uint16_size * batch) / E2E_time  [amortized]" << std::endl;
    std::cout << "HW BW   = (2*DB_i32 + 2*queries_i32 + 2*output_i64) / comp_time" << std::endl;
}

// ---------- Tensor Core: 4x int8 GEMM via cuBLAS (DoublePIR Step 1) ----------
//
// Uses cublasGemmEx with CUDA_R_8I for reliable tensor core int8 dispatch.
// Decomposes uint32 query into 4 byte slices, runs 4 int8×int8→int32 GEMMs
// with alpha/beta folding:
//   GEMM 0: D =        1 * DB·q0 + 0·D
//   GEMM 1: D =      256 * DB·q1 + 1·D
//   GEMM 2: D =    65536 * DB·q2 + 1·D
//   GEMM 3: D = 16777216 * DB·q3 + 1·D

void run_db8_q32_tensor(uint64_t M, uint64_t K, const std::vector<int>& batch_sizes,
                        int warmup, int iters) {
    const double GiB = 1024.0 * 1024.0 * 1024.0;
    std::cout << "\n=== cuBLAS Tensor Core: 4x int8 GEMM (DoublePIR Step 1) ===" << std::endl;
    std::cout << "DB: " << M << " x " << K << " int8 ("
              << std::fixed << std::setprecision(1)
              << (double(M) * K) / GiB << " GiB)" << std::endl;
    std::cout << "4 GEMMs with alpha/beta folding: int8 x int8 -> int32 (tensor cores via cuBLAS)" << std::endl;

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU memory: " << std::fixed << std::setprecision(1)
              << total_mem / GiB << " GiB total, "
              << free_mem / GiB << " GiB free" << std::endl;

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "HBM BW (theoretical): ~"
              << (2.0 * props.memoryClockRate * (props.memoryBusWidth / 8.0) / 1.0e6)
              << " GB/s" << std::endl;

    if (props.major < 7 || (props.major == 7 && props.minor < 5)) {
        std::cout << "SKIPPED: Tensor cores require SM75+ (Turing or later)" << std::endl;
        return;
    }

    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::setw(8) << "Batch"
              << std::setw(12) << "Comp (ms)"
              << std::setw(12) << "E2E (ms)"
              << std::setw(12) << "E2E QPS"
              << std::setw(18) << "Eff Tput (GB/s)"
              << std::setw(18) << "HW BW (GB/s)"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // Allocate DB (int8, row-major M×K = transposed col-major K×M for cuBLAS)
    int8_t* d_db;
    CUDA_CHECK(cudaMalloc(&d_db, M * K));
    {
        std::vector<int8_t> h_db(M * K);
        srand(42);
        for (size_t i = 0; i < h_db.size(); i++)
            h_db[i] = (int8_t)(rand() & 0xFF);
        CUDA_CHECK(cudaMemcpy(d_db, h_db.data(), M * K, cudaMemcpyHostToDevice));
    }

    for (int N : batch_sizes) {
        size_t query_elems = K * N;
        size_t output_elems = M * N;

        // Allocate and fill query (int32), keep host copy for E2E
        int32_t* d_query;
        CUDA_CHECK(cudaMalloc(&d_query, query_elems * sizeof(int32_t)));
        std::vector<int32_t> h_query(query_elems);
        for (size_t i = 0; i < h_query.size(); i++)
            h_query[i] = (int32_t)rand();
        CUDA_CHECK(cudaMemcpy(d_query, h_query.data(),
                              query_elems * sizeof(int32_t),
                              cudaMemcpyHostToDevice));

        // Decompose into 4 byte slices
        int8_t* d_q[4];
        for (int i = 0; i < 4; i++)
            CUDA_CHECK(cudaMalloc(&d_q[i], query_elems));
        int decomp_threads = 256;
        int decomp_blocks = ((int)query_elems + decomp_threads - 1) / decomp_threads;
        decompose_query_bytes_kernel<<<decomp_blocks, decomp_threads>>>(
            d_q[0], d_q[1], d_q[2], d_q[3], d_query, query_elems);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Output buffer (int32, col-major M×N)
        int32_t* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, output_elems * sizeof(int32_t)));

        // Host result buffer for E2E download
        std::vector<int32_t> h_result(output_elems);

        int32_t alphas[4] = {1, 256, 65536, 16777216};
        int32_t betas[4]  = {0, 1, 1, 1};

        // Warmup
        for (int w = 0; w < warmup; w++) {
            for (int g = 0; g < 4; g++) {
                CUBLAS_CHECK(cublasGemmEx(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    (int)M, N, (int)K,
                    &alphas[g],
                    d_db,     CUDA_R_8I, (int)K,
                    d_q[g],   CUDA_R_8I, (int)K,
                    &betas[g],
                    d_output, CUDA_R_32I, (int)M,
                    CUBLAS_COMPUTE_32I,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compute-only timing (GPU events)
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            for (int g = 0; g < 4; g++)
                CUBLAS_CHECK(cublasGemmEx(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    (int)M, N, (int)K,
                    &alphas[g],
                    d_db,     CUDA_R_8I, (int)K,
                    d_q[g],   CUDA_R_8I, (int)K,
                    &betas[g],
                    d_output, CUDA_R_32I, (int)M,
                    CUBLAS_COMPUTE_32I,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        double comp_ms = elapsed_ms / iters;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // E2E timing (wall clock: H->D query + decompose + compute + D->H result)
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            CUDA_CHECK(cudaMemcpy(d_query, h_query.data(),
                                  query_elems * sizeof(int32_t), cudaMemcpyHostToDevice));
            decompose_query_bytes_kernel<<<decomp_blocks, decomp_threads>>>(
                d_q[0], d_q[1], d_q[2], d_q[3], d_query, query_elems);
            for (int g = 0; g < 4; g++)
                CUBLAS_CHECK(cublasGemmEx(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    (int)M, N, (int)K,
                    &alphas[g],
                    d_db,     CUDA_R_8I, (int)K,
                    d_q[g],   CUDA_R_8I, (int)K,
                    &betas[g],
                    d_output, CUDA_R_32I, (int)M,
                    CUBLAS_COMPUTE_32I,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            CUDA_CHECK(cudaMemcpy(h_result.data(), d_output,
                                  output_elems * sizeof(int32_t), cudaMemcpyDeviceToHost));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double e2e_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

        double db_bytes = double(M) * K;  // int8
        double e2e_qps = N / (e2e_ms / 1000.0);
        double eff_tput_gbs = (db_bytes * N) / (e2e_ms / 1000.0) / 1.0e9;

        // HW BW: DB read 4x, 4 query byte arrays, output: 1 write + 3 RMW
        double hw_bw_gbs = (
            4.0 * db_bytes +
            4.0 * double(K) * N +
            (1 + 3 * 2) * double(M) * N * sizeof(int32_t)
        ) / (comp_ms / 1000.0) / 1.0e9;

        std::cout << std::setw(8) << N
                  << std::setw(12) << std::fixed << std::setprecision(2) << comp_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << e2e_ms
                  << std::setw(12) << std::fixed << std::setprecision(0) << e2e_qps
                  << std::setw(18) << std::fixed << std::setprecision(1) << eff_tput_gbs
                  << std::setw(18) << std::fixed << std::setprecision(1) << hw_bw_gbs
                  << std::endl;

        CUDA_CHECK(cudaFree(d_query));
        for (int i = 0; i < 4; i++)
            CUDA_CHECK(cudaFree(d_q[i]));
        CUDA_CHECK(cudaFree(d_output));
    }

    CUDA_CHECK(cudaFree(d_db));
    cublasDestroy(handle);
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Comp    = GPU compute only (no PCIe)" << std::endl;
    std::cout << "E2E     = H->D query upload + compute + D->H result download" << std::endl;
    std::cout << "Eff Tput = (DB_int8_size * batch) / E2E_time  [amortized]" << std::endl;
    std::cout << "HW BW   = (4*DB + 4*q_bytes + output_writes) / comp_time" << std::endl;
}

// ---------- Tensor Core: 15x int8 GEMM via cuBLAS (uint16 DB x uint64 query) ----------
//
// Decomposes uint16 DB into 2 byte slices, uint64 query into 8 byte slices.
// Product = sum_{i in 0..1, j in 0..7} db_b[i] * q_b[j] * 256^(i+j)
// Power 8 (i=1,j=7): 256^8 = 2^64 ≡ 0 mod 2^64 → skipped.
// 15 GEMMs total, using 2 int32 accumulators:
//   Low  (powers 0-3):  7 GEMMs, alpha = 256^p
//   High (powers 4-7):  8 GEMMs, alpha = 256^(p-4)
// Final: result = (uint64)low | ((uint64)high << 32)

void run_db16_q64_tensor(uint64_t M, uint64_t K, const std::vector<int>& batch_sizes,
                         int warmup, int iters) {
    const double GiB = 1024.0 * 1024.0 * 1024.0;
    std::cout << "\n=== cuBLAS Tensor Core: 15x int8 GEMM (uint16 DB x uint64 query) ===" << std::endl;
    std::cout << "DB: " << M << " x " << K << " uint16 ("
              << std::fixed << std::setprecision(1)
              << (double(M) * K * 2) / GiB << " GiB)" << std::endl;
    std::cout << "15 GEMMs (2 byte DB x 8 byte query, skip power 8): int8 x int8 -> int32 (tensor cores)" << std::endl;

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU memory: " << std::fixed << std::setprecision(1)
              << total_mem / GiB << " GiB total, "
              << free_mem / GiB << " GiB free" << std::endl;

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "HBM BW (theoretical): ~"
              << (2.0 * props.memoryClockRate * (props.memoryBusWidth / 8.0) / 1.0e6)
              << " GB/s" << std::endl;

    if (props.major < 7 || (props.major == 7 && props.minor < 5)) {
        std::cout << "SKIPPED: Tensor cores require SM75+ (Turing or later)" << std::endl;
        return;
    }

    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::setw(8) << "Batch"
              << std::setw(12) << "Comp (ms)"
              << std::setw(12) << "E2E (ms)"
              << std::setw(12) << "E2E QPS"
              << std::setw(18) << "Eff Tput (GB/s)"
              << std::setw(18) << "HW BW (GB/s)"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // Allocate and decompose DB (uint16, row-major M×K) into 2 int8 byte slices
    size_t db_elems = M * K;
    int8_t* d_db_b[2];
    {
        uint16_t* d_db_raw;
        CUDA_CHECK(cudaMalloc(&d_db_raw, db_elems * sizeof(uint16_t)));
        std::vector<uint16_t> h_db(db_elems);
        srand(42);
        for (size_t i = 0; i < h_db.size(); i++)
            h_db[i] = (uint16_t)(rand() & 0xFFFF);
        CUDA_CHECK(cudaMemcpy(d_db_raw, h_db.data(), db_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));

        for (int i = 0; i < 2; i++)
            CUDA_CHECK(cudaMalloc(&d_db_b[i], db_elems));

        int threads = 256;
        int blocks = ((int)db_elems + threads - 1) / threads;
        decompose_i16_bytes_kernel<<<blocks, threads>>>(d_db_b[0], d_db_b[1], d_db_raw, db_elems);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_db_raw));
    }

    // GEMM schedule: (db_byte, q_byte, alpha, beta, output_idx)
    // Low accumulator (powers 0-3): 7 GEMMs
    // High accumulator (powers 4-7): 8 GEMMs
    struct GemmSpec { int db_b; int q_b; int32_t alpha; int32_t beta; int out; };
    GemmSpec specs[15] = {
        // Low accumulator
        {0, 0, 1,        0, 0},   // power 0
        {0, 1, 256,      1, 0},   // power 1
        {1, 0, 256,      1, 0},   // power 1
        {0, 2, 65536,    1, 0},   // power 2
        {1, 1, 65536,    1, 0},   // power 2
        {0, 3, 16777216, 1, 0},   // power 3
        {1, 2, 16777216, 1, 0},   // power 3
        // High accumulator
        {0, 4, 1,        0, 1},   // power 4
        {1, 3, 1,        1, 1},   // power 4
        {0, 5, 256,      1, 1},   // power 5
        {1, 4, 256,      1, 1},   // power 5
        {0, 6, 65536,    1, 1},   // power 6
        {1, 5, 65536,    1, 1},   // power 6
        {0, 7, 16777216, 1, 1},   // power 7
        {1, 6, 16777216, 1, 1},   // power 7
    };

    for (int N : batch_sizes) {
        size_t query_elems = K * N;
        size_t output_elems = M * N;

        // Allocate query buffers (keep host + raw device for E2E timing)
        uint64_t* d_query_raw;
        CUDA_CHECK(cudaMalloc(&d_query_raw, query_elems * sizeof(uint64_t)));
        std::vector<uint64_t> h_query(query_elems);
        for (size_t i = 0; i < h_query.size(); i++)
            h_query[i] = ((uint64_t)rand() << 32) | (uint64_t)rand();
        CUDA_CHECK(cudaMemcpy(d_query_raw, h_query.data(),
                              query_elems * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));

        // Decompose into 8 int8 byte slices
        int8_t* d_q_b[8];
        for (int i = 0; i < 8; i++)
            CUDA_CHECK(cudaMalloc(&d_q_b[i], query_elems));

        int decomp_threads = 256;
        int decomp_blocks = ((int)query_elems + decomp_threads - 1) / decomp_threads;
        decompose_i64_bytes_kernel<<<decomp_blocks, decomp_threads>>>(
            d_q_b[0], d_q_b[1], d_q_b[2], d_q_b[3],
            d_q_b[4], d_q_b[5], d_q_b[6], d_q_b[7],
            d_query_raw, query_elems);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Two output accumulators (int32, col-major M×N)
        int32_t* d_output[2];
        CUDA_CHECK(cudaMalloc(&d_output[0], output_elems * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_output[1], output_elems * sizeof(int32_t)));

        // Host result buffers for E2E download
        std::vector<int32_t> h_result_lo(output_elems), h_result_hi(output_elems);

        // Warmup
        for (int w = 0; w < warmup; w++) {
            for (int g = 0; g < 15; g++) {
                CUBLAS_CHECK(cublasGemmEx(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    (int)M, N, (int)K,
                    &specs[g].alpha,
                    d_db_b[specs[g].db_b], CUDA_R_8I, (int)K,
                    d_q_b[specs[g].q_b],   CUDA_R_8I, (int)K,
                    &specs[g].beta,
                    d_output[specs[g].out], CUDA_R_32I, (int)M,
                    CUBLAS_COMPUTE_32I,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compute-only timing (GPU events)
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            for (int g = 0; g < 15; g++)
                CUBLAS_CHECK(cublasGemmEx(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    (int)M, N, (int)K,
                    &specs[g].alpha,
                    d_db_b[specs[g].db_b], CUDA_R_8I, (int)K,
                    d_q_b[specs[g].q_b],   CUDA_R_8I, (int)K,
                    &specs[g].beta,
                    d_output[specs[g].out], CUDA_R_32I, (int)M,
                    CUBLAS_COMPUTE_32I,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        double comp_ms = elapsed_ms / iters;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // E2E timing (wall clock: H->D query + decompose + compute + D->H result)
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            CUDA_CHECK(cudaMemcpy(d_query_raw, h_query.data(),
                                  query_elems * sizeof(uint64_t), cudaMemcpyHostToDevice));
            decompose_i64_bytes_kernel<<<decomp_blocks, decomp_threads>>>(
                d_q_b[0], d_q_b[1], d_q_b[2], d_q_b[3],
                d_q_b[4], d_q_b[5], d_q_b[6], d_q_b[7],
                d_query_raw, query_elems);
            for (int g = 0; g < 15; g++)
                CUBLAS_CHECK(cublasGemmEx(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    (int)M, N, (int)K,
                    &specs[g].alpha,
                    d_db_b[specs[g].db_b], CUDA_R_8I, (int)K,
                    d_q_b[specs[g].q_b],   CUDA_R_8I, (int)K,
                    &specs[g].beta,
                    d_output[specs[g].out], CUDA_R_32I, (int)M,
                    CUBLAS_COMPUTE_32I,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            CUDA_CHECK(cudaMemcpy(h_result_lo.data(), d_output[0],
                                  output_elems * sizeof(int32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_result_hi.data(), d_output[1],
                                  output_elems * sizeof(int32_t), cudaMemcpyDeviceToHost));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double e2e_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

        // Effective throughput: DB_uint16_size * batch / E2E time
        double db_bytes = double(M) * K * sizeof(uint16_t);
        double e2e_qps = N / (e2e_ms / 1000.0);
        double eff_tput_gbs = (db_bytes * N) / (e2e_ms / 1000.0) / 1.0e9;

        // HW BW: based on compute time (GPU utilization metric)
        double hw_bw_gbs = (
            15.0 * double(M) * K +                              // DB byte slice reads
            15.0 * double(K) * N +                              // query byte slice reads
            (1 + 6 * 2) * double(M) * N * sizeof(int32_t) +     // low accum (1 write + 6 RMW)
            (1 + 7 * 2) * double(M) * N * sizeof(int32_t)       // high accum (1 write + 7 RMW)
        ) / (comp_ms / 1000.0) / 1.0e9;

        std::cout << std::setw(8) << N
                  << std::setw(12) << std::fixed << std::setprecision(2) << comp_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << e2e_ms
                  << std::setw(12) << std::fixed << std::setprecision(0) << e2e_qps
                  << std::setw(18) << std::fixed << std::setprecision(1) << eff_tput_gbs
                  << std::setw(18) << std::fixed << std::setprecision(1) << hw_bw_gbs
                  << std::endl;

        CUDA_CHECK(cudaFree(d_query_raw));
        for (int i = 0; i < 8; i++)
            CUDA_CHECK(cudaFree(d_q_b[i]));
        CUDA_CHECK(cudaFree(d_output[0]));
        CUDA_CHECK(cudaFree(d_output[1]));
    }

    for (int i = 0; i < 2; i++)
        CUDA_CHECK(cudaFree(d_db_b[i]));
    cublasDestroy(handle);
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Comp    = GPU compute only (no PCIe)" << std::endl;
    std::cout << "E2E     = H->D query upload + compute + D->H result download" << std::endl;
    std::cout << "Eff Tput = (DB_uint16_size * batch) / E2E_time  [amortized]" << std::endl;
    std::cout << "HW BW   = (15*db_slices + 15*q_slices + output_accesses) / comp_time" << std::endl;
}

// ---------- CUTLASS uint8 Tensor Core: 1x wide GEMM (uint8 DB x uint32 query) ----------
//
// DB is already uint8 (1 byte), query decomposes into 4 bytes.
// Pack 4 query byte slices into K × (4*N), do 1 GEMM.
// DB read exactly once.

// SM80+ (Ampere): wider instruction shape
using CutlassGemmU8TC_Sm80 = cutlass::gemm::device::Gemm<
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

// SM75 (Turing): u8 IMMA with 8×8×16 instruction shape
using CutlassGemmU8TC_Sm75 = cutlass::gemm::device::Gemm<
    uint8_t, cutlass::layout::RowMajor,
    uint8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 2  // SM75: 2 stages (no async copy)
>;

static bool detect_sm80() {
    int device;
    cudaGetDevice(&device);
    int major;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    return major >= 8;
}

// Helper: run the correct TC GEMM based on SM version
static cutlass::Status run_u8tc_gemm(bool is_sm80,
    int M, int wide_N, int K,
    const uint8_t* A, int lda,
    const uint8_t* B, int ldb,
    int32_t* C, int ldc,
    void* workspace = nullptr)
{
    int32_t alpha = 1, beta = 0;
    cutlass::gemm::GemmCoord problem_size(M, wide_N, K);
    if (is_sm80) {
        CutlassGemmU8TC_Sm80::Arguments args{
            problem_size, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta}, 1
        };
        CutlassGemmU8TC_Sm80 gemm_op;
        auto s = gemm_op.initialize(args, workspace);
        if (s != cutlass::Status::kSuccess) return s;
        return gemm_op();
    } else {
        CutlassGemmU8TC_Sm75::Arguments args{
            problem_size, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta}, 1
        };
        CutlassGemmU8TC_Sm75 gemm_op;
        auto s = gemm_op.initialize(args, workspace);
        if (s != cutlass::Status::kSuccess) return s;
        return gemm_op();
    }
}

void run_db8_q32_tc_cutlass(uint64_t M, uint64_t K, const std::vector<int>& batch_sizes,
                             int warmup, int iters) {
    const double GiB = 1024.0 * 1024.0 * 1024.0;
    std::cout << "\n=== CUTLASS uint8 TC: 1x wide GEMM (uint8 DB x uint32 query) ===" << std::endl;
    std::cout << "DB: " << M << " x " << K << " uint8 ("
              << std::fixed << std::setprecision(1)
              << (double(M) * K) / GiB << " GiB)" << std::endl;
    std::cout << "1 GEMM: uint8 × uint8 → int32, M×K × K×(4*batch)" << std::endl;
    std::cout << "DB read exactly 1x" << std::endl;

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU memory: " << std::fixed << std::setprecision(1)
              << total_mem / GiB << " GiB total, "
              << free_mem / GiB << " GiB free" << std::endl;

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "HBM BW (theoretical): ~"
              << (2.0 * props.memoryClockRate * (props.memoryBusWidth / 8.0) / 1.0e6)
              << " GB/s" << std::endl;

    if (props.major < 7 || (props.major == 7 && props.minor < 5)) {
        std::cout << "SKIPPED: CUTLASS uint8 tensor cores require SM75+ (Turing or later)" << std::endl;
        return;
    }

    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::setw(8) << "Batch"
              << std::setw(12) << "Comp (ms)"
              << std::setw(12) << "E2E (ms)"
              << std::setw(12) << "E2E QPS"
              << std::setw(18) << "Eff Tput (GB/s)"
              << std::setw(18) << "HW BW (GB/s)"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Allocate DB (already uint8, row-major M×K)
    uint8_t* d_db;
    CUDA_CHECK(cudaMalloc(&d_db, M * K));
    {
        std::vector<uint8_t> h_db(M * K);
        srand(42);
        for (size_t i = 0; i < h_db.size(); i++)
            h_db[i] = (uint8_t)(rand() & 0xFF);
        CUDA_CHECK(cudaMemcpy(d_db, h_db.data(), M * K, cudaMemcpyHostToDevice));
    }

    for (int N : batch_sizes) {
        size_t query_elems = K * (size_t)N;
        size_t output_elems = M * (size_t)N;
        size_t wide_N = 4 * (size_t)N;  // packed query width

        // Allocate query buffers (keep host + raw device for E2E timing).
        // Host buffers are PINNED via cudaHostAlloc so that the
        // cudaMemcpy in the E2E loop is a true async DMA — no
        // pageable→staging copy on the CPU thread. Without pinning,
        // each H2D/D2H call eats ~50–100 µs of staging overhead
        // which artificially inflates E2E at small batches.
        uint8_t* d_query_packed;
        CUDA_CHECK(cudaMalloc(&d_query_packed, K * wide_N));
        uint32_t* d_query_raw;
        CUDA_CHECK(cudaMalloc(&d_query_raw, query_elems * sizeof(uint32_t)));
        uint32_t* h_query;
        CUDA_CHECK(cudaHostAlloc(&h_query, query_elems * sizeof(uint32_t),
                                 cudaHostAllocDefault));
        for (size_t i = 0; i < query_elems; i++)
            h_query[i] = (uint32_t)rand();
        CUDA_CHECK(cudaMemcpy(d_query_raw, h_query,
                              query_elems * sizeof(uint32_t), cudaMemcpyHostToDevice));

        int decomp_threads = 256;
        int decomp_blocks = ((int)query_elems + decomp_threads - 1) / decomp_threads;
        decompose_u32_bytes_packed_kernel<<<decomp_blocks, decomp_threads>>>(
            d_query_packed, d_query_raw, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Host result buffer for E2E download — also pinned.
        uint32_t* h_result;
        CUDA_CHECK(cudaHostAlloc(&h_result, output_elems * sizeof(uint32_t),
                                 cudaHostAllocDefault));

        // GEMM output: M × (4*N) int32
        int32_t* d_gemm_out;
        CUDA_CHECK(cudaMalloc(&d_gemm_out, M * wide_N * sizeof(int32_t)));

        // Accumulator: M × N uint32
        uint32_t* d_accum;
        CUDA_CHECK(cudaMalloc(&d_accum, output_elems * sizeof(uint32_t)));

        bool is_sm80 = detect_sm80();

        int acc_threads = 256;
        int acc_blocks = ((int)output_elems + acc_threads - 1) / acc_threads;

        // Warmup
        for (int w = 0; w < warmup; w++) {
            auto s = run_u8tc_gemm(is_sm80, M, (int)wide_N, K,
                d_db, (int)K, d_query_packed, (int)K, d_gemm_out, (int)M);
            if (s != cutlass::Status::kSuccess) {
                std::cerr << "CUTLASS TC GEMM failed: " << cutlassGetStatusString(s) << std::endl;
                break;
            }
            accumulate_db8_q32_kernel<<<acc_blocks, acc_threads>>>(d_accum, d_gemm_out, M, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compute-only timing (GPU events)
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            run_u8tc_gemm(is_sm80, M, (int)wide_N, K,
                d_db, (int)K, d_query_packed, (int)K, d_gemm_out, (int)M);
            accumulate_db8_q32_kernel<<<acc_blocks, acc_threads>>>(d_accum, d_gemm_out, M, N);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        double comp_ms = elapsed_ms / iters;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // E2E timing (wall clock: H->D query + compute + D->H result)
        // h_query and h_result are both pinned, so cudaMemcpy is true
        // async DMA with no driver-side staging copy.
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            CUDA_CHECK(cudaMemcpy(d_query_raw, h_query,
                                  query_elems * sizeof(uint32_t), cudaMemcpyHostToDevice));
            decompose_u32_bytes_packed_kernel<<<decomp_blocks, decomp_threads>>>(
                d_query_packed, d_query_raw, K, N);
            run_u8tc_gemm(is_sm80, M, (int)wide_N, K,
                d_db, (int)K, d_query_packed, (int)K, d_gemm_out, (int)M);
            accumulate_db8_q32_kernel<<<acc_blocks, acc_threads>>>(d_accum, d_gemm_out, M, N);
            CUDA_CHECK(cudaMemcpy(h_result, d_accum,
                                  output_elems * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double e2e_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

        double db_bytes = double(M) * K;  // uint8
        double e2e_qps = N / (e2e_ms / 1000.0);
        double eff_tput_gbs = (db_bytes * N) / (e2e_ms / 1000.0) / 1.0e9;

        // HW BW: based on compute time (GPU utilization metric)
        double hw_bw_gbs = (
            double(M) * K +                              // DB read
            double(K) * wide_N +                         // packed query read
            double(M) * wide_N * sizeof(int32_t) +       // GEMM output write
            double(M) * N * sizeof(uint32_t)              // accum write
        ) / (comp_ms / 1000.0) / 1.0e9;

        std::cout << std::setw(8) << N
                  << std::setw(12) << std::fixed << std::setprecision(2) << comp_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << e2e_ms
                  << std::setw(12) << std::fixed << std::setprecision(0) << e2e_qps
                  << std::setw(18) << std::fixed << std::setprecision(1) << eff_tput_gbs
                  << std::setw(18) << std::fixed << std::setprecision(1) << hw_bw_gbs
                  << std::endl;

        CUDA_CHECK(cudaFree(d_query_raw));
        CUDA_CHECK(cudaFree(d_query_packed));
        CUDA_CHECK(cudaFree(d_gemm_out));
        CUDA_CHECK(cudaFree(d_accum));
        CUDA_CHECK(cudaFreeHost(h_query));
        CUDA_CHECK(cudaFreeHost(h_result));
    }

    CUDA_CHECK(cudaFree(d_db));
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Comp    = GPU compute only (no PCIe)" << std::endl;
    std::cout << "E2E     = H->D query upload + compute + D->H result download (pinned)" << std::endl;
    std::cout << "Eff Tput = (DB_uint8_size * batch) / E2E_time  [amortized]" << std::endl;
    std::cout << "HW BW   = (db + packed_query + gemm_out + accum) / comp_time" << std::endl;
}

// ---------- CUTLASS uint8 Tensor Core: 2x wide GEMM (uint16 DB x uint64 query) ----------
//
// The new approach: decompose DB into 2 uint8 byte slices, pack query 8 byte
// slices contiguously into K × (8*N) matrix, then do 2 GEMMs (one per DB byte).
// Each GEMM: M×K (uint8) × K×(8*N) (uint8) → M×(8*N) (int32)
// DB is read exactly 2 times (vs 15 in the old approach).
// Reuses CutlassGemmU8TC_Sm80/Sm75 + run_u8tc_gemm helper defined above.

void run_db16_q64_tc_cutlass(uint64_t M, uint64_t K, const std::vector<int>& batch_sizes,
                              int warmup, int iters) {
    const double GiB = 1024.0 * 1024.0 * 1024.0;
    std::cout << "\n=== CUTLASS uint8 TC: 1x stacked GEMM (uint16 DB x uint64 query) ===" << std::endl;
    std::cout << "DB: " << M << " x " << K << " uint16 ("
              << std::fixed << std::setprecision(1)
              << (double(M) * K * 2) / GiB << " GiB)" << std::endl;
    std::cout << "1 GEMM: (2M)×K × K×(8*batch), DB bytes stacked vertically" << std::endl;
    std::cout << "DB + query each read exactly 1x" << std::endl;

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU memory: " << std::fixed << std::setprecision(1)
              << total_mem / GiB << " GiB total, "
              << free_mem / GiB << " GiB free" << std::endl;

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "HBM BW (theoretical): ~"
              << (2.0 * props.memoryClockRate * (props.memoryBusWidth / 8.0) / 1.0e6)
              << " GB/s" << std::endl;

    if (props.major < 7 || (props.major == 7 && props.minor < 5)) {
        std::cout << "SKIPPED: CUTLASS uint8 tensor cores require SM75+ (Turing or later)" << std::endl;
        return;
    }

    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::setw(8) << "Batch"
              << std::setw(12) << "Comp (ms)"
              << std::setw(12) << "E2E (ms)"
              << std::setw(12) << "E2E QPS"
              << std::setw(18) << "Eff Tput (GB/s)"
              << std::setw(18) << "HW BW (GB/s)"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Decompose DB (uint16) into stacked (2M)×K uint8: [low_bytes; high_bytes]
    size_t db_elems = M * K;
    uint8_t* d_db_stacked;
    CUDA_CHECK(cudaMalloc(&d_db_stacked, 2 * db_elems));
    {
        uint16_t* d_db_raw;
        CUDA_CHECK(cudaMalloc(&d_db_raw, db_elems * sizeof(uint16_t)));
        std::vector<uint16_t> h_db(db_elems);
        srand(42);
        for (size_t i = 0; i < h_db.size(); i++)
            h_db[i] = (uint16_t)(rand() & 0xFFFF);
        CUDA_CHECK(cudaMemcpy(d_db_raw, h_db.data(), db_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks = ((int)db_elems + threads - 1) / threads;
        decompose_u16_stacked_kernel<<<blocks, threads>>>(d_db_stacked, d_db_raw, db_elems);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_db_raw));
    }

    for (int N : batch_sizes) {
        size_t query_elems = K * (size_t)N;
        size_t output_elems = M * (size_t)N;
        size_t wide_N = 8 * (size_t)N;  // packed query width
        size_t tall_M = 2 * M;          // stacked DB height

        // Allocate query buffers (keep host + raw device for E2E timing)
        uint8_t* d_query_packed;
        CUDA_CHECK(cudaMalloc(&d_query_packed, K * wide_N));
        uint64_t* d_query_raw;
        CUDA_CHECK(cudaMalloc(&d_query_raw, query_elems * sizeof(uint64_t)));
        std::vector<uint64_t> h_query(query_elems);
        for (size_t i = 0; i < h_query.size(); i++)
            h_query[i] = ((uint64_t)rand() << 32) | (uint64_t)rand();
        CUDA_CHECK(cudaMemcpy(d_query_raw, h_query.data(),
                              query_elems * sizeof(uint64_t), cudaMemcpyHostToDevice));

        int decomp_threads = 256;
        int decomp_blocks = ((int)query_elems + decomp_threads - 1) / decomp_threads;
        decompose_u64_bytes_packed_kernel<<<decomp_blocks, decomp_threads>>>(
            d_query_packed, d_query_raw, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Host result buffer for E2E download
        std::vector<uint64_t> h_result(output_elems);

        // GEMM output: (2M) × (8*N) int32
        int32_t* d_gemm_out;
        CUDA_CHECK(cudaMalloc(&d_gemm_out, tall_M * wide_N * sizeof(int32_t)));

        // Accumulator: M × N uint64
        uint64_t* d_accum;
        CUDA_CHECK(cudaMalloc(&d_accum, output_elems * sizeof(uint64_t)));

        bool is_sm80 = detect_sm80();

        int acc_threads = 256;
        int acc_blocks = ((int)output_elems + acc_threads - 1) / acc_threads;

        // Warmup
        for (int w = 0; w < warmup; w++) {
            run_u8tc_gemm(is_sm80, (int)tall_M, (int)wide_N, K,
                d_db_stacked, (int)K, d_query_packed, (int)K, d_gemm_out, (int)tall_M);
            accumulate_db16_q64_stacked_kernel<<<acc_blocks, acc_threads>>>(
                d_accum, d_gemm_out, M, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compute-only timing (GPU events)
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            run_u8tc_gemm(is_sm80, (int)tall_M, (int)wide_N, K,
                d_db_stacked, (int)K, d_query_packed, (int)K, d_gemm_out, (int)tall_M);
            accumulate_db16_q64_stacked_kernel<<<acc_blocks, acc_threads>>>(
                d_accum, d_gemm_out, M, N);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        double comp_ms = elapsed_ms / iters;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // E2E timing (wall clock: H->D query + compute + D->H result)
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            CUDA_CHECK(cudaMemcpy(d_query_raw, h_query.data(),
                                  query_elems * sizeof(uint64_t), cudaMemcpyHostToDevice));
            decompose_u64_bytes_packed_kernel<<<decomp_blocks, decomp_threads>>>(
                d_query_packed, d_query_raw, K, N);
            run_u8tc_gemm(is_sm80, (int)tall_M, (int)wide_N, K,
                d_db_stacked, (int)K, d_query_packed, (int)K, d_gemm_out, (int)tall_M);
            accumulate_db16_q64_stacked_kernel<<<acc_blocks, acc_threads>>>(
                d_accum, d_gemm_out, M, N);
            CUDA_CHECK(cudaMemcpy(h_result.data(), d_accum,
                                  output_elems * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double e2e_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

        // Effective throughput: DB_uint16_size * batch / E2E time
        double db_bytes = double(M) * K * sizeof(uint16_t);
        double e2e_qps = N / (e2e_ms / 1000.0);
        double eff_tput_gbs = (db_bytes * N) / (e2e_ms / 1000.0) / 1.0e9;

        // HW BW: 1 stacked DB read (2M*K) + 1 packed query read (K*8N)
        //       + 1 GEMM output write (2M*8N*4) + accum write (M*N*8)
        double hw_bw_gbs = (
            2.0 * double(M) * K +                          // stacked DB (2M*K bytes)
            double(K) * wide_N +                           // packed query read 1x
            double(tall_M) * wide_N * sizeof(int32_t) +    // GEMM output write
            double(M) * N * sizeof(uint64_t)                // accum write
        ) / (comp_ms / 1000.0) / 1.0e9;

        std::cout << std::setw(8) << N
                  << std::setw(12) << std::fixed << std::setprecision(2) << comp_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << e2e_ms
                  << std::setw(12) << std::fixed << std::setprecision(0) << e2e_qps
                  << std::setw(18) << std::fixed << std::setprecision(1) << eff_tput_gbs
                  << std::setw(18) << std::fixed << std::setprecision(1) << hw_bw_gbs
                  << std::endl;

        CUDA_CHECK(cudaFree(d_query_raw));
        CUDA_CHECK(cudaFree(d_query_packed));
        CUDA_CHECK(cudaFree(d_gemm_out));
        CUDA_CHECK(cudaFree(d_accum));
    }

    CUDA_CHECK(cudaFree(d_db_stacked));
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Comp    = GPU compute only (no PCIe)" << std::endl;
    std::cout << "E2E     = H->D query upload + compute + D->H result download" << std::endl;
    std::cout << "Eff Tput = (DB_uint16_size * batch) / E2E_time  [amortized]" << std::endl;
    std::cout << "HW BW   = (stacked_db + packed_query + gemm_out + accum) / comp_time" << std::endl;
}

// ---------- Argument parsing ----------

std::vector<int> parse_batch_list(const std::string& s) {
    std::vector<int> result;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        result.push_back(std::stoi(token));
    }
    return result;
}

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]" << std::endl;
    std::cerr << "  --mode    MODE           One of: db32_q32, db32_q64, db16_q32, db16_q64, db16_crt," << std::endl;
    std::cerr << "                           db8_q32, db32_crt_i64, db8_q32_tc, db16_q64_tc," << std::endl;
    std::cerr << "                           db8_q32_tc_cutlass, db16_q64_tc_cutlass, all" << std::endl;
    std::cerr << "                           (default: all)" << std::endl;
    std::cerr << "  --batches 1,2,4,...      Comma-separated batch sizes" << std::endl;
    std::cerr << "  --log_dim N              Log2 of matrix dimension (default: 15)" << std::endl;
    std::cerr << "  --warmup  N              Warmup iterations (default: 3)" << std::endl;
    std::cerr << "  --iters   N              Timed iterations (default: 10)" << std::endl;
}

int main(int argc, char** argv) {
    std::string mode = "thesis";  // default: SIMT + TC benchmarks for thesis comparison
    int log_dim = 15;
    int warmup = 3;
    int iters = 10;
    std::vector<int> batches = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--batches" && i + 1 < argc) {
            batches = parse_batch_list(argv[++i]);
        } else if (arg == "--log_dim" && i + 1 < argc) {
            log_dim = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    uint64_t dim = 1ULL << log_dim;
    bool any = false;

    if (mode == "db32_q32" || mode == "all") {
        PirBench<uint32_t, uint32_t>::run(dim, dim, batches, warmup, iters,
            "uint32 DB x uint32 query");
        any = true;
    }
    if (mode == "db32_q64" || mode == "all") {
        PirBench<uint32_t, uint64_t>::run(dim, dim, batches, warmup, iters,
            "uint32 DB x uint64 query");
        any = true;
    }
    if (mode == "db16_q32" || mode == "all") {
        PirBench<uint16_t, uint32_t>::run(dim, dim, batches, warmup, iters,
            "uint16 DB x uint32 query");
        any = true;
    }
    if (mode == "db16_q64" || mode == "all" || mode == "thesis") {
        PirBench<uint16_t, uint64_t>::run(dim, dim, batches, warmup, iters,
            "uint16 DB x uint64 query");
        any = true;
    }
    if (mode == "db16_crt" || mode == "all") {
        run_crt(dim, dim, batches, warmup, iters);
        any = true;
    }
    if (mode == "db8_q32" || mode == "all" || mode == "thesis") {
        run_db8_q32(dim, dim, batches, warmup, iters);
        any = true;
    }
    if (mode == "db32_crt_i64" || mode == "all") {
        run_crt_i64(dim, dim, batches, warmup, iters);
        any = true;
    }
    if (mode == "db8_q32_tc" || mode == "all") {
        run_db8_q32_tensor(dim, dim, batches, warmup, iters);
        any = true;
    }
    if (mode == "db16_q64_tc" || mode == "all") {
        run_db16_q64_tensor(dim, dim, batches, warmup, iters);
        any = true;
    }
    if (mode == "db8_q32_tc_cutlass" || mode == "all" || mode == "thesis") {
        run_db8_q32_tc_cutlass(dim, dim, batches, warmup, iters);
        any = true;
    }
    if (mode == "db16_q64_tc_cutlass" || mode == "all" || mode == "thesis") {
        run_db16_q64_tc_cutlass(dim, dim, batches, warmup, iters);
        any = true;
    }

    if (!any) {
        std::cerr << "Unknown --mode: " << mode << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
