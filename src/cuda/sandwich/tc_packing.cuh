/**
 * SandwichPIR Tensor Core Packing — Header
 *
 * Single-modulus (crt_count=1) InspiRING packing via M-reformulation GEMM.
 * Byte-decomposed u8×u8→int32 CUTLASS GEMM with 4 byte-slices.
 *
 * The caller controls stream sequencing — all functions take a cudaStream_t.
 */

#pragma once

#include <cstdint>
#include <cstddef>

struct NTTParams;  // forward decl from common/ntt.cuh

struct SwTcPackingContext {
    // Byte-decomposed M (precomputed offline, 4 byte slices)
    // d_M_bytes[b] each [K_gemm × N_gemm] uint8, ColumnMajor
    uint8_t* d_M_bytes[4];

    // T̂ for final key-switch term (single modulus)
    uint64_t* d_bold_t_hat;      // [num_outputs × t × poly_len]

    // â for post-process (coefficient domain)
    uint64_t* d_a_hat;           // [num_outputs × poly_len]

    // Online scratch (allocated for max_batch_size)
    uint8_t* d_A_bytes_buf;      // [4 × max_B × K_gemm] u8
    int32_t* d_G;                // [4*max_B × 4*N_gemm] int32
    uint64_t* d_result;          // [max_B × N_gemm] u64
    uint64_t* d_scratch;         // [max_B × num_outputs × 4 × poly_len] u64

    // Dimensions
    size_t poly_len;
    size_t t_exp_left;
    size_t num_iter;             // d/2 - 1
    size_t num_outputs;          // ρ
    size_t max_batch_size;
    size_t K_gemm;               // t × d
    size_t N_gemm;               // ρ × d

    int gpu_tier;                // 0=SIMT, 1=SM75 TC, 2=SM80+ TC

    // Response encoding params
    size_t response_bytes_per_output;   // full wire format (mask + body), for reference only
    size_t body_bytes_per_output;       // = (poly_len * q_2_bits + 7) / 8
    size_t mask_bytes_per_output;       // = (poly_len * q_1_bits + 7) / 8
    uint64_t rlwe_q_prime_1;     // body modulus (2^10)
    uint64_t rlwe_q_prime_2;     // mask modulus (2^18)

    // Single modulus
    uint64_t mod0;
    uint64_t barrett_cr0;

    NTTParams ntt_params;

    // Verbose sub-phase events (only valid after sw_tc_packing_gemms, read by caller)
    cudaEvent_t ev_decomp, ev_gemm, ev_accum, ev_finalize;
    bool has_events;  // true if events were recorded this batch
};

#ifdef __cplusplus
extern "C" {
#endif

void* sw_tc_packing_init(
    uint64_t* d_bold_t,
    uint64_t* d_bold_t_bar,
    const uint64_t* d_bold_t_hat,
    const uint64_t* d_a_hat,
    const uint32_t* d_tables,
    const uint32_t* d_gen_pows,
    size_t num_iter, size_t t_exp_left, size_t poly_len,
    size_t num_outputs, size_t max_batch_size,
    uint64_t rlwe_q_prime_1, uint64_t rlwe_q_prime_2,
    size_t response_bytes_per_output,
    NTTParams ntt_params, int gpu_tier);

// Packing GEMM + accumulate + finalize (writes d_scratch)
void sw_tc_packing_gemms(
    void* context, cudaStream_t stream,
    const uint64_t* d_y_body,
    const uint64_t* d_z_body,
    size_t batch_size);

// Post-process: INTT + add body values + modswitch + bitpack body-only
// bytes into `d_response_body_out` at layout
//     [batch_size][num_outputs][body_bytes_per_output].
// Mask bytes are held separately in a host-side template; this kernel
// does not produce them.
void sw_tc_packing_post_process(
    void* context, cudaStream_t stream,
    const uint64_t* d_intermediate,
    uint8_t* d_response_body_out,
    size_t batch_size);

// One-time: fill a compact mask template buffer from d_a_hat.
// Output layout: [num_outputs][mask_bytes_per_output]. Call ONCE at init
// time after the tc_ctx is built; the resulting bytes are the same for
// every query and every client, so they are downloaded to host and
// shared across all subsequent compute_batch calls.
void sw_tc_packing_fill_mask_template(
    void* context, cudaStream_t stream,
    uint8_t* d_mask_template_out);

void sw_tc_packing_free(void* context);

#ifdef __cplusplus
}
#endif
