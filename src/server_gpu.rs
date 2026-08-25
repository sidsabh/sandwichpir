//! GPU-accelerated SandwichPIR server pipeline.
//!
//! Offline: GPU NTT hint (default) or GPU GEMM hint (HINT=gemm)
//!          + GPU InspiRING precomp + M-matrix build
//! Online:  GPU W=2^32 matmul + GPU TC packing (16 CUTLASS calls) + GPU post-process

#![cfg(feature = "cuda")]

use std::time::Instant;

use log::debug;
use spiral_rs::poly::*;

use crate::cuda;
use crate::measurement::Measurement;
use crate::packing::*;
use crate::scheme::*;
use crate::server::*;

/// Build the ring A-matrix in Z_{2^32} for the TC GEMM hint.
fn build_ring_a_w32(params: &spiral_rs::params::Params, db_rows: usize) -> Vec<u64> {
    let n = params.poly_len;
    let q = params.modulus;
    let mut ring_a = vec![0u64; n * db_rows];

    let query_polys = {
        let mut client = spiral_rs::client::Client::init(params);
        client.generate_secret_keys();
        let y_client = crate::client::YClient::new(&mut client, params);
        let query = y_client.generate_query_impl(
            SEED_0,
            params.db_dim_1,
            crate::packing::PackingType::CDKS,
            0,
            None,
            None,
        );
        query
            .iter()
            .map(|x| x.submatrix(0, 0, 1, 1))
            .collect::<Vec<_>>()
    };

    for (block, query_raw) in query_polys.iter().enumerate() {
        let perm = crate::util::negacyclic_perm(query_raw.get_poly(0, 0), 0, q);
        for z in 0..n {
            for k in 0..n {
                let idx = (z + n - k) % n;
                let coeff = perm[idx];
                let val_q = if z < k { q - coeff } else { coeff };
                ring_a[z * db_rows + block * n + k] =
                    ((val_q as u128 * (1u128 << 32) + (q as u128 / 2)) / q as u128) as u64
                        & 0xFFFFFFFF;
            }
        }
    }
    ring_a
}

/// Maximum db_rows for GEMM-based hint computation (limited by GPU memory layout).
const MAX_GEMM_HINT_ROWS: usize = 133_144;

/// CUDA device attribute IDs for compute capability query.
const CUDA_ATTR_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CUDA_ATTR_COMPUTE_CAPABILITY_MINOR: i32 = 76;

/// Detect GPU compute capability and return tier: 0=SIMT, 1=SM75 TC, 2=SM80+ TC
fn detect_gpu_tier() -> i32 {
    extern "C" {
        fn cudaGetDevice(device: *mut i32) -> i32;
        fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
    }
    let mut dev = 0i32;
    let mut major = 0i32;
    let mut minor = 0i32;
    unsafe {
        cudaGetDevice(&mut dev);
        cudaDeviceGetAttribute(&mut major, CUDA_ATTR_COMPUTE_CAPABILITY_MAJOR, dev);
        cudaDeviceGetAttribute(&mut minor, CUDA_ATTR_COMPUTE_CAPABILITY_MINOR, dev);
    }
    let sm = major * 10 + minor;
    if sm >= 80 {
        2
    } else if sm >= 75 {
        1
    } else {
        0
    }
}

impl<'a, T> YServer<'a, T>
where
    T: Sized + Copy + ToU64 + Default + Sync,
    *const T: ToM512,
{
    /// SandwichPIR GPU offline pipeline.
    pub fn sandwich_gpu_offline(
        &self,
        mut measurement: Option<&mut Measurement>,
        max_batch_size: usize,
    ) -> (OfflinePrecomputedValues<'_>, cuda::SandwichOnlineContext) {
        let params = self.params;
        assert!(params.crt_count == 1);

        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_cols = params.instances * params.poly_len;
        let num_rlwe_outputs = db_cols / params.poly_len;
        let gamma = params.poly_len;
        let gpu_tier = detect_gpu_tier();

        let db_u8: &[u8] = unsafe {
            std::slice::from_raw_parts(self.db().as_ptr() as *const u8, self.db().len())
        };

        // ── CPU setup ──
        let use_gemm_hint = std::env::var("HINT").map_or(false, |v| v == "gemm");
        let setup_start = Instant::now();

        let packing_params = PackParams::new_for_gpu(params, gamma);
        let t_pp = setup_start.elapsed().as_millis();
        let offline_keys = OfflinePackingKeys::init_masks_only(params, W_SEED, V_SEED);
        let t_keys = setup_start.elapsed().as_millis();

        let pp = &packing_params;
        let w_mask_flat: Vec<u64> = offline_keys.w_mask.as_ref().unwrap().as_slice().to_vec();
        let v_mask_flat: Vec<u64> = offline_keys.v_mask.as_ref().unwrap().as_slice().to_vec();
        let mod_inv_poly_flat: Vec<u64> = pp.mod_inv_poly.as_slice().to_vec();
        let tables_flat: Vec<u32> = pp
            .tables
            .iter()
            .flat_map(|t| t.iter().map(|&v| v as u32))
            .collect();
        let num_tables = pp.tables.len();
        let gen_pows_flat: Vec<u32> = pp.gen_pows.iter().map(|&v| v as u32).collect();
        let num_iter = params.poly_len / 2 - 1;
        let t_flat = setup_start.elapsed().as_millis();
        debug!("CPU setup: pack_params={} ms, keys={} ms, flatten={} ms, total={} ms",
               t_pp, t_keys - t_pp, t_flat - t_keys, t_flat);

        // ── Phase 1: GPU hint ──
        let hint_start = Instant::now();

        // hint_gpu_ctx: kept alive until precomp grabs d_hint_0 (NTT path only)
        let (hint_gpu_ctx, hint_compute_ms) = if use_gemm_hint {
            assert!(
                db_rows <= MAX_GEMM_HINT_ROWS,
                "HINT=gemm: db_rows ({}) exceeds safe limit ({}). Use NTT hint.",
                db_rows, MAX_GEMM_HINT_ROWS
            );
            let ring_a_w32 = build_ring_a_w32(params, db_rows);
            let hint_ctx = cuda::SandwichOnlineContext::new(
                db_u8,
                db_rows,
                self.db_rows_padded(),
                db_cols,
                params,
                params.poly_len,
                gpu_tier,
            );
            let compute_start = Instant::now();
            hint_ctx.compute_hint_gemm_on_device(&ring_a_w32, params.poly_len);
            let compute_ms = compute_start.elapsed().as_micros() as f64 / 1000.0;
            debug!(
                "SandwichPIR hint (GPU GEMM) in {:.2} ms",
                hint_start.elapsed().as_micros() as f64 / 1000.0
            );
            // GEMM path: hint stays on device in d_intermediate, take ownership
            let ptr = hint_ctx.take_intermediate();
            drop(hint_ctx);
            (Err(ptr), compute_ms)
        } else {
            let t_gen = Instant::now();
            let preprocessed_query = self.generate_pseudorandom_query(SEED_0);
            let t_gen_ms = t_gen.elapsed().as_micros() as f64 / 1000.0;
            let query_ntt_flat: Vec<u64> = preprocessed_query
                .iter()
                .flat_map(|p| p.as_slice().to_vec())
                .collect();
            debug!("A-matrix generation: {:.2} ms ({} polys, CPU NTT)", t_gen_ms, preprocessed_query.len());
            let gpu_hint = cuda::SandwichOfflineContext::new(
                db_u8,
                &query_ntt_flat,
                db_rows,
                self.db_rows_padded(),
                db_cols,
                params,
            );
            let upload_ms = hint_start.elapsed().as_micros() as f64 / 1000.0;
            let compute_start = Instant::now();
            gpu_hint.compute_hint_on_device();
            let hint_compute_ms = compute_start.elapsed().as_micros() as f64 / 1000.0;
            debug!(
                "SandwichPIR hint: upload={:.2} ms, compute={:.2} ms, total={:.2} ms",
                upload_ms, hint_compute_ms, hint_start.elapsed().as_micros() as f64 / 1000.0
            );
            // NTT path: hint stays on GPU
            (Ok(gpu_hint), hint_compute_ms)
        };

        if let Some(ref mut m) = measurement {
            m.offline.simplepir_prep_time_ms = hint_compute_ms;
        }

        // ── Phase 2: InspiRING precomp ──
        // Hint stays on GPU — no download/re-upload round-trip.
        let gpu_start = Instant::now();

        let precomp_estimate = {
            let cpn = params.crt_count * params.poly_len;
            let num_to_pack_half = params.poly_len / 2;
            let num_iter = num_to_pack_half - 1;
            let mono = 2 * params.poly_len * cpn * 8;
            let act = num_rlwe_outputs * params.poly_len * cpn * 8;
            let r_all = 2 * num_rlwe_outputs * num_to_pack_half * cpn * 8;
            let bold_t =
                2 * num_rlwe_outputs * num_iter * params.t_exp_left * params.poly_len * 8;
            let w_all = 2 * num_iter * params.t_exp_left * cpn * 8;
            mono + act + r_all + bold_t + w_all
        };
        let db_size = db_u8.len();
        // The NTT hint (poly_len × db_cols u64s ≈ 2 GB at 8 GB DB) is
        // resident on GPU while InspiRING precomp runs, but the old
        // `precomp_estimate` did not account for it. That meant on
        // tight-memory GPUs like L4 (24 GB), `need_staged` returned
        // false even though the actual peak (DB + hint + precomp
        // buffers) exceeded VRAM, producing an OOM at `bold_t`
        // allocation inside precomp.cu. Add it explicitly and bump
        // the safety margin from 256 MB to 1 GB to cover CUDA
        // scratch, fragmentation, and any companion buffers we're
        // not tracking.
        let hint_size = params.poly_len * db_cols * std::mem::size_of::<u64>();
        let gpu_free = cuda::gpu_free_memory();
        let need_staged =
            gpu_free < (db_size + hint_size + precomp_estimate + (1 << 30));

        let t_ctx = Instant::now();
        let mut online_ctx: Option<cuda::SandwichOnlineContext> = if !need_staged {
            Some(cuda::SandwichOnlineContext::new(
                db_u8,
                db_rows,
                self.db_rows_padded(),
                db_cols,
                params,
                max_batch_size,
                gpu_tier,
            ))
        } else {
            debug!(
                "SandwichPIR: staged mode (GPU free={:.0}MiB, DB={:.0}MiB, hint={:.0}MiB, precomp={:.0}MiB)",
                gpu_free as f64 / (1024.0 * 1024.0),
                db_size as f64 / (1024.0 * 1024.0),
                hint_size as f64 / (1024.0 * 1024.0),
                precomp_estimate as f64 / (1024.0 * 1024.0)
            );
            None
        };

        println!("SandwichPIR online-ctx new (db image + upload): {:.2} ms (staged={})",
            t_ctx.elapsed().as_micros() as f64 / 1000.0, need_staged);
        // Extract the hint pointer and drop the offline context BEFORE
        // running InspiRING precomp. The offline context owns a full
        // copy of the DB on GPU (8 GB at 8 GB DB) that's only needed
        // for the hint computation itself — once the hint is produced,
        // the DB can be freed to make room for precomp's bold_t
        // allocation. On tight-memory GPUs like L4 (24 GB) this is the
        // difference between fitting and OOM: without this, DB + hint +
        // precomp buffers + bold_t exceeds VRAM during precomp.
        //
        // NTT path: take_hint_device_ptr nulls the pointer inside the
        //   offline context, then drop(ctx) frees the DB + query NTT +
        //   scratch but NOT the hint, which we own from now on.
        // GEMM path: the hint pointer was already taken from the online
        //   context earlier; nothing to do here.
        let t_drop = Instant::now();
        let hint_ptr: *mut u64 = match hint_gpu_ctx {
            Ok(ctx) => {
                let ptr = ctx.take_hint_device_ptr();
                drop(ctx); // frees DB + other offline scratch, keeps hint
                ptr
            }
            Err(ptr) => ptr,
        };
        println!("SandwichPIR hint-ctx drop: {:.2} ms", t_drop.elapsed().as_micros() as f64 / 1000.0);

        let t_constr = Instant::now();
        let mut inspir_gpu = cuda::SwInspirPrecompContext::new(
            hint_ptr as *const u64,
            db_cols as u32,
            params,
            num_rlwe_outputs as u32,
            &w_mask_flat,
            &v_mask_flat,
            &mod_inv_poly_flat,
            &tables_flat,
            num_tables as u32,
            &gen_pows_flat,
        );
        println!("SandwichPIR inspir-ctx construct: {:.2} ms", t_constr.elapsed().as_micros() as f64 / 1000.0);
        let t_comp = Instant::now();
        inspir_gpu.compute();
        println!("SandwichPIR inspir compute: {:.2} ms", t_comp.elapsed().as_micros() as f64 / 1000.0);
        // Free the hint buffer now that precomp is done reading it.
        let t_free = Instant::now();
        cuda::free_gpu(hint_ptr);
        println!("SandwichPIR hint free: {:.2} ms", t_free.elapsed().as_micros() as f64 / 1000.0);

        // ── Phase 3: Build M-matrix, upload DB ──
        let t_alloc = Instant::now();
        if need_staged {
            online_ctx = Some(cuda::SandwichOnlineContext::new_deferred(
                db_rows,
                self.db_rows_padded(),
                db_cols,
                params,
                max_batch_size,
                gpu_tier,
            ));
        }

        println!("SandwichPIR online-ctx alloc: {:.2} ms (staged={})",
            t_alloc.elapsed().as_micros() as f64 / 1000.0, need_staged);
        let gen_pows_for_packing: Vec<u32> = gen_pows_flat[..num_iter].to_vec();
        let t_pk = Instant::now();
        online_ctx.as_ref().unwrap().init_packing(
            &mut inspir_gpu,
            &tables_flat,
            &gen_pows_for_packing,
            num_iter,
        );
        println!("SandwichPIR init_packing: {:.2} ms", t_pk.elapsed().as_micros() as f64 / 1000.0);

        let compute_ms = gpu_start.elapsed().as_micros() as f64 / 1000.0;
        let upload_start = Instant::now();
        if need_staged {
            online_ctx.as_ref().unwrap().upload_db(db_u8);
        }
        let upload_ms = upload_start.elapsed().as_micros() as f64 / 1000.0;
        println!(
            "SandwichPIR offline: compute={:.2} ms (hint={:.2} + precomp={:.2}), db upload={:.2} ms",
            hint_compute_ms + compute_ms,
            hint_compute_ms,
            compute_ms,
            upload_ms
        );

        debug!(
            "SandwichPIR GPU offline phases 2+3 in {:.2} ms (staged={})",
            compute_ms + upload_ms,
            need_staged
        );

        // Compute-only offline time (excludes DB upload, includes hint compute + precomp + M-matrix)
        if let Some(ref mut m) = measurement {
            m.offline.server_time_ms = hint_compute_ms + compute_ms;
        }

        let offline_vals = OfflinePrecomputedValues {
            hint_0: vec![],
            y_constants: (Vec::new(), Vec::new()),
            prepacked_lwe: vec![],
            precomp: vec![],
            packing_type: PackingType::InspiRING,
            packing_params: Some(packing_params),
            precomp_inspir_vec: None,
            offline_packing_keys: None,
            cuda_context: None,
            sp_cuda_context: None,
            word_cuda_context: None,
        };

        (offline_vals, online_ctx.unwrap())
    }
}

// ═══════════════════════════════════════════════════════════════
// SandwichGpuServer — multi-GPU sharding primitive
//
// One database shard pinned to one GPU. Bundles the YServer (with its slab
// of the DB), the offline precomp, and the online GPU context. In single-GPU
// mode (default) `SandwichGpuServer::shards` has length 1; in multi-GPU mode
// (MULTI_GPU=N) it has length N.
//
// Both `pir_server::PirServer` and the bench (`scheme::run_sandwichpir_on_params`)
// build a `SandwichGpuServer` and dispatch query batches through its
// `compute_batch` method. The single-shard path is byte-identical to today's
// direct `gpu_online_ctx.compute_batch(...)` call (one branch + one Vec[0]
// index, optimized away in release). The multi-shard path spawns N-1 worker
// threads via `std::thread::scope` and concatenates per-client responses.
//
// This abstraction is intentionally shaped so it could later be extended
// from "local GPU shard" to a `Box<dyn Shard>` trait with a remote-HTTP
// implementation, enabling multi-node distributed serving without changing
// any caller's API or the `pir_client` wire format.
// ═══════════════════════════════════════════════════════════════

/// One DB shard owned by one GPU.
pub struct SandwichGpuShard {
    pub device_id: i32,
    /// Owned for lifetime of the shard (leaked via Box::leak in `SandwichGpuServer::new`).
    #[allow(dead_code)]
    pub params: &'static spiral_rs::params::Params,
    /// The shard's YServer holding its column slab of the DB.
    pub y_server: &'static YServer<'static, u8>,
    /// Held for lifetime ownership; not used by the GPU online path.
    #[allow(dead_code)]
    pub offline_values: OfflinePrecomputedValues<'static>,
    pub gpu_online_ctx: cuda::SandwichOnlineContext,
    /// First output index this shard produces (relative to the FULL `num_outputs`).
    /// Implicit in iteration order today; explicit for future remote-shard impl.
    #[allow(dead_code)]
    pub output_offset: usize,
    pub num_outputs_local: usize,
}

// Safety: all fields are read-only after construction; SandwichOnlineContext
// is already Send+Sync; leaked &'static references are trivially Send+Sync.
unsafe impl Send for SandwichGpuShard {}
unsafe impl Sync for SandwichGpuShard {}

/// Unified GPU server primitive — owns one or more shards of a logical DB.
///
/// Construction reads `MULTI_GPU` env var (default `1`) to decide how many
/// shards to build. Each shard pins itself to a different GPU before any
/// CUDA call and runs its own offline phase. Online queries go through
/// `compute_batch`, which dispatches to a single shard or fans out to all
/// shards in parallel.
///
/// `mask_template` holds the rescaled+bitpacked mask bytes for every
/// output (`[num_outputs][mask_bytes_per_output]`), concatenated across
/// shards in output order. It is computed ONCE on the GPU at offline
/// time (from the per-shard `d_a_hat` precomp output) and downloaded to
/// host. The GPU never touches mask bytes during online queries — the
/// `SandwichBatchResponse` returned by `compute_batch` carries body-only
/// bytes plus a shared reference to this template. Callers assemble the
/// full wire format only when needed (first-query clients); body-only
/// clients skip the mask end-to-end.
pub struct SandwichGpuServer {
    pub shards: Vec<SandwichGpuShard>,
    pub num_outputs: usize,
    pub response_bytes_per_output: usize,
    pub body_bytes_per_output: usize,
    pub mask_bytes_per_output: usize,
    pub max_batch_size: usize,
    pub poly_len: usize,
    pub q_prime_1: u64,
    pub q_prime_2: u64,
    pub modulus: u64,
    /// Persistent PINNED response buffer:
    /// `[max_batch_size][num_outputs][response_bytes_per_output]`.
    /// Mask slots are tiled at offline init and NEVER touched online.
    /// Body slots are overwritten each `compute_batch` by direct GPU→host
    /// `cudaMemcpy3D`, with the buffer's pinned backing allowing true
    /// async DMA with no internal staging copy. `full_response(k)`
    /// returns a zero-copy `&[u8]` slice into this buffer.
    response_buf: crate::cuda::sandwich::PinnedHostBuffer<u8>,
    /// Raw mask template (one copy): `[num_outputs][mask_bytes_per_output]`.
    pub mask_template: Vec<u8>,
}

/// Batch response from `compute_batch`. Holds a reference to the
/// persistent pinned `response_buf` with mask already in place and body
/// freshly DMA'd. Both `full_response(k)` (zero-copy slice) and
/// `body_only(k)` (gather allocation of `num_outputs * body_bytes`) are
/// supported. No mask bytes are ever copied online.
pub struct SandwichBatchResponse<'a> {
    response_buf: &'a [u8],
    num_outputs: usize,
    body_bytes_per_output: usize,
    mask_bytes_per_output: usize,
    response_bytes_per_output: usize,
}

impl<'a> SandwichBatchResponse<'a> {
    /// Body-only bytes for client `client_idx`, contiguous in output
    /// order: `num_outputs * body_bytes_per_output` bytes. The data is
    /// gathered from the interleaved `response_buf` via `num_outputs`
    /// small `extend_from_slice` calls; cost is trivial
    /// (~80 KB for default parameters, well under 0.1 ms on host).
    /// Used by body-only clients (steady state after the client has
    /// cached the static mask template).
    pub fn body_only(&self, client_idx: usize) -> Vec<u8> {
        let body_bpo = self.body_bytes_per_output;
        let resp_bpo = self.response_bytes_per_output;
        let mask_bpo = self.mask_bytes_per_output;
        let per_client = self.num_outputs * resp_bpo;
        let mut out = Vec::with_capacity(self.num_outputs * body_bpo);
        for o in 0..self.num_outputs {
            let slot = client_idx * per_client + o * resp_bpo;
            out.extend_from_slice(&self.response_buf[slot + mask_bpo..slot + resp_bpo]);
        }
        out
    }

    /// Full wire-format response for client `client_idx`. ZERO-COPY
    /// slice into the persistent pinned response buffer: mask was
    /// pre-filled at offline init, body was DMA'd in-place by
    /// `cudaMemcpy3D` during this batch's `compute_batch`.
    pub fn full_response(&self, client_idx: usize) -> &[u8] {
        let per_client = self.num_outputs * self.response_bytes_per_output;
        let s = client_idx * per_client;
        &self.response_buf[s..s + per_client]
    }

    /// Full wire-format bytes for one (client, output) pair. Zero-copy
    /// slice. `response_bytes_per_output` bytes: `[mask | body]`.
    pub fn full_response_output(&self, client_idx: usize, output_idx: usize) -> &[u8] {
        let per_client = self.num_outputs * self.response_bytes_per_output;
        let s = client_idx * per_client + output_idx * self.response_bytes_per_output;
        &self.response_buf[s..s + self.response_bytes_per_output]
    }
}

unsafe impl Send for SandwichGpuServer {}
unsafe impl Sync for SandwichGpuServer {}

/// Build a persistent pinned response buffer with the mask template
/// pre-tiled into every (client, output) slot. Body slots are
/// zero-initialised (GPU will overwrite them per batch). Layout:
/// `[max_batch_size][num_outputs][response_bytes_per_output]` where
/// each output chunk is `[mask_bytes | body_bytes]`.
fn build_response_buf(
    mask_template: &[u8],
    num_outputs: usize,
    mask_bytes_per_output: usize,
    response_bytes_per_output: usize,
    max_batch_size: usize,
) -> crate::cuda::sandwich::PinnedHostBuffer<u8> {
    let per_client = num_outputs * response_bytes_per_output;
    let total = max_batch_size * per_client;
    let mut buf = crate::cuda::sandwich::PinnedHostBuffer::<u8>::new(total);
    let slice = buf.as_mut_slice();
    slice.fill(0);
    for c in 0..max_batch_size {
        for o in 0..num_outputs {
            let dst = c * per_client + o * response_bytes_per_output;
            let src = o * mask_bytes_per_output;
            slice[dst..dst + mask_bytes_per_output]
                .copy_from_slice(&mask_template[src..src + mask_bytes_per_output]);
        }
    }
    buf
}

impl SandwichGpuServer {
    /// Build from a flat row-major DB and a fully-constructed `Params`.
    /// `db` must be `num_real_rows × item_size_bytes` bytes where
    /// `item_size_bytes = params.instances * params.poly_len` (for `p = 256`).
    ///
    /// Reads `MULTI_GPU` env var: if `N > 1`, slices DB column-wise into N
    /// owned slabs and builds N shards in parallel via `std::thread::scope`,
    /// one shard per GPU. Each shard's offline phase runs on its own device.
    ///
    /// Single-GPU mode (default) is byte-identical to the existing
    /// `YServer::new_from_flat_db` + `sandwich_gpu_offline` path.
    pub fn new(
        db: Vec<u8>,
        params: spiral_rs::params::Params,
        max_batch_size: usize,
        mut measurement: Option<&mut Measurement>,
    ) -> Self {
        use crate::params::GetQPrime;

        let num_shards = std::env::var("MULTI_GPU")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(1);

        let poly_len = params.poly_len;
        let item_size_bytes = params.instances * poly_len; // p = 256, 1 byte/entry
        let num_real_rows = db.len() / item_size_bytes;

        let num_outputs = params.instances;
        let q_prime_1 = params.get_q_prime_1();
        let q_prime_2 = params.get_q_prime_2();
        let q1_bits = (q_prime_2 as f64).log2().ceil() as usize;
        let q2_bits = (q_prime_1 as f64).log2().ceil() as usize;
        let mask_bytes_per_output = (q1_bits * poly_len + 7) / 8;
        let body_bytes_per_output = (q2_bits * poly_len + 7) / 8;
        let response_bytes_per_output = mask_bytes_per_output + body_bytes_per_output;
        let modulus = params.modulus;

        assert!(
            num_outputs >= num_shards,
            "num_outputs ({}) must be >= MULTI_GPU ({})",
            num_outputs, num_shards
        );

        // ── Single-shard fast path: identical to the pre-sharding code ──
        if num_shards == 1 {
            let leaked_params: &'static spiral_rs::params::Params = Box::leak(Box::new(params));
            let y_server: &'static YServer<'static, u8> = Box::leak(Box::new(
                YServer::<u8>::new_from_flat_db(leaked_params, &db, num_real_rows),
            ));
            drop(db);
            let (offline_values, gpu_online_ctx) =
                y_server.sandwich_gpu_offline(measurement.as_deref_mut(), max_batch_size);
            let mask_template = gpu_online_ctx.download_mask_template();
            debug_assert_eq!(mask_template.len(), num_outputs * mask_bytes_per_output);
            let response_buf = build_response_buf(
                &mask_template, num_outputs, mask_bytes_per_output,
                response_bytes_per_output, max_batch_size,
            );
            return SandwichGpuServer {
                shards: vec![SandwichGpuShard {
                    device_id: 0,
                    params: leaked_params,
                    y_server,
                    offline_values,
                    gpu_online_ctx,
                    output_offset: 0,
                    num_outputs_local: num_outputs,
                }],
                num_outputs,
                response_bytes_per_output,
                body_bytes_per_output,
                mask_bytes_per_output,
                max_batch_size,
                poly_len,
                q_prime_1,
                q_prime_2,
                modulus,
                response_buf,
                mask_template,
            };
        }

        // ── Multi-GPU path ──
        // Distribute outputs (= db columns / poly_len) as evenly as possible.
        // Shard k owns outputs in [num_outputs * k / N, num_outputs * (k+1) / N),
        // so the first `num_outputs % N` shards get ⌈num_outputs/N⌉ outputs and
        // the rest get ⌊num_outputs/N⌋. Imbalance ≤ 1 output → worst-case wall
        // is ⌈num_outputs/N⌉ / num_outputs of the single-GPU work.
        let shard_outs: Vec<usize> = (0..num_shards)
            .map(|k| (num_outputs * (k + 1) / num_shards) - (num_outputs * k / num_shards))
            .collect();
        let shard_byte_starts: Vec<usize> = (0..num_shards)
            .map(|k| (num_outputs * k / num_shards) * poly_len)
            .collect();
        let shard_byte_ends: Vec<usize> = (0..num_shards)
            .map(|k| (num_outputs * (k + 1) / num_shards) * poly_len)
            .collect();
        let shard_output_offsets: Vec<usize> = (0..num_shards)
            .map(|k| num_outputs * k / num_shards)
            .collect();

        log::info!(
            "SandwichGpuServer: sharding {} outputs into {} shards [{}]",
            num_outputs, num_shards,
            shard_outs.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")
        );

        // Slice the input DB column-wise. Each shard gets a contiguous Vec<u8>.
        let mut shard_dbs: Vec<Vec<u8>> = (0..num_shards)
            .map(|k| {
                let bytes_per_row = shard_byte_ends[k] - shard_byte_starts[k];
                Vec::with_capacity(num_real_rows * bytes_per_row)
            })
            .collect();
        for row_idx in 0..num_real_rows {
            let row_start = row_idx * item_size_bytes;
            for k in 0..num_shards {
                let col_start = row_start + shard_byte_starts[k];
                let col_end = row_start + shard_byte_ends[k];
                shard_dbs[k].extend_from_slice(&db[col_start..col_end]);
            }
        }
        drop(db);

        // ── Parallel build: one std::thread per shard, each pinned to its
        // device. Each thread runs the full offline pipeline (YServer build +
        // sandwich_gpu_offline) on its slab. CUDA's current device is
        // thread-local so the per-thread `set_device` call routes all CUDA
        // calls in that thread to the right GPU.
        //
        // This parallelizes the offline phase across GPUs, giving ~Nx
        // offline speedup (each shard's work is 1/N of single-GPU).
        let shards_and_meas: Vec<(SandwichGpuShard, Measurement)> =
            std::thread::scope(|s| {
                let handles: Vec<_> = shard_dbs
                    .into_iter()
                    .enumerate()
                    .map(|(k, slab)| {
                        // Each shard's params has its OWN instances count
                        // (may differ by 1 across shards when num_outputs
                        // does not divide num_shards evenly).
                        let mut params_shard = params.clone();
                        params_shard.instances = shard_outs[k];
                        let output_offset = shard_output_offsets[k];
                        let num_outputs_local = shard_outs[k];
                        s.spawn(move || {
                            cuda::sandwich::set_device(k as i32);
                            let params: &'static spiral_rs::params::Params =
                                Box::leak(Box::new(params_shard));
                            let y_server: &'static YServer<'static, u8> =
                                Box::leak(Box::new(YServer::<u8>::new_from_flat_db(
                                    params,
                                    &slab,
                                    num_real_rows,
                                )));
                            drop(slab);

                            log::info!(
                                "Running offline precomputation on shard {} (device {}, {} outputs)...",
                                k, k, num_outputs_local
                            );
                            let mut shard_meas = Measurement::default();
                            let (offline_values, gpu_online_ctx) = y_server
                                .sandwich_gpu_offline(Some(&mut shard_meas), max_batch_size);
                            log::info!(
                                "Shard {} offline done: prep={:.2} ms, total={:.2} ms",
                                k,
                                shard_meas.offline.simplepir_prep_time_ms,
                                shard_meas.offline.server_time_ms
                            );

                            (
                                SandwichGpuShard {
                                    device_id: k as i32,
                                    params,
                                    y_server,
                                    offline_values,
                                    gpu_online_ctx,
                                    output_offset,
                                    num_outputs_local,
                                },
                                shard_meas,
                            )
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("offline shard worker panicked"))
                    .collect()
            });

        let mut shards: Vec<SandwichGpuShard> = Vec::with_capacity(num_shards);
        let mut max_offline_ms: f64 = 0.0;
        let mut max_prep_ms: f64 = 0.0;
        for (shard, shard_meas) in shards_and_meas {
            max_offline_ms = max_offline_ms.max(shard_meas.offline.server_time_ms);
            max_prep_ms = max_prep_ms.max(shard_meas.offline.simplepir_prep_time_ms);
            shards.push(shard);
        }
        // Sort by device_id so iteration order is deterministic = output_offset order.
        shards.sort_by_key(|s| s.device_id);

        // Download each shard's mask template slab and concatenate in
        // output_offset order. The resulting Vec has the same byte layout
        // as the single-shard template (num_outputs × mask_bytes_per_output).
        let mut mask_template = Vec::with_capacity(num_outputs * mask_bytes_per_output);
        for shard in &shards {
            cuda::sandwich::set_device(shard.device_id);
            let shard_mask = shard.gpu_online_ctx.download_mask_template();
            debug_assert_eq!(
                shard_mask.len(),
                shard.num_outputs_local * mask_bytes_per_output
            );
            mask_template.extend_from_slice(&shard_mask);
        }

        let response_buf = build_response_buf(
            &mask_template, num_outputs, mask_bytes_per_output,
            response_bytes_per_output, max_batch_size,
        );

        // Restore device 0 so subsequent host CUDA calls don't get confused.
        cuda::sandwich::set_device(0);

        // Report aggregated max-offline-time (the wall-clock equivalent if
        // all shards ran in parallel — relevant if we ever do parallel build).
        if let Some(ref mut m) = measurement {
            m.offline.server_time_ms = max_offline_ms;
            m.offline.simplepir_prep_time_ms = max_prep_ms;
        }

        SandwichGpuServer {
            shards,
            num_outputs,
            response_bytes_per_output,
            body_bytes_per_output,
            mask_bytes_per_output,
            max_batch_size,
            poly_len,
            q_prime_1,
            q_prime_2,
            modulus,
            response_buf,
            mask_template,
        }
    }

    /// Like `new` but takes the database as a borrowed `&[u8]` instead
    /// of an owned `Vec<u8>`. Used by `pir_server::PirServer::new_from_file`
    /// to feed a memory-mapped file straight through to
    /// `YServer::new_from_flat_db` without an intermediate owned copy
    /// — peak host memory during construction drops from
    /// ~16 GB (caller's Vec + aligned buffer) to ~8 GB (aligned buffer
    /// + small kernel page-cache working set for the mmap).
    ///
    /// Single-shard mode gets the full memory win. Multi-shard mode
    /// still has to gather columns into per-shard owned Vecs (same
    /// peak as the Vec path) because the YServer constructor expects
    /// a contiguous row-major slab per shard — future work: plumb a
    /// column-range parameter into `YServer::new_from_flat_db` so
    /// each shard can transpose directly from sub-strides of the
    /// shared `&[u8]` without owning a copy.
    pub fn new_from_slice(
        db: &[u8],
        params: spiral_rs::params::Params,
        max_batch_size: usize,
        mut measurement: Option<&mut Measurement>,
    ) -> Self {
        use crate::params::GetQPrime;

        let num_shards = std::env::var("MULTI_GPU")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(1);

        let poly_len = params.poly_len;
        let item_size_bytes = params.instances * poly_len;
        let num_real_rows = db.len() / item_size_bytes;

        let num_outputs = params.instances;
        let q_prime_1 = params.get_q_prime_1();
        let q_prime_2 = params.get_q_prime_2();
        let q1_bits = (q_prime_2 as f64).log2().ceil() as usize;
        let q2_bits = (q_prime_1 as f64).log2().ceil() as usize;
        let mask_bytes_per_output = (q1_bits * poly_len + 7) / 8;
        let body_bytes_per_output = (q2_bits * poly_len + 7) / 8;
        let response_bytes_per_output = mask_bytes_per_output + body_bytes_per_output;
        let modulus = params.modulus;

        assert!(
            num_outputs >= num_shards,
            "num_outputs ({}) must be >= MULTI_GPU ({})",
            num_outputs, num_shards
        );

        // ── Single-shard: pass &db straight to YServer; no owned Vec ──
        if num_shards == 1 {
            let leaked_params: &'static spiral_rs::params::Params = Box::leak(Box::new(params));
            let y_server: &'static YServer<'static, u8> = Box::leak(Box::new(
                YServer::<u8>::new_from_flat_db(leaked_params, db, num_real_rows),
            ));
            // `db` is caller-owned; we never took ownership, nothing to drop.
            let (offline_values, gpu_online_ctx) =
                y_server.sandwich_gpu_offline(measurement.as_deref_mut(), max_batch_size);
            let mask_template = gpu_online_ctx.download_mask_template();
            debug_assert_eq!(mask_template.len(), num_outputs * mask_bytes_per_output);
            let response_buf = build_response_buf(
                &mask_template, num_outputs, mask_bytes_per_output,
                response_bytes_per_output, max_batch_size,
            );
            return SandwichGpuServer {
                shards: vec![SandwichGpuShard {
                    device_id: 0,
                    params: leaked_params,
                    y_server,
                    offline_values,
                    gpu_online_ctx,
                    output_offset: 0,
                    num_outputs_local: num_outputs,
                }],
                num_outputs,
                response_bytes_per_output,
                body_bytes_per_output,
                mask_bytes_per_output,
                max_batch_size,
                poly_len,
                q_prime_1,
                q_prime_2,
                modulus,
                response_buf,
                mask_template,
            };
        }

        // ── Multi-shard path: still has to gather column subsets into
        // per-shard owned Vecs (same memory peak as `new()`). The mmap
        // optimization only helps single-shard for now.
        let db_owned: Vec<u8> = db.to_vec();
        Self::new(db_owned, params, max_batch_size, measurement)
    }

    /// Construct a SandwichGpuServer by streaming the database from an
    /// `io::Read` source. Single-shard only — multi-shard requires
    /// buffering the whole file and falls back to `new_from_slice`.
    ///
    /// This is the fastest path for real DBs on Lustre or other network
    /// filesystems where mmap page-fault RPC overhead kills throughput.
    /// Peak host memory: ~8 GB aligned buffer + 8 MB row-batch scratch.
    pub fn new_from_reader<R: std::io::Read>(
        reader: &mut R,
        num_real_rows: usize,
        params: spiral_rs::params::Params,
        max_batch_size: usize,
        mut measurement: Option<&mut Measurement>,
    ) -> std::io::Result<Self> {
        use crate::params::GetQPrime;

        let num_shards = std::env::var("MULTI_GPU")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(1);
        assert_eq!(
            num_shards, 1,
            "new_from_reader only supports single-shard (MULTI_GPU=1). \
             For multi-shard, fall back to SandwichGpuServer::new_from_slice \
             with a pre-loaded Vec."
        );

        let poly_len = params.poly_len;
        let num_outputs = params.instances;
        let q_prime_1 = params.get_q_prime_1();
        let q_prime_2 = params.get_q_prime_2();
        let q1_bits = (q_prime_2 as f64).log2().ceil() as usize;
        let q2_bits = (q_prime_1 as f64).log2().ceil() as usize;
        let mask_bytes_per_output = (q1_bits * poly_len + 7) / 8;
        let body_bytes_per_output = (q2_bits * poly_len + 7) / 8;
        let response_bytes_per_output = mask_bytes_per_output + body_bytes_per_output;
        let modulus = params.modulus;

        let leaked_params: &'static spiral_rs::params::Params = Box::leak(Box::new(params));
        let y_server: &'static YServer<'static, u8> = Box::leak(Box::new(
            YServer::<u8>::new_from_reader(leaked_params, reader, num_real_rows)?,
        ));

        let (offline_values, gpu_online_ctx) =
            y_server.sandwich_gpu_offline(measurement.as_deref_mut(), max_batch_size);
        let mask_template = gpu_online_ctx.download_mask_template();
        debug_assert_eq!(mask_template.len(), num_outputs * mask_bytes_per_output);
        let response_buf = build_response_buf(
            &mask_template, num_outputs, mask_bytes_per_output,
            response_bytes_per_output, max_batch_size,
        );
        Ok(SandwichGpuServer {
            shards: vec![SandwichGpuShard {
                device_id: 0,
                params: leaked_params,
                y_server,
                offline_values,
                gpu_online_ctx,
                output_offset: 0,
                num_outputs_local: num_outputs,
            }],
            num_outputs,
            response_bytes_per_output,
            body_bytes_per_output,
            mask_bytes_per_output,
            max_batch_size,
            poly_len,
            q_prime_1,
            q_prime_2,
            modulus,
            response_buf,
            mask_template,
        })
    }

    /// Bench-only constructor: build a SandwichGpuServer with a random
    /// database generated **directly** into each shard's aligned DB
    /// buffer, skipping any intermediate `Vec<u8>`. Fixes two pain
    /// points of the flat-DB path for the bench:
    ///   - no 2× DB memory peak during construction (OOM risk on
    ///     small-RAM hosts like g6.xlarge where 8 GB DB + 8 GB aligned
    ///     copy = 16 GB peak)
    ///   - no per-byte `fastrand::u8` + `collect` + tile-transpose
    ///     (~15–40 s); uses chunked xorshift64 filling 8 bytes per
    ///     step for ~1–2 s total on 8 GB.
    ///
    /// Multi-GPU dispatch is identical to `new()`: each shard spawns a
    /// worker thread via `std::thread::scope` and constructs its own
    /// randomly-filled YServer independently (no cross-shard
    /// dependencies).
    ///
    /// This constructor is **only** valid when `pt_modulus == 256`
    /// (the SandwichPIR default). Production deployments with real
    /// data must use `new()`.
    pub fn new_random_db(
        params: spiral_rs::params::Params,
        max_batch_size: usize,
        mut measurement: Option<&mut Measurement>,
    ) -> Self {
        use crate::params::GetQPrime;

        assert_eq!(
            params.pt_modulus, 256,
            "new_random_db assumes pt_modulus = 256 (SandwichPIR default)"
        );

        let num_shards = std::env::var("MULTI_GPU")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(1);

        let poly_len = params.poly_len;
        let num_real_rows = 1 << (params.db_dim_1 + params.poly_len_log2);

        let num_outputs = params.instances;
        let q_prime_1 = params.get_q_prime_1();
        let q_prime_2 = params.get_q_prime_2();
        let q1_bits = (q_prime_2 as f64).log2().ceil() as usize;
        let q2_bits = (q_prime_1 as f64).log2().ceil() as usize;
        let mask_bytes_per_output = (q1_bits * poly_len + 7) / 8;
        let body_bytes_per_output = (q2_bits * poly_len + 7) / 8;
        let response_bytes_per_output = mask_bytes_per_output + body_bytes_per_output;
        let modulus = params.modulus;

        assert!(
            num_outputs >= num_shards,
            "num_outputs ({}) must be >= MULTI_GPU ({})",
            num_outputs, num_shards
        );

        // ── Single-shard fast path ──
        if num_shards == 1 {
            let leaked_params: &'static spiral_rs::params::Params = Box::leak(Box::new(params));
            let y_server: &'static YServer<'static, u8> = Box::leak(Box::new(
                YServer::<u8>::new_random_filled(leaked_params, num_real_rows),
            ));
            let (offline_values, gpu_online_ctx) =
                y_server.sandwich_gpu_offline(measurement.as_deref_mut(), max_batch_size);
            let mask_template = gpu_online_ctx.download_mask_template();
            debug_assert_eq!(mask_template.len(), num_outputs * mask_bytes_per_output);
            let response_buf = build_response_buf(
                &mask_template, num_outputs, mask_bytes_per_output,
                response_bytes_per_output, max_batch_size,
            );
            return SandwichGpuServer {
                shards: vec![SandwichGpuShard {
                    device_id: 0,
                    params: leaked_params,
                    y_server,
                    offline_values,
                    gpu_online_ctx,
                    output_offset: 0,
                    num_outputs_local: num_outputs,
                }],
                num_outputs,
                response_bytes_per_output,
                body_bytes_per_output,
                mask_bytes_per_output,
                max_batch_size,
                poly_len,
                q_prime_1,
                q_prime_2,
                modulus,
                response_buf,
                mask_template,
            };
        }

        // ── Multi-shard path ──
        // Distribute outputs (= db columns / poly_len) as evenly as
        // possible. Unlike `new()`, there is no source Vec<u8> to slice:
        // each shard generates its own random DB slab directly into its
        // aligned buffer, in parallel via `std::thread::scope`.
        let shard_outs: Vec<usize> = (0..num_shards)
            .map(|k| (num_outputs * (k + 1) / num_shards) - (num_outputs * k / num_shards))
            .collect();
        let shard_output_offsets: Vec<usize> = (0..num_shards)
            .map(|k| num_outputs * k / num_shards)
            .collect();

        log::info!(
            "SandwichGpuServer (random): sharding {} outputs into {} shards [{}]",
            num_outputs, num_shards,
            shard_outs.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")
        );

        let shards_and_meas: Vec<(SandwichGpuShard, Measurement)> =
            std::thread::scope(|s| {
                let handles: Vec<_> = (0..num_shards)
                    .map(|k| {
                        let mut params_shard = params.clone();
                        params_shard.instances = shard_outs[k];
                        let output_offset = shard_output_offsets[k];
                        let num_outputs_local = shard_outs[k];
                        s.spawn(move || {
                            cuda::sandwich::set_device(k as i32);
                            let params: &'static spiral_rs::params::Params =
                                Box::leak(Box::new(params_shard));
                            let y_server: &'static YServer<'static, u8> =
                                Box::leak(Box::new(YServer::<u8>::new_random_filled(
                                    params, num_real_rows,
                                )));

                            log::info!(
                                "Random-filled shard {} (device {}, {} outputs), running offline...",
                                k, k, num_outputs_local
                            );
                            let mut shard_meas = Measurement::default();
                            let (offline_values, gpu_online_ctx) = y_server
                                .sandwich_gpu_offline(Some(&mut shard_meas), max_batch_size);

                            (
                                SandwichGpuShard {
                                    device_id: k as i32,
                                    params,
                                    y_server,
                                    offline_values,
                                    gpu_online_ctx,
                                    output_offset,
                                    num_outputs_local,
                                },
                                shard_meas,
                            )
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("offline shard worker panicked"))
                    .collect()
            });

        let mut shards: Vec<SandwichGpuShard> = Vec::with_capacity(num_shards);
        let mut max_offline_ms: f64 = 0.0;
        let mut max_prep_ms: f64 = 0.0;
        for (shard, shard_meas) in shards_and_meas {
            max_offline_ms = max_offline_ms.max(shard_meas.offline.server_time_ms);
            max_prep_ms = max_prep_ms.max(shard_meas.offline.simplepir_prep_time_ms);
            shards.push(shard);
        }
        shards.sort_by_key(|s| s.device_id);

        let mut mask_template = Vec::with_capacity(num_outputs * mask_bytes_per_output);
        for shard in &shards {
            cuda::sandwich::set_device(shard.device_id);
            let shard_mask = shard.gpu_online_ctx.download_mask_template();
            debug_assert_eq!(
                shard_mask.len(),
                shard.num_outputs_local * mask_bytes_per_output
            );
            mask_template.extend_from_slice(&shard_mask);
        }

        let response_buf = build_response_buf(
            &mask_template, num_outputs, mask_bytes_per_output,
            response_bytes_per_output, max_batch_size,
        );

        cuda::sandwich::set_device(0);

        if let Some(ref mut m) = measurement {
            m.offline.server_time_ms = max_offline_ms;
            m.offline.simplepir_prep_time_ms = max_prep_ms;
        }

        SandwichGpuServer {
            shards,
            num_outputs,
            response_bytes_per_output,
            body_bytes_per_output,
            mask_bytes_per_output,
            max_batch_size,
            poly_len,
            q_prime_1,
            q_prime_2,
            modulus,
            response_buf,
            mask_template,
        }
    }

    /// Run the online pipeline. Each shard DMAs its body bytes directly
    /// into the body slots of the persistent pinned `response_buf` via
    /// `cudaMemcpy3D` (mask slots were pre-filled at offline and are
    /// untouched). No intermediate host Vec, no scatter loop — the GPU
    /// writes straight into the wire-format buffer.
    ///
    /// Returns a `SandwichBatchResponse` whose `full_response(k)` is a
    /// zero-copy `&[u8]` slice into the pinned buffer.
    pub fn compute_batch(
        &mut self,
        queries: &[u64],
        y_body: &[u64],
        z_body: &[u64],
        k: usize,
    ) -> SandwichBatchResponse<'_> {
        let body_bpo = self.body_bytes_per_output;
        let mask_bpo = self.mask_bytes_per_output;
        let resp_bpo = self.response_bytes_per_output;
        let rho = self.num_outputs;

        // Raw pointer wrapper: shards write to DISJOINT byte ranges of
        // response_buf (determined by each shard's `output_offset`), so
        // sharing a *mut u8 across worker threads is safe. We can't
        // express this disjointness with Rust borrow checker so we use
        // a Send wrapper around the raw pointer.
        #[derive(Clone, Copy)]
        struct SendMutPtr(*mut u8);
        unsafe impl Send for SendMutPtr {}
        let resp_ptr = SendMutPtr(self.response_buf.as_mut_ptr());

        // ── Single-shard fast path ──
        if self.shards.len() == 1 {
            let shard = &self.shards[0];
            unsafe {
                shard.gpu_online_ctx.compute_batch_into(
                    queries, y_body, z_body, k,
                    resp_ptr.0,
                    /* dst_byte_offset */ mask_bpo,
                    /* dst_row_pitch   */ resp_bpo,
                    /* dst_slice_ysize */ rho,
                    /* copy_height     */ rho,
                    /* body_bpo        */ body_bpo,
                );
            }
        } else {
            // ── Multi-GPU dispatch ──
            // Each shard owns outputs [output_offset .. output_offset + num_outputs_local),
            // writing into the same pinned response_buf at disjoint byte
            // ranges within each client slot. We spawn workers for shards
            // 1..N and run shard 0 on the main thread in parallel.
            let shards = &self.shards;
            std::thread::scope(|s| {
                let handles: Vec<_> = shards.iter().skip(1).map(|shard| {
                    let ptr_wrap = resp_ptr; // move the SendMutPtr wrapper, not a raw *mut u8
                    s.spawn(move || {
                        let ptr = ptr_wrap; // keep the wrapper alive inside the closure
                        cuda::sandwich::set_device(shard.device_id);
                        unsafe {
                            shard.gpu_online_ctx.compute_batch_into(
                                queries, y_body, z_body, k,
                                ptr.0,
                                shard.output_offset * resp_bpo + mask_bpo,
                                resp_bpo,
                                rho,
                                shard.num_outputs_local,
                                body_bpo,
                            );
                        }
                    })
                }).collect();

                let primary = &shards[0];
                cuda::sandwich::set_device(primary.device_id);
                unsafe {
                    primary.gpu_online_ctx.compute_batch_into(
                        queries, y_body, z_body, k,
                        resp_ptr.0,
                        primary.output_offset * resp_bpo + mask_bpo,
                        resp_bpo,
                        rho,
                        primary.num_outputs_local,
                        body_bpo,
                    );
                }
                for h in handles {
                    h.join().expect("shard worker panicked");
                }
            });
        }

        // Borrow the filled prefix of the pinned buffer as the response view.
        let per_client = rho * resp_bpo;
        SandwichBatchResponse {
            response_buf: &self.response_buf.as_slice()[..k * per_client],
            num_outputs: rho,
            body_bytes_per_output: body_bpo,
            mask_bytes_per_output: mask_bpo,
            response_bytes_per_output: resp_bpo,
        }
    }

    /// Reassemble row bytes by concatenating each shard's column slab in order.
    /// (For testing / `get_row_direct` — not on the hot query path.)
    pub fn get_row_direct(&self, row: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for shard in &self.shards {
            out.extend(shard.y_server.get_row(row).iter().map(|&x| x as u8));
        }
        out
    }
}
