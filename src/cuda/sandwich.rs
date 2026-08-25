/// CUDA FFI bindings for SandwichPIR GPU offline (NTT hint) + online (matmul + TC packing).
///
/// All GPU code lives in src/cuda/sandwich/. No modifications to existing CUDA files.

use super::flatten_ntt_tables;
use crate::params::GetQPrime;

// ==================== Device control (multi-GPU sharding) ====================
//
// CUDA's current device is *thread-local*. The Rust sharded orchestrator
// spawns one worker thread per GPU shard and calls `set_device(shard_idx)`
// at the top of each thread, so every subsequent CUDA library call from
// that thread routes to the right device.

#[cfg(feature = "cuda")]
extern "C" {
    fn sandwich_set_device(dev: i32) -> i32;
    fn sandwich_device_count() -> i32;
    fn sandwich_alloc_pinned(bytes: usize) -> *mut std::ffi::c_void;
    fn sandwich_free_pinned(ptr: *mut std::ffi::c_void);
}

/// Pin the calling thread to the given CUDA device. Returns 0 on success.
#[cfg(feature = "cuda")]
pub fn set_device(dev: i32) -> i32 {
    unsafe { sandwich_set_device(dev) }
}

/// Number of CUDA-capable devices visible to the process.
#[cfg(feature = "cuda")]
pub fn device_count() -> i32 {
    unsafe { sandwich_device_count() }
}

// ==================== Pinned host memory ====================
//
// Buffers backed by `cudaHostAlloc`-allocated page-locked memory.
// `cudaMemcpyAsync` uses these as DMA sources directly (no internal
// staging copy, no host thread blocking, true async transfer).
// Critical for multi-GPU sharded uploads where multiple host threads
// would otherwise serialize on the CPU staging memcpy.

/// Owned page-locked host buffer of `T` elements.
#[cfg(feature = "cuda")]
pub struct PinnedHostBuffer<T: Copy> {
    ptr: *mut T,
    len: usize,
}

#[cfg(feature = "cuda")]
unsafe impl<T: Copy + Send> Send for PinnedHostBuffer<T> {}
#[cfg(feature = "cuda")]
unsafe impl<T: Copy + Sync> Sync for PinnedHostBuffer<T> {}

#[cfg(feature = "cuda")]
impl<T: Copy> PinnedHostBuffer<T> {
    /// Allocate a pinned host buffer holding `len` elements of `T`.
    /// Panics on allocation failure.
    pub fn new(len: usize) -> Self {
        let bytes = len * std::mem::size_of::<T>();
        let ptr = unsafe { sandwich_alloc_pinned(bytes) } as *mut T;
        if ptr.is_null() {
            panic!("PinnedHostBuffer::new({}): allocation of {} bytes failed", len, bytes);
        }
        PinnedHostBuffer { ptr, len }
    }

    pub fn len(&self) -> usize { self.len }
    pub fn as_ptr(&self) -> *const T { self.ptr }
    pub fn as_mut_ptr(&mut self) -> *mut T { self.ptr }
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

#[cfg(feature = "cuda")]
impl<T: Copy> Drop for PinnedHostBuffer<T> {
    fn drop(&mut self) {
        unsafe { sandwich_free_pinned(self.ptr as *mut std::ffi::c_void) }
    }
}

// ==================== Offline: NTT Hint ====================

#[cfg(feature = "cuda")]
extern "C" {
    fn sandwich_offline_init(
        db: *const u8,
        db_rows: usize, db_rows_padded: usize, db_cols: usize,
        query_ntt: *const u64,
        poly_len: u32,
        moduli: *const u64,
        barrett_cr: *const u64,
        forward_table: *const u64,
        forward_prime_table: *const u64,
        inverse_table: *const u64,
        inverse_prime_table: *const u64,
        modulus: u64,
    ) -> *mut std::ffi::c_void;

    fn sandwich_offline_compute_hint(context: *mut std::ffi::c_void) -> i32;
    fn sandwich_offline_get_hint(context: *mut std::ffi::c_void, hint_out: *mut u64) -> i32;
    fn sandwich_offline_get_hint_device_ptr(context: *mut std::ffi::c_void) -> *mut u64;
    fn sandwich_offline_take_hint_device_ptr(context: *mut std::ffi::c_void) -> *mut u64;
    fn sandwich_offline_free(context: *mut std::ffi::c_void);
}

#[cfg(feature = "cuda")]
pub struct SandwichOfflineContext {
    ctx: *mut std::ffi::c_void,
    pub poly_len: usize,
    pub db_cols: usize,
}

#[cfg(feature = "cuda")]
unsafe impl Send for SandwichOfflineContext {}
#[cfg(feature = "cuda")]
unsafe impl Sync for SandwichOfflineContext {}

#[cfg(feature = "cuda")]
impl SandwichOfflineContext {
    pub fn new(
        db: &[u8],
        query_ntt: &[u64],
        db_rows: usize, db_rows_padded: usize, db_cols: usize,
        params: &spiral_rs::params::Params,
    ) -> Self {
        let (fwd, fwd_p, inv, inv_p) = flatten_ntt_tables(params);
        let ctx = unsafe {
            sandwich_offline_init(
                db.as_ptr(), db_rows, db_rows_padded, db_cols,
                query_ntt.as_ptr(),
                params.poly_len as u32,
                params.moduli.as_ptr(),
                params.barrett_cr_1.as_ptr(),
                fwd.as_ptr(), fwd_p.as_ptr(),
                inv.as_ptr(), inv_p.as_ptr(),
                params.modulus,
            )
        };
        assert!(!ctx.is_null(), "SandwichPIR offline init failed");
        Self { ctx, poly_len: params.poly_len, db_cols }
    }

    /// Compute NTT hint on GPU and download to host.
    /// Returns hint in row-major: hint[z * db_cols + col].
    pub fn compute_hint(&self) -> Vec<u64> {
        self.compute_hint_on_device();
        let mut hint = vec![0u64; self.poly_len * self.db_cols];
        let result = unsafe { sandwich_offline_get_hint(self.ctx, hint.as_mut_ptr()) };
        assert_eq!(result, 0, "SandwichPIR hint download failed");
        hint
    }

    /// Compute NTT hint on GPU, keep on device. Use get_hint_device_ptr() to access.
    pub fn compute_hint_on_device(&self) {
        let result = unsafe { sandwich_offline_compute_hint(self.ctx) };
        assert_eq!(result, 0, "SandwichPIR hint computation failed");
    }

    /// Get device pointer to the computed hint (valid until context is dropped).
    pub fn get_hint_device_ptr(&self) -> *mut u64 {
        unsafe { sandwich_offline_get_hint_device_ptr(self.ctx) }
    }

    /// Take ownership of the hint device pointer (caller must cudaFree it).
    /// Nulls out the internal pointer so drop() won't free it.
    pub fn take_hint_device_ptr(&self) -> *mut u64 {
        unsafe { sandwich_offline_take_hint_device_ptr(self.ctx) }
    }
}

#[cfg(feature = "cuda")]
impl Drop for SandwichOfflineContext {
    fn drop(&mut self) {
        unsafe { sandwich_offline_free(self.ctx) }
    }
}

// ==================== InspiRING Precomp ====================

#[cfg(feature = "cuda")]
extern "C" {
    fn sw_inspir_precomp_init(
        d_hint_0: *const u64, db_cols: u32,
        poly_len: u32, crt_count: u32, t_exp_left: u32,
        modulus_log2: u32, q2_bits: u32, num_outputs: u32,
        moduli: *const u64, barrett_cr: *const u64,
        forward_table: *const u64, forward_prime_table: *const u64,
        inverse_table: *const u64, inverse_prime_table: *const u64,
        mod0_inv_mod1: u64, mod1_inv_mod0: u64,
        barrett_cr_0_modulus: u64, barrett_cr_1_modulus: u64, modulus: u64,
        w_mask: *const u64, v_mask: *const u64, mod_inv_poly: *const u64,
        tables: *const u32, num_tables: u32,
        gen_pows: *const u32, gen_pows_len: u32,
    ) -> *mut std::ffi::c_void;

    fn sw_inspir_precomp_compute(context: *mut std::ffi::c_void);
    fn sw_inspir_precomp_get_results(
        context: *mut std::ffi::c_void,
        out_bold_t: *mut *mut u64, out_bold_t_bar: *mut *mut u64,
        out_bold_t_hat: *mut *mut u64, out_a_hat: *mut *mut u64,
        s1: *mut usize, s2: *mut usize, s3: *mut usize, s4: *mut usize,
    );
    fn sw_inspir_precomp_free(context: *mut std::ffi::c_void, free_outputs: bool);
}

#[cfg(feature = "cuda")]
pub struct SwInspirPrecompContext {
    ctx: *mut std::ffi::c_void,
}

#[cfg(feature = "cuda")]
impl SwInspirPrecompContext {
    pub fn new(
        d_hint_0: *const u64, db_cols: u32,
        params: &spiral_rs::params::Params, num_outputs: u32,
        w_mask: &[u64], v_mask: &[u64], mod_inv_poly: &[u64],
        tables: &[u32], num_tables: u32, gen_pows: &[u32],
    ) -> Self {
        let (fwd, fwd_p, inv, inv_p) = flatten_ntt_tables(params);
        let ctx = unsafe {
            sw_inspir_precomp_init(
                d_hint_0, db_cols,
                params.poly_len as u32, params.crt_count as u32,
                params.t_exp_left as u32, params.modulus_log2 as u32,
                params.q2_bits as u32, num_outputs,
                params.moduli.as_ptr(), params.barrett_cr_1.as_ptr(),
                fwd.as_ptr(), fwd_p.as_ptr(), inv.as_ptr(), inv_p.as_ptr(),
                params.mod0_inv_mod1, params.mod1_inv_mod0,
                params.barrett_cr_0_modulus, params.barrett_cr_1_modulus,
                params.modulus,
                w_mask.as_ptr(), v_mask.as_ptr(), mod_inv_poly.as_ptr(),
                tables.as_ptr(), num_tables,
                gen_pows.as_ptr(), gen_pows.len() as u32,
            )
        };
        assert!(!ctx.is_null(), "SwInspirPrecomp init failed");
        Self { ctx }
    }

    pub fn compute(&mut self) {
        unsafe { sw_inspir_precomp_compute(self.ctx) }
    }

    /// Take ownership of device pointers to precomp results.
    pub fn take_results(&mut self) -> (*mut u64, *mut u64, *mut u64, *mut u64) {
        let (mut t, mut tb, mut th, mut ah) = (
            std::ptr::null_mut(), std::ptr::null_mut(),
            std::ptr::null_mut(), std::ptr::null_mut(),
        );
        let (mut s1, mut s2, mut s3, mut s4) = (0usize, 0, 0, 0);
        unsafe {
            sw_inspir_precomp_get_results(
                self.ctx, &mut t, &mut tb, &mut th, &mut ah,
                &mut s1, &mut s2, &mut s3, &mut s4,
            );
        }
        (t, tb, th, ah)
    }
}

#[cfg(feature = "cuda")]
impl Drop for SwInspirPrecompContext {
    fn drop(&mut self) {
        unsafe { sw_inspir_precomp_free(self.ctx, false) }
    }
}

// ==================== Online: Matmul + TC Packing ====================

#[cfg(feature = "cuda")]
extern "C" {
    fn sandwich_online_init(
        db: *const u8,
        db_rows: usize, db_rows_padded: usize, db_cols: usize,
        poly_len: usize, t_exp_left: usize,
        num_outputs: usize, max_batch_size: usize,
        Q: u64, q_prime_1: u64, q_prime_2: u64,
        response_bytes_per_output: usize,
        h_moduli: *const u64, h_barrett_cr: *const u64,
        h_fwd: *const u64, h_fwd_p: *const u64,
        h_inv: *const u64, h_inv_p: *const u64,
        modulus: u64,
        gpu_tier: i32,
    ) -> *mut std::ffi::c_void;

    fn sandwich_online_upload_db(context: *mut std::ffi::c_void, h_db: *const u8);
    fn sandwich_online_init_packing(
        context: *mut std::ffi::c_void,
        d_bold_t: *mut u64, d_bold_t_bar: *mut u64,
        d_bold_t_hat: *const u64, d_a_hat: *const u64,
        d_tables: *const u32, d_gen_pows: *const u32,
        num_iter: usize,
        tables_count: usize, gen_pows_count: usize,
    );

    fn sandwich_online_compute(
        context: *mut std::ffi::c_void,
        h_queries: *const u64, batch_size: usize,
        h_y_body: *const u64, h_z_body: *const u64,
        h_response_buf: *mut u8,
        dst_byte_offset: usize,
        dst_row_pitch: usize,
        dst_slice_ysize: usize,
        copy_height: usize,
        body_bpo: usize,
    );

    fn sandwich_online_hint_gemm(
        context: *mut std::ffi::c_void,
        h_a_rows: *const u64, batch_size: usize,
        h_result_out: *mut u64,
    );
    fn sandwich_online_hint_gemm_on_device(
        context: *mut std::ffi::c_void,
        h_a_rows: *const u64, batch_size: usize,
    );
    fn sandwich_online_take_intermediate(context: *mut std::ffi::c_void) -> *mut u64;
    fn sandwich_online_download_mask_template(
        context: *mut std::ffi::c_void,
        h_out: *mut u8,
    );
    fn sandwich_online_free(context: *mut std::ffi::c_void);
}

#[cfg(feature = "cuda")]
pub struct SandwichOnlineContext {
    ctx: *mut std::ffi::c_void,
    pub num_outputs: usize,
    /// Full wire-format response size per output (mask + body). Retained
    /// for metadata / callers that need to reason about the assembled
    /// response; NOT the size that `compute_batch` returns.
    pub response_bytes_per_output: usize,
    /// What `compute_batch` actually produces per output (body-only).
    pub body_bytes_per_output: usize,
    /// Static mask template size per output.
    pub mask_bytes_per_output: usize,
}

#[cfg(feature = "cuda")]
unsafe impl Send for SandwichOnlineContext {}
#[cfg(feature = "cuda")]
unsafe impl Sync for SandwichOnlineContext {}

#[cfg(feature = "cuda")]
impl SandwichOnlineContext {
    /// Create without uploading DB (for staged memory management).
    /// Call upload_db() later after M-matrix build frees bold_t.
    pub fn new_deferred(
        db_rows: usize, db_rows_padded: usize, db_cols: usize,
        params: &spiral_rs::params::Params,
        max_batch_size: usize,
        gpu_tier: i32,
    ) -> Self {
        let poly_len = params.poly_len;
        let num_outputs = db_cols / poly_len;
        let q_prime_1 = params.get_q_prime_1();
        let q_prime_2 = params.get_q_prime_2();
        let q1_bits = (q_prime_2 as f64).log2().ceil() as usize;
        let q2_bits = (q_prime_1 as f64).log2().ceil() as usize;
        let response_bytes_per_output = ((q1_bits + q2_bits) * poly_len + 7) / 8;
        let mask_bytes_per_output = (q1_bits * poly_len + 7) / 8;
        let body_bytes_per_output = (q2_bits * poly_len + 7) / 8;

        let (fwd, fwd_p, inv, inv_p) = flatten_ntt_tables(params);
        let ctx = unsafe {
            sandwich_online_init(
                std::ptr::null(), // no DB yet
                db_rows, db_rows_padded, db_cols,
                poly_len, params.t_exp_left, num_outputs, max_batch_size,
                params.modulus, q_prime_1, q_prime_2, response_bytes_per_output,
                params.moduli.as_ptr(), params.barrett_cr_1.as_ptr(),
                fwd.as_ptr(), fwd_p.as_ptr(), inv.as_ptr(), inv_p.as_ptr(),
                params.modulus, gpu_tier,
            )
        };
        assert!(!ctx.is_null(), "SandwichPIR online init (deferred) failed");
        Self {
            ctx, num_outputs, response_bytes_per_output,
            body_bytes_per_output, mask_bytes_per_output,
        }
    }

    /// Upload DB after M-matrix build (for staged memory management).
    pub fn upload_db(&self, db: &[u8]) {
        unsafe { sandwich_online_upload_db(self.ctx, db.as_ptr()); }
    }

    pub fn new(
        db: &[u8],
        db_rows: usize, db_rows_padded: usize, db_cols: usize,
        params: &spiral_rs::params::Params,
        max_batch_size: usize,
        gpu_tier: i32,
    ) -> Self {
        let poly_len = params.poly_len;
        let num_outputs = db_cols / poly_len;
        let q_prime_1 = params.get_q_prime_1();
        let q_prime_2 = params.get_q_prime_2();
        let q1_bits = (q_prime_2 as f64).log2().ceil() as usize;
        let q2_bits = (q_prime_1 as f64).log2().ceil() as usize;
        let response_bytes_per_output = ((q1_bits + q2_bits) * poly_len + 7) / 8;
        let mask_bytes_per_output = (q1_bits * poly_len + 7) / 8;
        let body_bytes_per_output = (q2_bits * poly_len + 7) / 8;

        let (fwd, fwd_p, inv, inv_p) = flatten_ntt_tables(params);
        let ctx = unsafe {
            sandwich_online_init(
                db.as_ptr(), db_rows, db_rows_padded, db_cols,
                poly_len, params.t_exp_left, num_outputs, max_batch_size,
                params.modulus, q_prime_1, q_prime_2, response_bytes_per_output,
                params.moduli.as_ptr(), params.barrett_cr_1.as_ptr(),
                fwd.as_ptr(), fwd_p.as_ptr(), inv.as_ptr(), inv_p.as_ptr(),
                params.modulus, gpu_tier,
            )
        };
        assert!(!ctx.is_null(), "SandwichPIR online init failed");
        Self {
            ctx, num_outputs, response_bytes_per_output,
            body_bytes_per_output, mask_bytes_per_output,
        }
    }

    pub fn init_packing(
        &self,
        precomp: &mut SwInspirPrecompContext,
        tables: &[u32],
        gen_pows: &[u32],
        num_iter: usize,
    ) {
        let (d_bold_t, d_bold_t_bar, d_bold_t_hat, d_a_hat) = precomp.take_results();
        let d_tables = super::upload_to_gpu(tables);
        let d_gen_pows = super::upload_to_gpu(gen_pows);
        unsafe {
            sandwich_online_init_packing(
                self.ctx, d_bold_t, d_bold_t_bar,
                d_bold_t_hat, d_a_hat,
                d_tables, d_gen_pows, num_iter,
                tables.len(), gen_pows.len(),
            );
        }
        super::free_gpu(d_tables);
        super::free_gpu(d_gen_pows);
    }

    /// Compute A × DB via byte-decomposed GEMM, download to host.
    pub fn compute_hint_gemm(&self, a_rows_w32: &[u64], batch_size: usize) -> Vec<u64> {
        let db_cols = self.num_outputs * 2048;
        let mut result = vec![0u64; db_cols * batch_size];
        unsafe {
            sandwich_online_hint_gemm(
                self.ctx, a_rows_w32.as_ptr(), batch_size, result.as_mut_ptr(),
            );
        }
        result
    }

    /// Compute A × DB via byte-decomposed GEMM, keep on device.
    pub fn compute_hint_gemm_on_device(&self, a_rows_w32: &[u64], batch_size: usize) {
        unsafe { sandwich_online_hint_gemm_on_device(self.ctx, a_rows_w32.as_ptr(), batch_size) }
    }

    /// Take d_intermediate device pointer (nulls it so drop won't free).
    pub fn take_intermediate(&self) -> *mut u64 {
        unsafe { sandwich_online_take_intermediate(self.ctx) }
    }

    /// Run matmul + packing + post-process, downloading body bytes
    /// directly into the body slots of a pre-tiled pinned host response
    /// buffer via `cudaMemcpy3D`. The mask bytes are assumed to have
    /// been pre-filled at offline init time and are NOT touched by this
    /// call. No intermediate host allocation, no post-download scatter.
    ///
    /// # Safety
    /// `response_buf` must be a valid pointer to a pinned host buffer of
    /// at least `batch_size * dst_slice_ysize * dst_row_pitch` bytes.
    /// The caller must ensure exclusive mutable access for the duration
    /// of this call (the write range of this shard is disjoint from
    /// any concurrent shards' write ranges by construction in
    /// `SandwichGpuServer::compute_batch`).
    pub unsafe fn compute_batch_into(
        &self,
        queries: &[u64],
        y_body: &[u64],
        z_body: &[u64],
        batch_size: usize,
        response_buf: *mut u8,
        dst_byte_offset: usize,
        dst_row_pitch: usize,
        dst_slice_ysize: usize,
        copy_height: usize,
        body_bpo: usize,
    ) {
        sandwich_online_compute(
            self.ctx,
            queries.as_ptr(), batch_size,
            y_body.as_ptr(), z_body.as_ptr(),
            response_buf,
            dst_byte_offset,
            dst_row_pitch,
            dst_slice_ysize,
            copy_height,
            body_bpo,
        );
    }

    /// Download the static mask template from this shard's GPU.
    /// Layout: `[num_outputs][mask_bytes_per_output]`. Called once per
    /// shard during offline — the bytes are identical for every query
    /// and persist in the server struct for the whole server lifetime.
    pub fn download_mask_template(&self) -> Vec<u8> {
        let bytes = self.num_outputs * self.mask_bytes_per_output;
        let mut out = vec![0u8; bytes];
        unsafe {
            sandwich_online_download_mask_template(self.ctx, out.as_mut_ptr());
        }
        out
    }
}

#[cfg(feature = "cuda")]
impl Drop for SandwichOnlineContext {
    fn drop(&mut self) {
        unsafe { sandwich_online_free(self.ctx) }
    }
}
