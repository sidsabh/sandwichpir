/// CUDA-accelerated SandwichPIR computation.

pub mod sandwich;

#[cfg(feature = "cuda")]
pub use sandwich::{SandwichOfflineContext, SandwichOnlineContext, SwInspirPrecompContext};

/// Upload a host buffer to GPU, returning a device pointer.
#[cfg(feature = "cuda")]
pub(crate) fn upload_to_gpu<T>(data: &[T]) -> *mut T {
    extern "C" {
        fn cudaMalloc(devPtr: *mut *mut std::ffi::c_void, size: usize) -> i32;
        fn cudaMemcpy(
            dst: *mut std::ffi::c_void,
            src: *const std::ffi::c_void,
            count: usize,
            kind: i32,
        ) -> i32;
    }
    let bytes = data.len() * std::mem::size_of::<T>();
    let mut d_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    unsafe {
        let err = cudaMalloc(&mut d_ptr, bytes);
        assert!(err == 0, "cudaMalloc failed: {}", err);
        let err = cudaMemcpy(
            d_ptr,
            data.as_ptr() as *const std::ffi::c_void,
            bytes,
            1,
        ); // 1 = cudaMemcpyHostToDevice
        assert!(err == 0, "cudaMemcpy failed: {}", err);
    }
    d_ptr as *mut T
}

/// Free GPU memory.
#[cfg(feature = "cuda")]
pub(crate) fn free_gpu<T>(d_ptr: *mut T) {
    extern "C" {
        fn cudaFree(devPtr: *mut std::ffi::c_void) -> i32;
    }
    unsafe {
        cudaFree(d_ptr as *mut std::ffi::c_void);
    }
}

/// Query free GPU memory in bytes.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_free_memory() -> usize {
    extern "C" {
        fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    }
    let (mut free, mut total) = (0usize, 0usize);
    unsafe {
        cudaMemGetInfo(&mut free, &mut total);
    }
    free
}

/// Flatten spiral-rs NTT tables into contiguous arrays for GPU upload.
#[cfg(feature = "cuda")]
pub(crate) fn flatten_ntt_tables(
    params: &spiral_rs::params::Params,
) -> (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>) {
    let poly_len = params.poly_len;
    let crt_count = params.crt_count;
    let mut forward_table = Vec::with_capacity(crt_count * poly_len);
    let mut forward_prime_table = Vec::with_capacity(crt_count * poly_len);
    let mut inverse_table = Vec::with_capacity(crt_count * poly_len);
    let mut inverse_prime_table = Vec::with_capacity(crt_count * poly_len);
    for i in 0..crt_count {
        forward_table.extend_from_slice(&params.ntt_tables[i][0]);
        forward_prime_table.extend_from_slice(&params.ntt_tables[i][1]);
        inverse_table.extend_from_slice(&params.ntt_tables[i][2]);
        inverse_prime_table.extend_from_slice(&params.ntt_tables[i][3]);
    }
    (
        forward_table,
        forward_prime_table,
        inverse_table,
        inverse_prime_table,
    )
}
