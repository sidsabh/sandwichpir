#![cfg_attr(target_feature = "avx512f", feature(stdarch_x86_avx512))]

pub mod bits;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod client;
pub mod convolution;
pub mod kernel;
pub mod lwe;
pub mod matmul;
pub mod measurement;
pub mod modulus_switch;
pub mod noise_analysis;
pub mod packing;
pub mod params;
pub mod scheme;
pub mod server;
#[cfg(feature = "cuda")]
pub mod server_gpu;
pub mod transpose;
pub mod util;
