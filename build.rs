use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Compile C++ matmul (AVX kernels for CPU fallback)
    println!("cargo:rerun-if-changed=src/matmul.cpp");
    cc::Build::new()
        .cpp(true)
        .file("src/matmul.cpp")
        .flag("-O3")
        .flag("-march=native")
        .flag("-std=c++11")
        .compile("matmul");

    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        if Path::new("/usr/local/cuda/bin/nvcc").exists()
            || Path::new("/opt/cuda/bin/nvcc").exists()
            || Path::new("/usr/bin/nvcc").exists()
            || Path::new("/opt/apps/cuda/12.2/bin/nvcc").exists()
        {
            compile_cuda();
        } else {
            panic!("CUDA feature enabled but nvcc not found.");
        }
    }
}

fn compile_cuda() {
    println!("cargo:rerun-if-changed=src/cuda/sandwich/offline.cu");
    println!("cargo:rerun-if-changed=src/cuda/sandwich/online.cu");
    println!("cargo:rerun-if-changed=src/cuda/sandwich/tc_packing.cu");
    println!("cargo:rerun-if-changed=src/cuda/sandwich/tc_packing.cuh");
    println!("cargo:rerun-if-changed=src/cuda/sandwich/packing.cuh");
    println!("cargo:rerun-if-changed=src/cuda/sandwich/precomp.cu");
    println!("cargo:rerun-if-changed=src/cuda/common/ntt.cuh");

    let out_dir = env::var("OUT_DIR").unwrap();
    let lib_path = PathBuf::from(&out_dir).join("libsandwichpir_cuda.so");

    let arch = detect_gpu_arch().unwrap_or("sm_61".to_string());
    println!("cargo:warning=Compiling CUDA for {}", arch);

    let cutlass_dir = find_cutlass_dir();
    // The scan accumulates u8*u8 products mod 2^32. Upstream CUTLASS emits
    // saturating (.satfinite) int8 MMA PTX, which silently clamps partial
    // sums once the scan dimension exceeds 2^16 rows, so decode fails on
    // databases >= 4 GB. Refuse to build against an unpatched CUTLASS.
    for header in ["include/cutlass/arch/mma_sm80.h", "include/cutlass/arch/mma_sm75.h"] {
        let path = Path::new(&cutlass_dir).join(header);
        if let Ok(text) = std::fs::read_to_string(&path) {
            if text.contains(".satfinite") {
                panic!(
                    "CUTLASS at {} emits saturating int8 MMA ({}). Apply \
                     rodeo/patches/12-cutlass-no-satfinite.patch (rodeo/setup/setup-gpu.sh \
                     does this automatically): decode fails on databases >= 4 GB otherwise.",
                    cutlass_dir, header
                );
            }
        }
    }
    let cutlass_include = format!("-I{}/include", cutlass_dir);
    let cutlass_tools = format!("-I{}/tools/util/include", cutlass_dir);

    let nvcc_args: Vec<String> = vec![
        "-O3".into(),
        format!("-arch={}", arch),
        "-Xcompiler".into(),
        "-fPIC".into(),
        "-shared".into(),
        "-std=c++17".into(),
        "--expt-relaxed-constexpr".into(),
        "-Isrc/cuda".into(),
        cutlass_include,
        cutlass_tools,
        "-lcublas".into(),
        "src/cuda/sandwich/offline.cu".into(),
        "src/cuda/sandwich/online.cu".into(),
        "src/cuda/sandwich/tc_packing.cu".into(),
        "src/cuda/sandwich/precomp.cu".into(),
        "-o".into(),
        lib_path.to_str().unwrap().into(),
    ];

    let status = Command::new("nvcc")
        .args(&nvcc_args)
        .status()
        .expect("Failed to execute nvcc");

    if !status.success() {
        panic!("CUDA compilation failed");
    }

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=sandwichpir_cuda");

    let cuda_home = env::var("CUDA_HOME").unwrap_or("/opt/apps/cuda/12.2".to_string());
    println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
    // Embed rpath so the binary can find the .so at runtime
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out_dir);

    // Copy .so to target dir so $ORIGIN rpath works for downstream binaries
    if let Ok(target_dir) = env::var("CARGO_TARGET_DIR") {
        let dest = PathBuf::from(&target_dir).join("release").join("libsandwichpir_cuda.so");
        let _ = std::fs::copy(&lib_path, &dest);
    }
    // Also try the default target layout
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let dest = PathBuf::from(&manifest_dir).join("target/release/libsandwichpir_cuda.so");
    let _ = std::fs::copy(&lib_path, &dest);

    if Path::new("/usr/local/cuda/lib64").exists() {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
    if Path::new("/opt/cuda/lib64").exists() {
        println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
    }
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
}

fn find_cutlass_dir() -> String {
    if let Ok(dir) = env::var("CUTLASS_DIR") {
        if Path::new(&dir).join("include/cutlass/cutlass.h").exists() {
            return dir;
        }
    }
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let candidates = [
        PathBuf::from(&manifest_dir).join("../cutlass"),
        env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/tmp"))
            .join("cutlass"),
        PathBuf::from("/usr/local/cutlass"),
    ];
    for p in &candidates {
        if p.join("include/cutlass/cutlass.h").exists() {
            return p.canonicalize().unwrap_or_else(|_| p.clone()).to_string_lossy().into_owned();
        }
    }
    panic!("CUTLASS headers not found. Set CUTLASS_DIR or clone https://github.com/NVIDIA/cutlass");
}

fn detect_gpu_arch() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(&["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if output.status.success() {
        let cap = String::from_utf8_lossy(&output.stdout)
            .trim()
            .lines()
            .next()?
            .replace(".", "");
        Some(format!("sm_{}", cap))
    } else {
        None
    }
}
