use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/");
    
    // Check for CUDA toolkit
    let nvcc_path = which::which("nvcc").expect(
        "nvcc not found in PATH. Please install CUDA Toolkit 12.0+ from https://developer.nvidia.com/cuda-downloads"
    );
    
    println!("cargo:warning=Found nvcc at: {}", nvcc_path.display());
    
    // Detect CUDA version
    let nvcc_version = Command::new("nvcc")
        .arg("--version")
        .output()
        .expect("Failed to get nvcc version");
    
    let version_str = String::from_utf8_lossy(&nvcc_version.stdout);
    println!("cargo:warning=CUDA version: {}", version_str.lines().last().unwrap_or("unknown"));
    
    // Detect compute capability from GPU (if available)
    let compute_arch = detect_gpu_compute_capability().unwrap_or_else(|| {
        println!("cargo:warning=Could not detect GPU, using default arch sm_75 (Turing)");
        "sm_75".to_string()
    });
    
    println!("cargo:warning=Compiling for compute capability: {}", compute_arch);
    
    let cuda_dir = PathBuf::from("cuda");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    // CUDA source files
    let cuda_sources = vec![
        "blake2b.cu",
        "argon2.cu",
        "vm.cu",
        "instructions.cu",
        "kernel.cu",
    ];
    
    // Compile each CUDA source to object file
    let mut object_files = Vec::new();
    
    for source in &cuda_sources {
        let source_path = cuda_dir.join(source);
        let obj_name = source.replace(".cu", ".o");
        let obj_path = out_dir.join(&obj_name);
        
        println!("cargo:warning=Compiling CUDA: {}", source);
        
        let status = Command::new("nvcc")
            .args(&[
                // Compute capability
                &format!("-arch={}", compute_arch),
                
                // C++ standard
                "-std=c++17",
                
                // Optimization
                "-O3",
                "-use_fast_math",
                
                // Debugging info for profiling
                "-lineinfo",
                "-g",
                
                // Generate device code
                "-dc",  // Separate compilation
                
                // Compiler options
                "-Xcompiler", "-fPIC",
                "-Xcompiler", "-Wall",
                
                // PTX optimization
                "--ptxas-options=-v",
                "--ptxas-options=-O3",
                
                // Output
                "-o", obj_path.to_str().unwrap(),
                
                // Input
                source_path.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to execute nvcc");
        
        if !status.success() {
            panic!("nvcc compilation failed for {}", source);
        }
        
        object_files.push(obj_path);
    }
    
    // Create device link object
    let dlink_obj = out_dir.join("dlink.o");
    
    println!("cargo:warning=Creating device link object: {}", dlink_obj.display());
    
    let status = Command::new("nvcc")
        .args(&[
            &format!("-arch={}", compute_arch),
            "-dlink",
            "-Xcompiler", "-fPIC",
            "-o", dlink_obj.to_str().unwrap(),
        ])
        .args(object_files.iter().map(|p| p.to_str().unwrap()))
        .status()
        .expect("Failed to create device link object");
    
    if !status.success() {
        panic!("nvcc device linking failed");
    }
    
    // Tell cargo where to find objects - we'll link them individually
    for obj in &object_files {
        println!("cargo:rustc-link-arg={}", obj.display());
    }
    println!("cargo:rustc-link-arg={}", dlink_obj.display());
    
    // Link CUDA runtime libraries
    println!("cargo:rustc-link-lib=static=cudadevrt");  // Device runtime for separate compilation  
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
    
    // Link CUDA runtime from system
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-search=native={}/targets/x86_64-linux/lib", cuda_path);
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-search=native=/usr/local/cuda/targets/x86_64-linux/lib");
    }
}

fn detect_gpu_compute_capability() -> Option<String> {
    // Try to query GPU using nvidia-smi
    let output = Command::new("nvidia-smi")
        .args(&["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    
    if output.status.success() {
        let cap = String::from_utf8_lossy(&output.stdout)
            .trim()
            .lines()
            .next()?
            .to_string();
        
        // Convert "8.9" to "sm_89"
        let cleaned = cap.replace(".", "");
        
        // Cap at maximum supported architecture for CUDA 13.0
        // CUDA 13.0 supports up to sm_90 (Hopper)
        let arch_num: u32 = cleaned.parse().ok()?;
        let capped_arch = if arch_num > 90 {
            println!("cargo:warning=GPU reports compute capability {}, capping at sm_90 for CUDA 13.0 compatibility", cap);
            "sm_90".to_string()
        } else {
            format!("sm_{}", cleaned)
        };
        
        return Some(capped_arch);
    }
    
    None
}
