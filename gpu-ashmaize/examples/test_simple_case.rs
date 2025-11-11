use ashmaize::{hash as cpu_hash, Rom, RomGenerationType};
use gpu_ashmaize;

fn main() {
    println!("=== Simple Test: Minimal Parameters ===\n");
    
    // Create simplest possible ROM
    println!("Creating simple ROM...");
    let rom = Rom::new(b"test", RomGenerationType::FullRandom, 1024);
    println!("✓ ROM created\n");
    
    // Use simple parameters
    let salt = b"salt";
    let nb_loops = 2;  // Minimum allowed
    let nb_instrs = 256;  // Minimum allowed
    
    // Test CPU
    println!("Computing CPU hash...");
    let cpu_result = cpu_hash(salt, &rom, nb_loops, nb_instrs);
    println!("CPU result: {}", hex::encode(&cpu_result[..16]));
    
    // Test GPU
    println!("\nComputing GPU hash...");
    let gpu_result = gpu_ashmaize::hash(salt, &rom, nb_loops, nb_instrs);
    println!("GPU result: {}", hex::encode(&gpu_result[..16]));
    
    // Compare
    println!("\n=== Comparison ===");
    if cpu_result == gpu_result {
        println!("✓ PASS: CPU and GPU match!");
        println!("\nFull hash:");
        for chunk in cpu_result.chunks(16) {
            println!("{}", hex::encode(chunk));
        }
    } else {
        println!("✗ FAIL: CPU and GPU differ");
        println!("\nCPU hash:");
        for chunk in cpu_result.chunks(16) {
            println!("{}", hex::encode(chunk));
        }
        println!("\nGPU hash:");
        for chunk in gpu_result.chunks(16) {
            println!("{}", hex::encode(chunk));
        }
    }
}
