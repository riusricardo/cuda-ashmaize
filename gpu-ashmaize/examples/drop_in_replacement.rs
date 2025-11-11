/**
 * Example: Drop-in GPU replacement for CPU implementation
 * 
 * Shows how to switch between CPU and GPU with minimal code changes
 */

use ashmaize::{Rom, RomGenerationType};

fn main() {
    println!("=== Drop-in GPU Replacement Demo ===\n");
    
    // Create ROM (same for both CPU and GPU)
    let rom = Rom::new(
        b"test_seed",
        RomGenerationType::TwoStep {
            pre_size: 16 * 1024,
            mixing_numbers: 4,
        },
        1 * 1024 * 1024,
    );
    
    // Parameters
    let nb_loops = 8;
    let nb_instrs = 256;
    let salt = b"test_salt";
    
    // Compute with CPU
    println!("Computing with CPU...");
    let cpu_hash = ashmaize::hash(salt, &rom, nb_loops, nb_instrs);
    print!("  CPU result: ");
    for byte in &cpu_hash[..16] {
        print!("{:02x}", byte);
    }
    println!("...");
    
    // Compute with GPU - IDENTICAL API!
    println!("\nComputing with GPU...");
    let gpu_hash = gpu_ashmaize::hash(salt, &rom, nb_loops, nb_instrs);
    print!("  GPU result: ");
    for byte in &gpu_hash[..16] {
        print!("{:02x}", byte);
    }
    println!("...");
    
    // Verify they match
    println!();
    if cpu_hash == gpu_hash {
        println!("✓ SUCCESS: CPU and GPU produce identical results!");
    } else {
        println!("✗ ERROR: Results differ!");
        std::process::exit(1);
    }
    
    println!("\nAPI Compatibility:");
    println!("  CPU: ashmaize::hash(salt, rom, nb_loops, nb_instrs) -> [u8; 64]");
    println!("  GPU: gpu_ashmaize::hash(salt, rom, nb_loops, nb_instrs) -> [u8; 64]");
    println!("\nThe function signatures are IDENTICAL!");
    println!("Switch by changing one line:");
    println!("  - use ashmaize::hash;");
    println!("  + use gpu_ashmaize::hash;");
}
