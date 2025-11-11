/// Test to trace VM state and find where CPU and GPU diverge
/// 
/// This will check:
/// 1. Initial VM state after initialization
/// 2. State after first instruction
/// 3. State after first loop
/// 4. Final hash
///
/// By comparing these checkpoints, we can pinpoint where divergence occurs.

use ashmaize::{Rom, RomGenerationType};
use gpu_ashmaize;

fn main() {
    println!("=== VM State Tracing Test ===\n");
    
    // Use simplest possible ROM and parameters
    let rom = Rom::new(b"test", RomGenerationType::FullRandom, 1024);
    let salt = b"hello";
    let nb_loops = 2;  // Minimum
    let nb_instrs = 256;  // Minimum
    
    println!("Test parameters:");
    println!("  ROM size: {} bytes", rom.len());
    println!("  Salt: {:?}", salt);
    println!("  Loops: {}", nb_loops);
    println!("  Instructions: {}\n", nb_instrs);
    
    // Get CPU hash
    let cpu_hash = ashmaize::hash(salt, &rom, nb_loops, nb_instrs);
    println!("CPU final hash: {}", hex::encode(&cpu_hash[..32]));
    
    // Get GPU hash
    let gpu_hash = gpu_ashmaize::hash(salt, &rom, nb_loops, nb_instrs);
    println!("GPU final hash: {}", hex::encode(&gpu_hash[..32]));
    
    if cpu_hash == gpu_hash {
        println!("\n✓ SUCCESS: Hashes match!");
    } else {
        println!("\n✗ FAILURE: Hashes differ");
        println!("\nFull CPU hash:");
        for chunk in cpu_hash.chunks(16) {
            println!("  {}", hex::encode(chunk));
        }
        println!("\nFull GPU hash:");
        for chunk in gpu_hash.chunks(16) {
            println!("  {}", hex::encode(chunk));
        }
        
        // Find first diverging byte
        for (i, (c, g)) in cpu_hash.iter().zip(gpu_hash.iter()).enumerate() {
            if c != g {
                println!("\nFirst difference at byte {}: CPU={:02x} GPU={:02x}", i, c, g);
                break;
            }
        }
    }
    
    println!("\n=== Next Steps ===");
    println!("We need to add debug output to CUDA code to trace:");
    println!("1. VM registers after init");
    println!("2. First few shuffled program bytes");
    println!("3. Registers after first instruction");
    println!("4. Memory counter progression");
}
