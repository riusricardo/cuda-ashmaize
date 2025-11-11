/**
 * Check GPU determinism - does it always produce the same (wrong) hash?
 */

use ashmaize::{Rom, RomGenerationType};

fn main() {
    println!("=== GPU Determinism Test ===\n");
    
    let rom = Rom::new(
        b"123",
        RomGenerationType::TwoStep {
            pre_size: 16 * 1024,
            mixing_numbers: 4,
        },
        10 * 1024 * 1024,
    );
    
    let salt = b"hello";
    let nb_loops = 8;
    let nb_instrs = 256;
    
    println!("Running GPU hash 5 times...\n");
    
    let mut hashes = Vec::new();
    for i in 1..=5 {
        let hash = gpu_ashmaize::hash(salt, &rom, nb_loops, nb_instrs);
        print!("Run {}: ", i);
        for j in 0..16 {
            print!("{:02x}", hash[j]);
        }
        println!("...");
        hashes.push(hash);
    }
    
    println!();
    
    // Check if all are identical
    let first = &hashes[0];
    let all_same = hashes.iter().all(|h| h == first);
    
    if all_same {
        println!("✓ GPU is DETERMINISTIC (all 5 runs match)");
        println!("  Problem: produces wrong but consistent result");
    } else {
        println!("✗ GPU is NON-DETERMINISTIC (runs differ!)");
        println!("  Problem: random/race condition bug");
    }
}
