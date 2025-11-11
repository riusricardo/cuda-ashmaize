/**
 * Large ROM Test - High Level VM Testing
 * 
 * Tests GPU with very large ROMs (256MB, 512MB, 1GB) at the full VM level
 * to ensure no memory access issues occur with realistic large datasets.
 */

use ashmaize::{Rom, RomGenerationType};
use std::time::Instant;

const KB: usize = 1024;
const MB: usize = 1024 * KB;

fn test_rom_size(size: usize, size_label: &str) {
    println!("\n=== Testing {} ROM ({} bytes) ===", size_label, size);
    
    let start = Instant::now();
    let rom = Rom::new(
        b"large_rom_test_seed",
        RomGenerationType::TwoStep {
            pre_size: 16 * MB,
            mixing_numbers: 4,
        },
        size,
    );
    let rom_gen_time = start.elapsed();
    println!("ROM generation: {:.2?}", rom_gen_time);
    
    let salt = b"test_large_rom";
    let nb_loops = 8;
    let nb_instrs = 256;
    
    // CPU hash
    println!("Computing CPU hash...");
    let cpu_start = Instant::now();
    let cpu_hash = ashmaize::hash(salt, &rom, nb_loops, nb_instrs);
    let cpu_time = cpu_start.elapsed();
    println!("CPU time: {:.2?}", cpu_time);
    
    // GPU hash
    println!("Computing GPU hash...");
    let gpu_start = Instant::now();
    let gpu_hash = gpu_ashmaize::hash(salt, &rom, nb_loops, nb_instrs);
    let gpu_time = gpu_start.elapsed();
    println!("GPU time: {:.2?}", gpu_time);
    
    // Compare
    let matches = cpu_hash == gpu_hash;
    if matches {
        println!("✓✓✓ SUCCESS - Hashes match!");
        print!("Hash: ");
        for i in 0..16 {
            print!("{:02x}", cpu_hash[i]);
        }
        println!("...");
        println!("CPU: {:.2?} | GPU: {:.2?} | Speedup: {:.2}x", 
                 cpu_time, gpu_time, 
                 cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
    } else {
        println!("✗✗✗ FAILED - Hashes differ!");
        print!("CPU: ");
        for i in 0..16 {
            print!("{:02x}", cpu_hash[i]);
        }
        println!("...");
        print!("GPU: ");
        for i in 0..16 {
            print!("{:02x}", gpu_hash[i]);
        }
        println!("...");
        panic!("Hash mismatch for {} ROM!", size_label);
    }
}

fn test_stress_with_large_rom() {
    println!("\n=== Stress Test: Multiple Hashes with 1GB ROM ===");
    
    let size = 1024 * MB;
    println!("Generating 1GB ROM...");
    let start = Instant::now();
    let rom = Rom::new(
        b"stress_test_seed",
        RomGenerationType::TwoStep {
            pre_size: 16 * MB,
            mixing_numbers: 4,
        },
        size,
    );
    println!("ROM generation: {:.2?}", start.elapsed());
    
    let salts = vec![b"test1", b"test2", b"test3", b"test4", b"test5"];
    let nb_loops = 8;
    let nb_instrs = 256;
    
    println!("\nHashing {} different salts...", salts.len());
    for (i, salt) in salts.iter().enumerate() {
        print!("Salt {}: ", i + 1);
        let cpu_hash = ashmaize::hash(*salt, &rom, nb_loops, nb_instrs);
        let gpu_hash = gpu_ashmaize::hash(*salt, &rom, nb_loops, nb_instrs);
        
        if cpu_hash == gpu_hash {
            println!("✓ Match");
        } else {
            println!("✗ MISMATCH!");
            panic!("Hash mismatch on salt {}", i + 1);
        }
    }
    
    println!("✓✓✓ All stress tests passed!");
}

fn test_varying_parameters_large_rom() {
    println!("\n=== Varying Parameters with 512MB ROM ===");
    
    let size = 512 * MB;
    println!("Generating 512MB ROM...");
    let start = Instant::now();
    let rom = Rom::new(
        b"param_test_seed",
        RomGenerationType::TwoStep {
            pre_size: 16 * MB,
            mixing_numbers: 4,
        },
        size,
    );
    println!("ROM generation: {:.2?}", start.elapsed());
    
    let salt = b"param_test";
    let test_cases = vec![
        (4, 256, "4 loops, 256 instrs"),
        (8, 256, "8 loops, 256 instrs"),
        (8, 512, "8 loops, 512 instrs"),
        (16, 256, "16 loops, 256 instrs"),
    ];
    
    for (nb_loops, nb_instrs, label) in test_cases {
        print!("Testing {}: ", label);
        let cpu_hash = ashmaize::hash(salt, &rom, nb_loops, nb_instrs);
        let gpu_hash = gpu_ashmaize::hash(salt, &rom, nb_loops, nb_instrs);
        
        if cpu_hash == gpu_hash {
            println!("✓");
        } else {
            println!("✗ MISMATCH!");
            panic!("Hash mismatch for {}", label);
        }
    }
    
    println!("✓✓✓ All parameter variations passed!");
}

fn main() {
    println!("╔════════════════════════════════════════════════════╗");
    println!("║     Large ROM High-Level VM Testing Suite         ║");
    println!("╚════════════════════════════════════════════════════╝");
    println!();
    println!("This test verifies GPU correctness with very large ROMs");
    println!("at the full VM level (not just low-level primitives).");
    println!();
    
    // Test increasing ROM sizes
    test_rom_size(256 * MB, "256MB");
    test_rom_size(512 * MB, "512MB");
    test_rom_size(1024 * MB, "1GB");
    
    // Stress test with multiple operations
    test_stress_with_large_rom();
    
    // Test different parameters
    test_varying_parameters_large_rom();
    
    println!("\n╔════════════════════════════════════════════════════╗");
    println!("║     ✓✓✓ ALL LARGE ROM TESTS PASSED! ✓✓✓          ║");
    println!("╚════════════════════════════════════════════════════╝");
}
