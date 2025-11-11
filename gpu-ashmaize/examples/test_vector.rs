/**
 * Test GPU implementation against exact Rust test vector
 */

use ashmaize::{Rom, RomGenerationType};

fn main() {
    println!("=== Testing Against Rust test_eq() Vector ===\n");
    
    // Exact same ROM as src/lib.rs test_eq()
    const PRE_SIZE: usize = 16 * 1024;
    const SIZE: usize = 10 * 1024 * 1024;
    const NB_LOOPS: u32 = 8;
    const NB_INSTRS: u32 = 256;
    
    const EXPECTED_CPU: [u8; 64] = [
        56, 148, 1, 228, 59, 96, 211, 173, 9, 98, 68, 61, 89, 171, 124, 171, 124, 183, 200,
        196, 29, 43, 133, 168, 218, 217, 255, 71, 234, 182, 97, 158, 231, 156, 56, 230, 61, 54,
        248, 199, 150, 15, 66, 0, 149, 185, 85, 177, 192, 220, 237, 77, 195, 106, 140, 223,
        175, 93, 238, 220, 57, 159, 180, 243,
    ];
    
    let rom = Rom::new(
        b"123",
        RomGenerationType::TwoStep {
            pre_size: PRE_SIZE,
            mixing_numbers: 4,
        },
        SIZE,
    );
    
    let salt = b"hello";
    
    println!("Test parameters:");
    println!("  ROM: seed=b\"123\", TwoStep{{pre_size={}, mixing=4}}, size={}", PRE_SIZE, SIZE);
    println!("  Salt: {:?}", std::str::from_utf8(salt).unwrap());
    println!("  Params: nb_loops={}, nb_instrs={}\n", NB_LOOPS, NB_INSTRS);
    
    // Compute with CPU

    
    println!("Computing with CPU...");
    let cpu_hash = ashmaize::hash(salt, &rom, NB_LOOPS, NB_INSTRS);
    
    // Verify CPU matches expected
    if cpu_hash == EXPECTED_CPU {
        println!("  ✓ CPU matches expected vector");
    } else {
        println!("  ✗ CPU DOESN'T MATCH EXPECTED!");
        println!("  Expected: {:02x}{:02x}{:02x}{:02x}...", 
                 EXPECTED_CPU[0], EXPECTED_CPU[1], EXPECTED_CPU[2], EXPECTED_CPU[3]);
        println!("  Got:      {:02x}{:02x}{:02x}{:02x}...", 
                 cpu_hash[0], cpu_hash[1], cpu_hash[2], cpu_hash[3]);
    }
    
    // Compute with GPU
    println!("\nComputing with GPU...");
    let gpu_hash = gpu_ashmaize::hash(salt, &rom, NB_LOOPS, NB_INSTRS);
    
    // Compare
    println!("\nResults:");
    println!("  Expected: {:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}...", 
             EXPECTED_CPU[0], EXPECTED_CPU[1], EXPECTED_CPU[2], EXPECTED_CPU[3],
             EXPECTED_CPU[4], EXPECTED_CPU[5], EXPECTED_CPU[6], EXPECTED_CPU[7]);
    println!("  CPU:      {:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}...", 
             cpu_hash[0], cpu_hash[1], cpu_hash[2], cpu_hash[3],
             cpu_hash[4], cpu_hash[5], cpu_hash[6], cpu_hash[7]);
    println!("  GPU:      {:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}...", 
             gpu_hash[0], gpu_hash[1], gpu_hash[2], gpu_hash[3],
             gpu_hash[4], gpu_hash[5], gpu_hash[6], gpu_hash[7]);
    
    println!("\n{}", "=".repeat(60));
    if cpu_hash == EXPECTED_CPU && gpu_hash == EXPECTED_CPU {
        println!("✓✓✓ SUCCESS ✓✓✓");
        println!("Both CPU and GPU match the expected test vector!");
    } else if cpu_hash == gpu_hash {
        println!("⚠ PARTIAL: CPU and GPU match each other but not expected");
    } else if cpu_hash == EXPECTED_CPU {
        println!("✗ GPU INCORRECT: CPU is correct but GPU differs");
    } else {
        println!("✗✗✗ BOTH WRONG: Neither CPU nor GPU match expected!");
    }
    println!("{}", "=".repeat(60));
}
