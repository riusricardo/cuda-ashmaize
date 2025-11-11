/**
 * Minimal reproduction test
 */

use ashmaize::{Rom, RomGenerationType};

fn main() {
    println!("=== Minimal Test: Exact Vector from lib.rs ===\n");
    
    // Exact parameters from passing test
    const PRE_SIZE: usize = 16 * 1024;
    const SIZE: usize = 10 * 1024 * 1024;
    const NB_LOOPS: u32 = 8;
    const NB_INSTRS: u32 = 256;
    
    const EXPECTED: [u8; 64] = [
        56, 148, 1, 228, 59, 96, 211, 173, 9, 98, 68, 61, 89, 171, 124, 171, 
        124, 183, 200, 196, 29, 43, 133, 168, 218, 217, 255, 71, 234, 182, 97, 158, 
        231, 156, 56, 230, 61, 54, 248, 199, 150, 15, 66, 0, 149, 185, 85, 177, 
        192, 220, 237, 77, 195, 106, 140, 223, 175, 93, 238, 220, 57, 159, 180, 243,
    ];
    
    println!("Creating ROM...");
    let rom = Rom::new(
        b"123",
        RomGenerationType::TwoStep {
            pre_size: PRE_SIZE,
            mixing_numbers: 4,
        },
        SIZE,
    );
    println!("✓ ROM created: {} bytes\n", SIZE);
    
    let salt = b"hello";
    
    println!("Computing CPU hash...");
    let cpu_hash = ashmaize::hash(salt, &rom, NB_LOOPS, NB_INSTRS);
    
    print!("CPU result:      ");
    for i in 0..16 {
        print!("{:02x}", cpu_hash[i]);
    }
    println!("...");
    
    print!("Expected result: ");
    for i in 0..16 {
        print!("{:02x}", EXPECTED[i]);
    }
    println!("...");
    
    if cpu_hash == EXPECTED {
        println!("✓ CPU matches expected\n");
    } else {
        println!("✗ CPU DOES NOT match expected!\n");
        return;
    }
    
    println!("Computing GPU hash...");
    let gpu_hash = gpu_ashmaize::hash(salt, &rom, NB_LOOPS, NB_INSTRS);
    
    print!("GPU result:      ");
    for i in 0..16 {
        print!("{:02x}", gpu_hash[i]);
    }
    println!("...");
    
    if gpu_hash == EXPECTED {
        println!("✓ GPU matches expected");
        println!("✓✓✓ SUCCESS - GPU WORKS!");
    } else {
        println!("✗ GPU does NOT match expected");
        
        if cpu_hash == gpu_hash {
            println!("⚠ BUT: CPU and GPU match each other (both wrong?)");
        } else {
            println!("✗ CPU and GPU also differ from each other");
        }
    }
    
    println!("\nFull hashes:");
    println!("Expected:");
    for i in 0..64 {
        print!("{:02x}", EXPECTED[i]);
        if (i+1) % 16 == 0 { println!(); }
    }
    
    println!("\nCPU:");
    for i in 0..64 {
        print!("{:02x}", cpu_hash[i]);
        if (i+1) % 16 == 0 { println!(); }
    }
    
    println!("\nGPU:");
    for i in 0..64 {
        print!("{:02x}", gpu_hash[i]);
        if (i+1) % 16 == 0 { println!(); }
    }
}
