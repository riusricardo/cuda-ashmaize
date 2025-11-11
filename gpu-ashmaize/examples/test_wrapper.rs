/**
 * Compare GpuMiner.hash() vs gpu_ashmaize::hash()
 */

use ashmaize::{Rom, RomGenerationType};
use gpu_ashmaize::GpuMiner;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing GpuMiner vs Free Function ===\n");
    
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
    
    // Method 1: Free function
    println!("Method 1: gpu_ashmaize::hash()");
    let hash1 = gpu_ashmaize::hash(salt, &rom, nb_loops, nb_instrs);
    print!("  Result: ");
    for i in 0..16 {
        print!("{:02x}", hash1[i]);
    }
    println!("...\n");
    
    // Method 2: GpuMiner instance
    println!("Method 2: GpuMiner::with_params() + upload_rom() + hash()");
    let mut miner = GpuMiner::with_params(nb_loops, nb_instrs)?;
    miner.upload_rom(&rom)?;
    let hash2 = miner.hash(salt)?;
    print!("  Result: ");
    for i in 0..16 {
        print!("{:02x}", hash2[i]);
    }
    println!("...\n");
    
    // Compare
    if hash1 == hash2 {
        println!("✓ Both methods produce same result");
        println!("  (Both are wrong, but at least consistent)");
    } else {
        println!("✗ Methods produce DIFFERENT results!");
        println!("  This indicates wrapper/state management bug");
    }
    
    Ok(())
}
