/// Test ROM addressing with small ROM to debug
use ashmaize::{Rom, RomGenerationType};
use gpu_ashmaize;

fn main() {
    println!("=== ROM Addressing Test ===\n");
    
    // Create tiny 256-byte ROM for easy debugging
    let rom = Rom::new(b"tiny", RomGenerationType::FullRandom, 256);
    
    // Hash with minimal parameters
    let cpu_hash = ashmaize::hash(b"test", &rom, 2, 256);
    let gpu_hash = gpu_ashmaize::hash(b"test", &rom, 2, 256);
    
    println!("ROM size: {} bytes", rom.len());
    println!("\nCPU hash: {}", hex::encode(&cpu_hash[..16]));
    println!("GPU hash: {}", hex::encode(&gpu_hash[..16]));
    
    if cpu_hash == gpu_hash {
        println!("\n✓✓✓ SUCCESS! Hashes match with tiny ROM!");
    } else {
        println!("\n✗ Still different with tiny ROM");
    }
}
