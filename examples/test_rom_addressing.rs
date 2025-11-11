// Test to verify CPU ROM addressing behavior
use ashmaize::{Rom, RomGenerationType};

fn main() {
    println!("Testing CPU ROM addressing behavior\n");
    
    // Create a small ROM for testing
    let seed = [0x42u8; 32];
    let rom = Rom::new(RomGenerationType::Instant, seed);
    
    println!("ROM size: {} bytes", rom.len());
    println!("ROM blocks: {}", rom.len() / 64);
    println!();
    
    // Test a few addresses to see what byte offsets they access
    for addr in [0, 1, 2, 100, 1000, 4095, 4096, 5000, 10000, 100000] {
        let data = rom.at(addr);
        
        // Print first 8 bytes to identify the data
        print!("ROM addr {} -> data: ", addr);
        for i in 0..8 {
            print!("{:02x} ", data[i]);
        }
        println!();
    }
    
    println!("\nAnalysis:");
    println!("If addr X and addr Y return same first bytes, they access the same region.");
    println!("This will reveal if addressing wraps at 4096 blocks or actual ROM size.");
}
