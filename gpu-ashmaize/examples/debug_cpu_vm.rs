/// CPU-side test to print VM initial state for comparison with GPU
use ashmaize::{Rom, RomGenerationType};

fn main() {
    println!("=== CPU VM State Debug ===\n");
    
    let rom = Rom::new(b"test", RomGenerationType::FullRandom, 1024);
    let salt = b"hello";
    let nb_instrs = 256;
    
    // We need to access VM internals - let's just call hash and see the result
    // We'll need to modify the CPU code to add debug output too
    
    println!("Running CPU hash with debug output...");
    println!("(Need to add debug output to CPU VM::new)");
    
    let hash = ashmaize::hash(salt, &rom, 2, nb_instrs);
    println!("\nCPU hash: {}", hex::encode(&hash[..32]));
}
