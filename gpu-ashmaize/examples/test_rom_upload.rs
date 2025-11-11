/// Test to verify ROM upload and reads
use ashmaize::{Rom, RomGenerationType};
use gpu_ashmaize;

fn main() {
    println!("=== ROM Upload Test ===\n");
    
    let rom = Rom::new(b"test", RomGenerationType::FullRandom, 1024);
    
    // Print first 32 bytes of ROM
    let rom_ptr = rom.as_ptr();
    let rom_bytes: &[u8] = unsafe { std::slice::from_raw_parts(rom_ptr, 32) };
    
    println!("CPU ROM first 32 bytes:");
    for chunk in rom_bytes.chunks(8) {
        for b in chunk {
            print!("{:02x} ", b);
        }
        println!();
    }
    
    // Now let GPU hash it - if ROM is uploaded wrong, we'll see different results
    let hash = gpu_ashmaize::hash(b"test", &rom, 2, 256);
    println!("\nGPU hash: {}", hex::encode(&hash[..16]));
}
