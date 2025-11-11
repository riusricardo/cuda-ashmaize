/**
 * Verify GPU and CPU implementations produce identical results
 */

use ashmaize::{Rom, RomGenerationType};

fn main() {
    println!("=== API Compatibility Test: 1:1 Drop-in Replacement ===\n");
    
    // Create ROM
    let rom = Rom::new(
        b"test_seed",
        RomGenerationType::TwoStep {
            pre_size: 16 * 1024,
            mixing_numbers: 4,
        },
        1 * 1024 * 1024,
    );
    
    // Debug: Print ROM info
    println!("ROM info:");
    println!("  Size: {} bytes", rom.len());
    unsafe {
        let digest_ptr = rom.digest_as_ptr();
        let digest = std::slice::from_raw_parts(digest_ptr, 16);
        print!("  Digest (first 16): ");
        for &b in digest {
            print!("{:02x}", b);
        }
        println!();
        
        let data_ptr = rom.as_ptr();
        let data = std::slice::from_raw_parts(data_ptr, 16);
        print!("  Data (first 16):   ");
        for &b in data {
            print!("{:02x}", b);
        }
        println!("\n");
    }
    
    let nb_loops = 8;
    let nb_instrs = 256;
    
    // Test 1: Verify EXACT same function signature
    println!("Test 1: API Signature Compatibility");
    println!("  CPU: ashmaize::hash(salt, rom, nb_loops, nb_instrs) -> [u8; 64]");
    println!("  GPU: gpu_ashmaize::hash(salt, rom, nb_loops, nb_instrs) -> [u8; 64]");
    println!("  ✓ Signatures are IDENTICAL - no Result wrapper!");
    
    // Test 2: Multiple salts with exact same calling convention
    println!("\nTest 2: Result Compatibility");
    let test_salts = vec![
        b"hello".to_vec(),
        b"world".to_vec(),
        b"test123".to_vec(),
        b"ashmaize".to_vec(),
    ];
    
    println!("  Testing {} different salts...\n", test_salts.len());
    
    let mut all_match = true;
    
    for (i, salt) in test_salts.iter().enumerate() {
        // EXACT SAME CALL - returns [u8; 64] directly, no unwrap needed
        let cpu_hash: [u8; 64] = ashmaize::hash(salt, &rom, nb_loops, nb_instrs);
        let gpu_hash: [u8; 64] = gpu_ashmaize::hash(salt, &rom, nb_loops, nb_instrs);
        
        // Compare
        let matches = cpu_hash == gpu_hash;
        all_match = all_match && matches;
        
        let status = if matches { "✓" } else { "✗" };
        let salt_str = String::from_utf8_lossy(salt);
        println!("  {} Salt {}: '{:10}' - {}", 
                 status, i + 1, salt_str,
                 if matches { "MATCH" } else { "MISMATCH!" });
        
        if !matches {
            println!("    CPU: {:02x}{:02x}{:02x}{:02x}...", 
                     cpu_hash[0], cpu_hash[1], cpu_hash[2], cpu_hash[3]);
            println!("    GPU: {:02x}{:02x}{:02x}{:02x}...", 
                     gpu_hash[0], gpu_hash[1], gpu_hash[2], gpu_hash[3]);
        }
    }
    
    println!("\n{}", "═".repeat(60));
    if all_match {
        println!("✓✓✓ ALL TESTS PASSED ✓✓✓");
        println!("\nGPU implementation is a PERFECT 1:1 drop-in replacement!");
        println!("You can switch by changing a single line:");
        println!("  - use ashmaize::hash;      // CPU");
        println!("  + use gpu_ashmaize::hash;  // GPU");
        println!("{}", "═".repeat(60));
    } else {
        println!("✗✗✗ TESTS FAILED ✗✗✗");
        println!("GPU and CPU produce different results!");
        println!("{}", "═".repeat(60));
        std::process::exit(1);
    }
}
