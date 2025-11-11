/**
 * Systematic GPU vs CPU Debug Test
 * 
 * Tests progressively to find where GPU/CPU diverge
 */

use ashmaize::{Rom, RomGenerationType};

fn print_hash(label: &str, hash: &[u8; 64]) {
    print!("{}: ", label);
    for i in 0..8 {
        print!("{:02x}", hash[i]);
    }
    println!("...");
}

fn test_rom_digest_match(rom: &Rom) -> bool {
    println!("\n=== Test 1: ROM Digest Comparison ===");
    
    unsafe {
        let digest_ptr = rom.digest_as_ptr();
        let digest = std::slice::from_raw_parts(digest_ptr, 64);
        
        print!("ROM digest (first 32): ");
        for i in 0..32 {
            print!("{:02x}", digest[i]);
        }
        println!();
    }
    
    println!("✓ ROM created successfully");
    true
}

fn test_same_rom_different_salts(rom: &Rom, nb_loops: u32, nb_instrs: u32) {
    println!("\n=== Test 2: Same ROM, Different Salts ===");
    
    let salts: Vec<&[u8]> = vec![b"a", b"b", b"test", b"hello"];
    
    for salt in salts {
        let cpu_hash = ashmaize::hash(salt, rom, nb_loops, nb_instrs);
        let gpu_hash = gpu_ashmaize::hash(salt, rom, nb_loops, nb_instrs);
        
        let matches = cpu_hash == gpu_hash;
        let status = if matches { "✓" } else { "✗" };
        
        println!("{} Salt '{}':", status, String::from_utf8_lossy(salt));
        if !matches {
            print_hash("  CPU", &cpu_hash);
            print_hash("  GPU", &gpu_hash);
        } else {
            print_hash("  Both", &cpu_hash);
        }
    }
}

fn test_same_salt_different_params(rom: &Rom) {
    println!("\n=== Test 3: Same Salt, Different Parameters ===");
    
    let salt = b"test";
    let params = vec![
        (2, 256),
        (4, 256),
        (8, 256),
        (8, 512),
    ];
    
    for (nb_loops, nb_instrs) in params {
        let cpu_hash = ashmaize::hash(salt, rom, nb_loops, nb_instrs);
        let gpu_hash = gpu_ashmaize::hash(salt, rom, nb_loops, nb_instrs);
        
        let matches = cpu_hash == gpu_hash;
        let status = if matches { "✓" } else { "✗" };
        
        println!("{} loops={}, instrs={}:", status, nb_loops, nb_instrs);
        if !matches {
            print_hash("  CPU", &cpu_hash);
            print_hash("  GPU", &gpu_hash);
        }
    }
}

fn test_different_rom_sizes() {
    println!("\n=== Test 4: Different ROM Sizes ===");
    
    let salt = b"test";
    let nb_loops = 8;
    let nb_instrs = 256;
    
    let sizes = vec![
        64 * 1024,      // 64 KB
        256 * 1024,     // 256 KB
        1024 * 1024,    // 1 MB
        10 * 1024 * 1024, // 10 MB
    ];
    
    for size in sizes {
        let rom = Rom::new(
            b"test_seed",
            RomGenerationType::TwoStep {
                pre_size: 16 * 1024,
                mixing_numbers: 4,
            },
            size,
        );
        
        let cpu_hash = ashmaize::hash(salt, &rom, nb_loops, nb_instrs);
        let gpu_hash = gpu_ashmaize::hash(salt, &rom, nb_loops, nb_instrs);
        
        let matches = cpu_hash == gpu_hash;
        let status = if matches { "✓" } else { "✗" };
        
        println!("{} ROM size: {} bytes:", status, size);
        if !matches {
            print_hash("  CPU", &cpu_hash);
            print_hash("  GPU", &gpu_hash);
        }
    }
}

fn test_exact_vector() {
    println!("\n=== Test 5: Exact Test Vector from lib.rs ===");
    
    // Exact same as src/lib.rs test_eq()
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
    
    let rom = Rom::new(
        b"123",
        RomGenerationType::TwoStep {
            pre_size: PRE_SIZE,
            mixing_numbers: 4,
        },
        SIZE,
    );
    
    let salt = b"hello";
    
    let cpu_hash = ashmaize::hash(salt, &rom, NB_LOOPS, NB_INSTRS);
    let gpu_hash = gpu_ashmaize::hash(salt, &rom, NB_LOOPS, NB_INSTRS);
    
    let cpu_matches = cpu_hash == EXPECTED;
    let gpu_matches = gpu_hash == EXPECTED;
    let both_match = cpu_hash == gpu_hash;
    
    println!("Expected vector:");
    print_hash("  Expected", &EXPECTED);
    print_hash("  CPU", &cpu_hash);
    print_hash("  GPU", &gpu_hash);
    
    if cpu_matches && gpu_matches {
        println!("✓✓✓ BOTH MATCH EXPECTED!");
    } else if cpu_matches {
        println!("✓ CPU matches expected");
        println!("✗ GPU does NOT match expected");
    } else if gpu_matches {
        println!("✗ CPU does NOT match expected");
        println!("✓ GPU matches expected");
    } else if both_match {
        println!("⚠ CPU and GPU match each other, but BOTH differ from expected");
    } else {
        println!("✗✗✗ ALL THREE DIFFER!");
    }
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Systematic GPU vs CPU Debugging                          ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    
    // Create a standard ROM for most tests
    let rom = Rom::new(
        b"debug_seed",
        RomGenerationType::TwoStep {
            pre_size: 16 * 1024,
            mixing_numbers: 4,
        },
        1 * 1024 * 1024, // 1 MB
    );
    
    test_rom_digest_match(&rom);
    test_same_rom_different_salts(&rom, 8, 256);
    test_same_salt_different_params(&rom);
    test_different_rom_sizes();
    test_exact_vector();
    
    println!("\n{}", "═".repeat(60));
    println!("Debug complete. Analyze patterns above to find divergence.");
}
