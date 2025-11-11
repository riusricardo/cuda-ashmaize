//! Low-level CPU vs GPU primitive tests
//! 
//! These tests verify that Blake2b and Argon2H' produce identical results
//! on CPU (using cryptoxide) and GPU (using our CUDA implementation)

use cryptoxide::hashing::blake2b::Blake2b;
use cryptoxide::kdf::argon2;

// Test helper to call GPU Blake2b via FFI
// For now we'll use the hash function as a proxy since we don't expose Blake2b directly
// We'll need to add FFI wrappers for direct Blake2b and Argon2 testing

#[test]
fn test_argon2_hprime_empty_input() {
    // Test Argon2H' with empty input
    let input = b"";
    let mut cpu_output = vec![0u8; 64];
    argon2::hprime(&mut cpu_output, input);
    
    println!("CPU Argon2H'(empty, 64): {}", hex::encode(&cpu_output));
    
    // Expected from CUDA test: 07e03f4501d50861a5172eee28398042e23dface4bd90edde04b754276042c75
    let expected = hex::decode("07e03f4501d50861a5172eee28398042e23dface4bd90edde04b754276042c75").unwrap();
    assert_eq!(&cpu_output[..32], &expected[..], "CPU Argon2H' should match expected value");
}

#[test]
fn test_argon2_hprime_small_input() {
    // Test with "hello"
    let input = b"hello";
    let mut cpu_output = vec![0u8; 64];
    argon2::hprime(&mut cpu_output, input);
    
    println!("CPU Argon2H'('hello', 64): {}", hex::encode(&cpu_output[..32]));
    
    // Expected from CUDA test: f62ff338765b14ac81d35b2aa70147dbc36e71d24dd3825f7d0165e2f664c549
    let expected = hex::decode("f62ff338765b14ac81d35b2aa70147dbc36e71d24dd3825f7d0165e2f664c549").unwrap();
    assert_eq!(&cpu_output[..32], &expected[..], "CPU Argon2H' should match CUDA result");
}

#[test]
fn test_argon2_hprime_abc_64bytes() {
    // Test with "abc" producing 64 bytes
    let input = b"abc";
    let mut cpu_output = vec![0u8; 64];
    argon2::hprime(&mut cpu_output, input);
    
    println!("CPU Argon2H'('abc', 64): {}", hex::encode(&cpu_output));
    
    // Expected from CUDA manual verification test
    let expected = hex::decode(
        "f32577a3172f56657d531faaa43077bb8c9726ada7bb04dd337ec5a65454abff\
         a9b6078f9bb3e078684bd90172bd7200c7d12c1428bfe095da20283a78b6dc07"
    ).unwrap();
    
    assert_eq!(cpu_output, expected, "CPU Argon2H' should exactly match CUDA");
}

#[test]
fn test_argon2_hprime_256_bytes() {
    // Test VM initialization size (256 bytes)
    let input = b"test seed for vm initialization";
    let mut cpu_output = vec![0u8; 256];
    argon2::hprime(&mut cpu_output, input);
    
    println!("CPU Argon2H'(31 bytes, 256): {}", hex::encode(&cpu_output[..64]));
    
    // Expected from CUDA test: 64d452562dc9c4307a1ef5a13128f6003270600b646864cb7c6e29e17816929e...
    let expected_start = hex::decode(
        "64d452562dc9c4307a1ef5a13128f6003270600b646864cb7c6e29e17816929e\
         382dfd28f405bbf31be01b79216a9729504b132cc20fb23f530888112ade7a65"
    ).unwrap();
    
    assert_eq!(&cpu_output[..64], &expected_start[..], "CPU Argon2H' 256-byte output should match CUDA");
}

#[test]
fn test_argon2_hprime_448_bytes() {
    // Test exact VM init size (448 bytes = 32 regs + 3 digests)
    let input = b"rom_digest_plus_salt";
    let mut cpu_output = vec![0u8; 448];
    argon2::hprime(&mut cpu_output, input);
    
    println!("CPU Argon2H'(20 bytes, 448) first 64: {}", hex::encode(&cpu_output[..64]));
    
    // Verify it produces reasonable output
    assert_ne!(&cpu_output[..64], &vec![0u8; 64][..], "Output should not be all zeros");
    assert_ne!(&cpu_output[64..128], &cpu_output[..64], "Different chunks should differ");
}

#[test]
fn test_argon2_hprime_5120_bytes() {
    // Test program shuffle size (5120 bytes = 256 instructions * 20 bytes)
    let input = b"program shuffle seed";
    let mut cpu_output = vec![0u8; 5120];
    argon2::hprime(&mut cpu_output, input);
    
    println!("CPU Argon2H'(20 bytes, 5120) first 64: {}", hex::encode(&cpu_output[..64]));
    
    // Expected from CUDA test: 7d9d53fad047c1462353774248cc4516d0f2f4f79c8b0cea805edb66d6f456ad...
    let expected_start = hex::decode(
        "7d9d53fad047c1462353774248cc4516d0f2f4f79c8b0cea805edb66d6f456ad\
         43f41832cff4dddc337b291383db045d6e497c079aece385ec72b98a3171d1a5"
    ).unwrap();
    
    assert_eq!(&cpu_output[..64], &expected_start[..], "CPU Argon2H' program-size output should match CUDA");
}

#[test]
fn test_blake2b_empty() {
    use cryptoxide::hashing::blake2b::Blake2b;
    
    let input = b"";
    let result = Blake2b::<512>::new().update(input).finalize();
    
    println!("CPU Blake2b(empty): {}", hex::encode(&result[..32]));
    
    // Known Blake2b-512 hash of empty string
    let expected = hex::decode(
        "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419\
         d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce"
    ).unwrap();
    
    assert_eq!(&result[..], &expected[..], "CPU Blake2b empty should match known value");
}

#[test]
fn test_blake2b_abc() {
    let input = b"abc";
    let result = Blake2b::<512>::new().update(input).finalize();
    
    println!("CPU Blake2b('abc'): {}", hex::encode(&result[..32]));
    
    // Known Blake2b-512 hash of "abc"
    let expected = hex::decode(
        "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1\
         7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923"
    ).unwrap();
    
    assert_eq!(&result[..], &expected[..], "CPU Blake2b 'abc' should match known value");
}

#[test]
fn test_blake2b_updates() {
    // Test incremental updates
    let result1 = Blake2b::<512>::new()
        .update(b"hello")
        .update(b"world")
        .finalize();
    
    let result2 = Blake2b::<512>::new()
        .update(b"helloworld")
        .finalize();
    
    assert_eq!(result1, result2, "Incremental updates should match single update");
}

#[test]
fn test_blake2b_with_key() {
    // Test keyed hashing
    let key = b"secret key";
    let data = b"message";
    
    let result = Blake2b::<512>::new_keyed(key).update(data).finalize();
    
    println!("CPU Blake2b keyed: {}", hex::encode(&result[..32]));
    
    // Should differ from non-keyed version
    let result_no_key = Blake2b::<512>::new().update(data).finalize();
    assert_ne!(result, result_no_key, "Keyed hash should differ from non-keyed");
}

#[test]
fn test_vm_init_components() {
    // Test the exact components used in VM initialization
    // This tests the Argon2H' call that initializes VM state
    
    let rom_digest = [0x42u8; 64];  // Dummy ROM digest
    let salt = b"test_salt";
    
    // Create input: rom_digest || salt
    let mut input = rom_digest.to_vec();
    input.extend_from_slice(salt);
    
    // Generate 448 bytes: 32 regs (256) + prog_digest (64) + mem_digest (64) + prog_seed (64)
    let mut init_buffer = vec![0u8; 448];
    argon2::hprime(&mut init_buffer, &input);
    
    println!("VM init buffer first 64 bytes: {}", hex::encode(&init_buffer[..64]));
    
    // Verify structure
    let regs_bytes = &init_buffer[0..256];
    let prog_digest_init = &init_buffer[256..320];
    let mem_digest_init = &init_buffer[320..384];
    let prog_seed = &init_buffer[384..448];
    
    // Parse first register as example
    let reg0 = u64::from_le_bytes(regs_bytes[0..8].try_into().unwrap());
    println!("First register: 0x{:016x}", reg0);
    println!("prog_digest init: {}", hex::encode(&prog_digest_init[..16]));
    println!("mem_digest init: {}", hex::encode(&mem_digest_init[..16]));
    println!("prog_seed: {}", hex::encode(&prog_seed[..16]));
    
    // Sanity checks
    assert_ne!(reg0, 0, "First register should not be zero");
    assert_ne!(prog_digest_init, mem_digest_init, "Digest inits should differ");
}
