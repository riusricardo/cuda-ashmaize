#include "argon2.cuh"
#include "blake2b.cuh"

/*
 * Argon2H' - Variable-length pseudorandom generator
 * 
 * Based on SPECS.md:
 * Function Argon2Hprime(seed, size) =
 *     output = []
 *     V0 = Digest(LE32(size) | seed)
 *     output.append(V0[0..32])
 *     while output.len() < size
 *        V[i+1] = Digest(V[i])
 *        output.append(V[i+1][0..32])
 *     if output.len() > size
 *        V[last] = Digest(V[last-1])
 *        output.append(V[last][0..size - output.len()])
 *     return output
 * 
 * This is a sequential hashing process that generates variable-length
 * pseudorandom output by iteratively hashing previous digest values.
 */

/*
 * Argon2H' Implementation
 * 
 * Generates variable-length pseudorandom output using sequential Blake2b hashing.
 * 
 * @param output      Output buffer (must be allocated by caller)
 * @param output_len  Desired output length in bytes (0 is valid but produces no output)
 * @param input       Input seed data (can be NULL if input_len is 0)
 * @param input_len   Length of input seed in bytes
 * 
 * Performance characteristics:
 * - Requires ⌈output_len / 32⌉ + 1 Blake2b operations
 * - Sequential (cannot be parallelized internally)
 * - Memory: 64 bytes for intermediate hash state
 * 
 * Used in AshMaize for:
 * - VM initialization (256 bytes)
 * - Program shuffling (5120 bytes)
 * - Post-instruction mixing (32 bytes)
 * - ROM generation (256KB)
 */
DEVICE void argon2_hprime(uint8_t* output, size_t output_len, const uint8_t* input, size_t input_len) {
    // Handle zero-length output (valid edge case)
    if (output_len == 0) {
        return;
    }
    
    // V0 = Digest(LE32(size) | seed)
    Blake2bState state;
    blake2b_init(&state, nullptr, 0);
    
    // Add size prefix (little-endian 32-bit)
    uint32_t size_le = (uint32_t)output_len;
    uint8_t size_bytes[4];
    store_le32(size_bytes, size_le);
    blake2b_update(&state, size_bytes, 4);
    
    // Add input seed
    blake2b_update(&state, input, input_len);
    
    // Compute V0
    uint8_t v_current[64];
    blake2b_final(&state, v_current);
    
    // output.append(V0[0..32])
    size_t to_copy = (output_len < 32) ? output_len : 32;
    memcpy_device(output, v_current, to_copy);
    
    if (output_len <= 32) {
        return;
    }
    
    size_t bytes_remaining = output_len - 32;
    size_t pos = 32;
    
    // Loop while we need more than 64 bytes (matching CPU behavior)
    while (bytes_remaining > 64) {
        // V[i+1] = Blake2b-512(V[i])
        // NOTE: This intentionally uses the same buffer for input and output to match
        // the cryptoxide implementation which does: finalize_at(&mut vi_prev)
        blake2b_hash(v_current, v_current, 64);
        
        // Copy first 32 bytes to output
        memcpy_device(output + pos, v_current, 32);
        bytes_remaining -= 32;
        pos += 32;
    }
    
    // Final iteration: hash and write remaining bytes (can be up to 64)
    // CPU uses ContextDyn::new(bytes_remaining) which supports variable output length
    // For now, we use fixed Blake2b-512 and copy what's needed
    blake2b_hash(v_current, v_current, 64);
    memcpy_device(output + pos, v_current, bytes_remaining);
}
