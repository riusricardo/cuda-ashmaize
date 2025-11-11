#ifndef ASHMAIZE_BLAKE2B_CUH
#define ASHMAIZE_BLAKE2B_CUH

#include "common.cuh"

// Blake2b-512 state structure
struct Blake2bState {
    uint64_t h[8];        // Hash state (64 bytes)
    uint64_t t[2];        // Total bytes processed (16 bytes)
    uint64_t f[2];        // Finalization flags (16 bytes)
    uint8_t buf[BLAKE2B_BLOCK_SIZE];  // Input buffer (128 bytes)
    size_t buflen;        // Current buffer length
    size_t outlen;        // Output length (typically 64)
};

// Blake2b initialization
DEVICE void blake2b_init(Blake2bState* S, const uint8_t* key, size_t keylen);

// Initialize from existing data (for VM digest setup)
DEVICE void blake2b_init_from_data(Blake2bState* S, const uint8_t* data, size_t len);

// Update with new data (incremental)
DEVICE void blake2b_update(Blake2bState* S, const uint8_t* data, size_t len);

// Finalize and produce digest
DEVICE void blake2b_final(Blake2bState* S, uint8_t* out);

// Clone state (for Special1/Special2 operands)
DEVICE void blake2b_clone(Blake2bState* dst, const Blake2bState* src);

// One-shot hash (convenience)
DEVICE void blake2b_hash(uint8_t* out, const uint8_t* in, size_t inlen);

#endif // ASHMAIZE_BLAKE2B_CUH
