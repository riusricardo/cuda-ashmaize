#include "blake2b.cuh"

// Blake2b initialization vectors (first 64 bits of fractional parts of sqrt of first 8 primes)
__constant__ static const uint64_t blake2b_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

// Blake2b sigma permutations
__constant__ static const uint8_t blake2b_sigma[12][16] = {
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13 , 0 },
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 }
};

// Blake2b mixing function G
DEVICE FORCEINLINE void blake2b_G(
    uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d,
    uint64_t x, uint64_t y
) {
    a = a + b + x;
    d = rotr64(d ^ a, 32);
    c = c + d;
    b = rotr64(b ^ c, 24);
    a = a + b + y;
    d = rotr64(d ^ a, 16);
    c = c + d;
    b = rotr64(b ^ c, 63);
}

// Blake2b compression function
DEVICE void blake2b_compress(Blake2bState* S, const uint8_t* block) {
    uint64_t m[16];
    uint64_t v[16];
    
    // Load message block in little-endian
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        m[i] = load_le64(block + i * 8);
    }
    
    // Initialize work vector
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = S->h[i];
        v[i + 8] = blake2b_IV[i];
    }
    
    // Mix counter and finalization flags
    v[12] ^= S->t[0];
    v[13] ^= S->t[1];
    v[14] ^= S->f[0];
    v[15] ^= S->f[1];
    
    // 12 rounds of mixing
    #pragma unroll
    for (int round = 0; round < 12; ++round) {
        const uint8_t* s = blake2b_sigma[round];
        
        // Column mixing
        blake2b_G(v[0], v[4], v[ 8], v[12], m[s[ 0]], m[s[ 1]]);
        blake2b_G(v[1], v[5], v[ 9], v[13], m[s[ 2]], m[s[ 3]]);
        blake2b_G(v[2], v[6], v[10], v[14], m[s[ 4]], m[s[ 5]]);
        blake2b_G(v[3], v[7], v[11], v[15], m[s[ 6]], m[s[ 7]]);
        
        // Diagonal mixing
        blake2b_G(v[0], v[5], v[10], v[15], m[s[ 8]], m[s[ 9]]);
        blake2b_G(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
        blake2b_G(v[2], v[7], v[ 8], v[13], m[s[12]], m[s[13]]);
        blake2b_G(v[3], v[4], v[ 9], v[14], m[s[14]], m[s[15]]);
    }
    
    // XOR the two halves
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        S->h[i] ^= v[i] ^ v[i + 8];
    }
}

// Initialize Blake2b state
DEVICE void blake2b_init(Blake2bState* S, const uint8_t* key, size_t keylen) {
    // Initialize hash state with IV
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        S->h[i] = blake2b_IV[i];
    }
    
    // Parameter block (outlen = 64, keylen, fanout=1, depth=1)
    S->h[0] ^= 0x01010000 ^ (keylen << 8) ^ 64;
    
    // Initialize counters
    S->t[0] = 0;
    S->t[1] = 0;
    S->f[0] = 0;
    S->f[1] = 0;
    
    S->buflen = 0;
    S->outlen = 64;
    
    // Always zero the buffer to prevent non-determinism
    memset_device(S->buf, 0, BLAKE2B_BLOCK_SIZE);
    
    // Process key if present
    if (keylen > 0) {
        memcpy_device(S->buf, key, keylen);
        S->buflen = BLAKE2B_BLOCK_SIZE;
    }
}

// Initialize from existing data (for VM digests)
DEVICE void blake2b_init_from_data(Blake2bState* S, const uint8_t* data, size_t len) {
    // Standard init with no key
    blake2b_init(S, nullptr, 0);
    
    // Update with initialization data
    blake2b_update(S, data, len);
}

// Update Blake2b state with new data
DEVICE void blake2b_update(Blake2bState* S, const uint8_t* data, size_t len) {
    if (len == 0) return;
    
    size_t offset = 0;
    
    // Process buffered data first
    if (S->buflen > 0) {
        size_t fill = BLAKE2B_BLOCK_SIZE - S->buflen;
        if (len <= fill) {
            memcpy_device(S->buf + S->buflen, data, len);
            S->buflen += len;
            return;
        }
        
        // Fill buffer and process
        memcpy_device(S->buf + S->buflen, data, fill);
        S->t[0] += BLAKE2B_BLOCK_SIZE;
        if (S->t[0] < BLAKE2B_BLOCK_SIZE) {
            S->t[1]++; // Carry
        }
        blake2b_compress(S, S->buf);
        S->buflen = 0;
        offset = fill;
    }
    
    // Process full blocks (but keep last block for finalization)
    while (offset + BLAKE2B_BLOCK_SIZE < len) {
        S->t[0] += BLAKE2B_BLOCK_SIZE;
        if (S->t[0] < BLAKE2B_BLOCK_SIZE) {
            S->t[1]++; // Carry
        }
        blake2b_compress(S, data + offset);
        offset += BLAKE2B_BLOCK_SIZE;
    }
    
    // Buffer remaining data (including last full block if any)
    if (offset < len) {
        size_t remaining = len - offset;
        memcpy_device(S->buf, data + offset, remaining);
        S->buflen = remaining;
    }
}

// Finalize Blake2b and produce digest
DEVICE void blake2b_final(Blake2bState* S, uint8_t* out) {
    // Pad final block with zeros
    if (S->buflen < BLAKE2B_BLOCK_SIZE) {
        memset_device(S->buf + S->buflen, 0, BLAKE2B_BLOCK_SIZE - S->buflen);
    }
    
    // Update counter with final block
    S->t[0] += S->buflen;
    if (S->t[0] < S->buflen) {
        S->t[1]++; // Carry
    }
    
    // Set finalization flag
    S->f[0] = ~0ULL;
    
    // Final compression
    blake2b_compress(S, S->buf);
    
    // Output hash in little-endian
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        store_le64(out + i * 8, S->h[i]);
    }
}

// Clone Blake2b state (for Special operands)
DEVICE void blake2b_clone(Blake2bState* dst, const Blake2bState* src) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst->h[i] = src->h[i];
    }
    
    dst->t[0] = src->t[0];
    dst->t[1] = src->t[1];
    dst->f[0] = src->f[0];
    dst->f[1] = src->f[1];
    
    memcpy_device(dst->buf, src->buf, BLAKE2B_BLOCK_SIZE);
    dst->buflen = src->buflen;
    dst->outlen = src->outlen;
}

// One-shot Blake2b hash
DEVICE void blake2b_hash(uint8_t* out, const uint8_t* in, size_t inlen) {
    Blake2bState S;
    blake2b_init(&S, nullptr, 0);
    blake2b_update(&S, in, inlen);
    blake2b_final(&S, out);
}
