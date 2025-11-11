/**
 * Main AshMaize Mining Kernel - Header
 */

#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>
#include <cstdint>

/**
 * Main mining kernel
 * 
 * Each thread processes one nonce (salt value).
 */
__global__ void ashmaize_hash_kernel(
    const uint8_t* rom_digest,
    const uint8_t* salts,
    size_t salt_size,
    uint32_t nb_loops,
    uint32_t nb_instrs,
    cudaTextureObject_t rom_texture,
    size_t rom_size,
    uint8_t* outputs
);

/**
 * Mining kernel with difficulty checking
 * 
 * Only outputs salts that meet the difficulty requirement.
 */
__global__ void ashmaize_mine_kernel(
    const uint8_t* rom_digest,
    uint64_t salt_base,
    uint32_t num_salts,
    uint32_t nb_loops,
    uint32_t nb_instrs,
    cudaTextureObject_t rom_texture,
    uint32_t difficulty_bits,
    uint64_t* found_salts,
    uint32_t* found_count,
    uint32_t max_found
);

/**
 * Check if hash meets difficulty requirement
 */
__device__ bool check_difficulty(const uint8_t* hash, uint32_t zero_bits);

#endif // KERNEL_CUH
