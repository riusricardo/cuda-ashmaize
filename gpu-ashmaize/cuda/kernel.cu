/**
 * Main AshMaize Mining Kernel
 * 
 * This implements the complete mining algorithm matching src/lib.rs
 */

#include "common.cuh"
#include "vm.cuh"
#include "instructions.cuh"
#include "argon2.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Structure to hold ROM data on GPU
struct RomHandle {
    uint8_t* d_rom_data;
    uint8_t* d_rom_digest;
    cudaTextureObject_t rom_texture;
    size_t rom_size;
};

/**
 * Execute a full mining iteration
 * 
 * Implements VM::execute() from Rust:
 * - Shuffle program using Argon2H'
 * - Execute nb_instrs instructions
 * - Post-instructions mixing
 */
/**
 * VM execution loop
 * Implements VM::execute() from Rust:
 * - Shuffle program using Argon2H'
 * - Execute nb_instrs instructions
 * - Post-instructions mixing
 */
__device__ void vm_execute(VM* vm, cudaTextureObject_t rom_texture, size_t rom_size, uint32_t nb_instrs) {
    // Safety: Use the actual program size (which may be clamped in vm_init)
    uint32_t actual_instrs = vm->program.nb_instrs;
    size_t program_size = actual_instrs * INSTR_SIZE;
    
    // Ensure we don't exceed buffer bounds
    if (program_size > MAX_PROGRAM_SIZE) {
        program_size = MAX_PROGRAM_SIZE;
        actual_instrs = MAX_PROGRAM_INSTRS;
    }
    
    // Shuffle program using Argon2H' on prog_seed
    argon2_hprime(vm->program.instructions, program_size, vm->prog_seed, 64);
    
    // DEBUG: Print first few shuffled bytes (only once)
    // if (threadIdx.x == 0 && blockIdx.x == 0 && vm->loop_counter == 0) {
    //     printf("GPU After shuffle - program[0-7]: %02x %02x %02x %02x %02x %02x %02x %02x\n",
    //            vm->program.instructions[0], vm->program.instructions[1],
    //            vm->program.instructions[2], vm->program.instructions[3],
    //            vm->program.instructions[4], vm->program.instructions[5],
    //            vm->program.instructions[6], vm->program.instructions[7]);
    // }
    
    // Execute instructions (use clamped value for safety)
    for (uint32_t i = 0; i < actual_instrs; i++) {
        // DEBUG: Print instruction 2 bytes before execution
        // if (i == 2 && threadIdx.x == 0 && blockIdx.x == 0 && vm->loop_counter == 0) {
        //     uint32_t start = (vm->ip * INSTR_SIZE) % (actual_instrs * INSTR_SIZE);
        //     printf("GPU Before instr 2 - bytes[0-7]: %02x %02x %02x %02x %02x %02x %02x %02x\n",
        //            vm->program.instructions[start], vm->program.instructions[start+1],
        //            vm->program.instructions[start+2], vm->program.instructions[start+3],
        //            vm->program.instructions[start+4], vm->program.instructions[start+5],
        //            vm->program.instructions[start+6], vm->program.instructions[start+7]);
        //     printf("GPU Before instr 2 - reg[2] BEFORE: %016llx\n", vm->regs[2]);
        // }
        
        vm_step(vm, rom_texture, rom_size);
        
        // DEBUG: Print registers after first few instructions
        // if (threadIdx.x == 0 && blockIdx.x == 0 && vm->loop_counter == 0) {
        //     if (i == 0) {
        //         printf("GPU After instr 0 - reg[0-3]: %016llx %016llx %016llx %016llx\n",
        //                vm->regs[0], vm->regs[1], vm->regs[2], vm->regs[3]);
        //     } else if (i == 1) {
        //         printf("GPU After instr 1 - reg[0-3]: %016llx %016llx %016llx %016llx\n",
        //                vm->regs[0], vm->regs[1], vm->regs[2], vm->regs[3]);
        //     } else if (i == 2) {
        //         printf("GPU After instr 2 - reg[0-3]: %016llx %016llx %016llx %016llx\n",
        //                vm->regs[0], vm->regs[1], vm->regs[2], vm->regs[3]);
        //     }
        // }
    }
    
    // DEBUG: Print state before post_instructions
    // if (threadIdx.x == 0 && blockIdx.x == 0 && vm->loop_counter == 0) {
    //     printf("GPU Before post_instr - reg[0-3]: %016llx %016llx %016llx %016llx\n",
    //            vm->regs[0], vm->regs[1], vm->regs[2], vm->regs[3]);
    //     printf("GPU Before post_instr - memory_counter: %u\n", vm->memory_counter);
    // }
    
    // Post-instructions mixing
    vm_post_instructions(vm);
}

/**
 * Check if hash meets difficulty requirement
 * 
 * Matches hash_structure_good() from examples/hash.rs
 */
__device__ bool check_difficulty(const uint8_t* hash, uint32_t zero_bits) {
    uint32_t full_bytes = zero_bits / 8;
    uint32_t remaining_bits = zero_bits % 8;
    
    // Check full zero bytes
    for (uint32_t i = 0; i < full_bytes; i++) {
        if (hash[i] != 0) return false;
    }
    
    // Check remaining bits in next byte
    if (remaining_bits > 0) {
        uint8_t mask = 0xFF << (8 - remaining_bits);
        if ((hash[full_bytes] & mask) != 0) return false;
    }
    
    return true;
}

/**
 * Main mining kernel with early exit optimization
 * 
 * Each thread processes one salt value.
 * If any thread finds a solution, all threads exit early to save GPU time.
 * Implements hash() function from src/lib.rs
 */
GLOBAL void ashmaize_mine_kernel(
    cudaTextureObject_t rom_texture,
    size_t rom_size,
    const uint8_t* __restrict__ d_salts,
    uint8_t* __restrict__ d_hashes,
    uint8_t* __restrict__ d_success_flags,
    const uint8_t* __restrict__ rom_digest,
    const uint8_t* __restrict__ difficulty_target,
    uint32_t nb_loops,
    uint32_t nb_instrs,
    uint32_t batch_size,
    uint32_t salt_len,
    uint32_t* __restrict__ d_solution_found  // Global flag for early exit
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;
    
    // Get this thread's salt
    const uint8_t* salt = &d_salts[tid * salt_len];
    
    // Initialize VM
    VM vm;
    vm_init(&vm, rom_digest, salt, salt_len, nb_instrs);
    
    // DEBUG: Print first 4 registers after init (only for thread 0)
    // if (tid == 0) {
    //     printf("GPU VM Init - First 4 regs: %016llx %016llx %016llx %016llx\n",
    //            vm.regs[0], vm.regs[1], vm.regs[2], vm.regs[3]);
    //     printf("GPU VM Init - prog_seed[0-7]: %02x %02x %02x %02x %02x %02x %02x %02x\n",
    //            vm.prog_seed[0], vm.prog_seed[1], vm.prog_seed[2], vm.prog_seed[3],
    //            vm.prog_seed[4], vm.prog_seed[5], vm.prog_seed[6], vm.prog_seed[7]);
    // }
    
    // Execute nb_loops iterations (typically 8)
    // Check for early exit after each loop if another thread found solution
    for (uint32_t loop = 0; loop < nb_loops; loop++) {
        // Early exit check: if another thread found a solution, stop computing
        if (d_solution_found != nullptr && atomicAdd(d_solution_found, 0) > 0) {
            d_success_flags[tid] = 0;  // Mark as not checked
            return;
        }
        
        vm_execute(&vm, rom_texture, rom_size, nb_instrs);
    }
    
    // Finalize and write output
    uint8_t* output = &d_hashes[tid * 64];
    vm_finalize(&vm, output);
    
    // Check difficulty if provided
    if (difficulty_target != nullptr) {
        // For now, use simple zero-bit check
        // difficulty_target[0] contains the number of required zero bits
        uint32_t required_bits = difficulty_target[0];
        bool is_solution = check_difficulty(output, required_bits);
        d_success_flags[tid] = is_solution ? 1 : 0;
        
        // If this thread found a solution, signal all other threads to stop
        if (is_solution && d_solution_found != nullptr) {
            atomicAdd(d_solution_found, 1);
        }
    } else {
        d_success_flags[tid] = 1;  // Always success if no difficulty check
    }
}

// C interface for Rust FFI
extern "C" {

// Initialize CUDA
int gpu_init() {
    // Select device
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        return -1;  // No CUDA devices
    }
    
    CUDA_CHECK(cudaSetDevice(0));
    return 0;
}

// Cleanup CUDA
int gpu_cleanup() {
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}

// Upload ROM to GPU and create texture
void* gpu_upload_rom(
    const uint8_t* rom_data,
    size_t rom_size,
    const uint8_t* rom_digest
) {
    // Safety: Validate input parameters
    if (rom_data == nullptr || rom_digest == nullptr) {
        fprintf(stderr, "Error: NULL pointer passed to gpu_upload_rom\n");
        return nullptr;
    }
    
    if (rom_size < 64) {
        fprintf(stderr, "Error: ROM size (%zu) must be at least 64 bytes\n", rom_size);
        return nullptr;
    }
    
    if (rom_size > 1024 * 1024 * 1024) {  // 1GB sanity check
        fprintf(stderr, "Error: ROM size (%zu) exceeds maximum (1GB)\n", rom_size);
        return nullptr;
    }
    
    RomHandle* handle = new RomHandle();
    handle->rom_size = rom_size;
    
    // Allocate and copy ROM data
    CUDA_CHECK(cudaMalloc(&handle->d_rom_data, rom_size));
    CUDA_CHECK(cudaMemcpy(handle->d_rom_data, rom_data, rom_size, cudaMemcpyHostToDevice));
    
    // Create texture object for ROM
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = handle->d_rom_data;
    resDesc.res.linear.desc = cudaCreateChannelDesc<uint8_t>();
    resDesc.res.linear.sizeInBytes = rom_size;
    
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    
    CUDA_CHECK(cudaCreateTextureObject(&handle->rom_texture, &resDesc, &texDesc, nullptr));
    
    // Allocate and copy ROM digest
    CUDA_CHECK(cudaMalloc(&handle->d_rom_digest, 64));
    CUDA_CHECK(cudaMemcpy(handle->d_rom_digest, rom_digest, 64, cudaMemcpyHostToDevice));
    
    return handle;
}

// Mine batch of salts with optional early exit
int gpu_mine_batch(
    void* rom_handle,
    const uint8_t* salts,
    uint8_t* hashes,
    uint8_t* flags,
    uint32_t batch_size,
    uint32_t salt_len,
    uint32_t nb_loops,
    uint32_t nb_instrs
) {
    RomHandle* handle = (RomHandle*)rom_handle;
    
    // Safety: Validate all input parameters
    if (!handle) {
        fprintf(stderr, "Error: NULL ROM handle\n");
        return -1;
    }
    if (!salts || !hashes || !flags) {
        fprintf(stderr, "Error: NULL pointer in batch parameters\n");
        return -1;
    }
    if (batch_size == 0 || batch_size > 1000000) {
        fprintf(stderr, "Error: Invalid batch_size %u (must be 1-1000000)\n", batch_size);
        return -1;
    }
    if (salt_len == 0 || salt_len > 1024) {
        fprintf(stderr, "Error: Invalid salt_len %u (must be 1-1024)\n", salt_len);
        return -1;
    }
    if (nb_instrs == 0 || nb_instrs > MAX_PROGRAM_INSTRS) {
        fprintf(stderr, "Error: Invalid nb_instrs %u (must be 1-%zu)\n", nb_instrs, MAX_PROGRAM_INSTRS);
        return -1;
    }
    if (handle->rom_size < 64) {
        fprintf(stderr, "Error: ROM size %zu is too small (minimum 64 bytes)\n", handle->rom_size);
        return -1;
    }
    
    // Allocate device memory for salts
    uint8_t* d_salts;
    CUDA_CHECK(cudaMalloc(&d_salts, batch_size * salt_len));
    CUDA_CHECK(cudaMemcpy(d_salts, salts, batch_size * salt_len, cudaMemcpyHostToDevice));
    
    // Allocate device memory for outputs
    uint8_t* d_hashes;
    CUDA_CHECK(cudaMalloc(&d_hashes, batch_size * 64));
    
    uint8_t* d_flags;
    CUDA_CHECK(cudaMalloc(&d_flags, batch_size));
    
    // Allocate solution flag for early exit (initialized to 0)
    uint32_t* d_solution_found;
    CUDA_CHECK(cudaMalloc(&d_solution_found, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_solution_found, 0, sizeof(uint32_t)));
    
    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    ashmaize_mine_kernel<<<num_blocks, threads_per_block>>>(
        handle->rom_texture,
        handle->rom_size,
        d_salts,
        d_hashes,
        d_flags,
        handle->d_rom_digest,
        nullptr,  // No difficulty check for now
        nb_loops,
        nb_instrs,
        batch_size,
        salt_len,
        d_solution_found  // Pass early exit flag
    );
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(hashes, d_hashes, batch_size * 64, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(flags, d_flags, batch_size, cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_salts));
    CUDA_CHECK(cudaFree(d_hashes));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_solution_found));
    
    return 0;
}

// Free ROM
void gpu_free_rom(void* rom_handle) {
    if (!rom_handle) return;
    
    RomHandle* handle = (RomHandle*)rom_handle;
    
    // Destroy texture
    if (handle->rom_texture) {
        CUDA_CHECK(cudaDestroyTextureObject(handle->rom_texture));
    }
    
    // Free device memory
    if (handle->d_rom_data) {
        CUDA_CHECK(cudaFree(handle->d_rom_data));
    }
    
    if (handle->d_rom_digest) {
        CUDA_CHECK(cudaFree(handle->d_rom_digest));
    }
    
    delete handle;
}

} // extern "C"
