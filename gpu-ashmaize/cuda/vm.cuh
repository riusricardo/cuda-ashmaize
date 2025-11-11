#ifndef ASHMAIZE_VM_CUH
#define ASHMAIZE_VM_CUH

#include "common.cuh"
#include "blake2b.cuh"

// Maximum program size
// 1024 instructions * 20 bytes = 20KB per VM instance
// This is a reasonable limit that balances flexibility with memory usage
// For 256 threads/block: 256 * 20KB = 5MB of register/local memory
constexpr size_t MAX_PROGRAM_INSTRS = 1024;
constexpr size_t MAX_PROGRAM_SIZE = MAX_PROGRAM_INSTRS * INSTR_SIZE;

// Program structure
struct Program {
    uint8_t instructions[MAX_PROGRAM_SIZE];  // Inline storage for instructions
    uint32_t nb_instrs;
};

// VM state (per thread)
struct VM {
    // Registers
    uint64_t regs[NB_REGS];
    
    // Program counter and counters
    uint32_t ip;
    uint32_t memory_counter;
    uint32_t loop_counter;
    
    // Digest contexts
    Blake2bState prog_digest;
    Blake2bState mem_digest;
    
    // Program seed
    uint8_t prog_seed[DIGEST_SIZE];
    
    // Program
    Program program;
};

// VM initialization
DEVICE void vm_init(
    VM* vm,
    const uint8_t* rom_digest,
    const uint8_t* salt,
    size_t salt_len,
    uint32_t nb_instrs
);

// Execute one instruction (device-only due to texture)
__device__ void vm_step(VM* vm, cudaTextureObject_t rom_texture, size_t rom_size);

// Post-instructions mixing
DEVICE void vm_post_instructions(VM* vm);

// Finalize VM and produce hash
DEVICE void vm_finalize(VM* vm, uint8_t* output_hash);

#endif // ASHMAIZE_VM_CUH
