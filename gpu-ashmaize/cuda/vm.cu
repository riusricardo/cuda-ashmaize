#include "vm.cuh"
#include "argon2.cuh"
#include "instructions.cuh"
#include <stdio.h>

/*
 * VM Initialization
 * 
 * Initializes VM state from ROM digest and salt using Argon2H'.
 * 
 * Layout of 448-byte init_buffer:
 * - Bytes 0-255:   32 registers × 8 bytes (little-endian)
 * - Bytes 256-319: prog_digest initialization (64 bytes)
 * - Bytes 320-383: mem_digest initialization (64 bytes)
 * - Bytes 384-447: prog_seed (64 bytes)
 * 
 * @param vm          VM state to initialize
 * @param rom_digest  ROM digest (64 bytes)
 * @param salt        Salt bytes (variable length)
 * @param salt_len    Length of salt
 * @param nb_instrs   Number of instructions per program
 */
DEVICE void vm_init(
    VM* vm,
    const uint8_t* rom_digest,
    const uint8_t* salt,
    size_t salt_len,
    uint32_t nb_instrs
) {
    // Prepare Argon2H' input: rom_digest || salt
    uint8_t input[128];  // Max reasonable salt size
    memset_device(input, 0, 128);  // Zero initialize to prevent non-determinism
    memcpy_device(input, rom_digest, 64);
    if (salt_len > 0) {
        memcpy_device(input + 64, salt, salt_len);
    }
    
    // Derive 448 bytes of initialization data
    // This provides: 32 regs (256 bytes) + 3 digests (192 bytes)
    uint8_t init_buffer[448];
    argon2_hprime(init_buffer, 448, input, 64 + salt_len);
    
    // Initialize 32 registers from bytes 0-255 (little-endian)
    #pragma unroll
    for (int i = 0; i < NB_REGS; ++i) {
        vm->regs[i] = load_le64(&init_buffer[i * 8]);
    }
    
    // Initialize prog_digest from bytes 256-319
    blake2b_init_from_data(&vm->prog_digest, &init_buffer[256], 64);
    
    // Initialize mem_digest from bytes 320-383
    blake2b_init_from_data(&vm->mem_digest, &init_buffer[320], 64);
    
    // Initialize prog_seed from bytes 384-447
    memcpy_device(vm->prog_seed, &init_buffer[384], 64);
    
    // Initialize counters
    vm->ip = 0;
    vm->memory_counter = 0;
    vm->loop_counter = 0;
    
    // Initialize program structure with bounds checking
    // Clamp nb_instrs to MAX_PROGRAM_INSTRS to prevent buffer overflow
    if (nb_instrs > MAX_PROGRAM_INSTRS) {
        // Safety: Clamp to maximum supported size
        // This prevents undefined behavior from oversized programs
        vm->program.nb_instrs = MAX_PROGRAM_INSTRS;
    } else {
        vm->program.nb_instrs = nb_instrs;
    }
    
    // Initialize program instructions to zero (matches Rust Program::new)
    // The program starts as zeros and gets shuffled with Argon2H' on first execute
    size_t program_size = vm->program.nb_instrs * INSTR_SIZE;
    memset_device(vm->program.instructions, 0, program_size);
}

/*
 * VM Post-Instructions Mixing
 * 
 * Performs register mixing after executing all instructions in a loop.
 * This is the core diffusion mechanism that mixes VM state.
 * 
 * Algorithm:
 * 1. Sum all registers
 * 2. Clone and finalize digests with sum
 * 3. Hash: mixing_seed = Blake2b(prog_value || mem_value || loop_counter)
 * 4. Generate: mixing_data = Argon2H'(mixing_seed, 8192 bytes)
 * 5. XOR 32 rounds × 32 registers with mixing_data
 * 6. Update prog_seed for next loop
 * 7. Increment loop_counter
 * 
 * @param vm  VM state to mix
 */
DEVICE void vm_post_instructions(VM* vm) {
    // Step 1: Sum all registers
    uint64_t sum_regs = 0;
    #pragma unroll
    for (int i = 0; i < NB_REGS; ++i) {
        sum_regs += vm->regs[i];
    }
    
    // Step 2: Clone digests and finalize with sum
    uint8_t prog_value[64];
    uint8_t mem_value[64];
    
    {
        Blake2bState prog_clone;
        blake2b_clone(&prog_clone, &vm->prog_digest);
        uint8_t sum_bytes[8];
        store_le64(sum_bytes, sum_regs);
        blake2b_update(&prog_clone, sum_bytes, 8);
        blake2b_final(&prog_clone, prog_value);
    }
    
    {
        Blake2bState mem_clone;
        blake2b_clone(&mem_clone, &vm->mem_digest);
        uint8_t sum_bytes[8];
        store_le64(sum_bytes, sum_regs);
        blake2b_update(&mem_clone, sum_bytes, 8);
        blake2b_final(&mem_clone, mem_value);
    }
    
    // Step 3: Generate mixing seed
    // mixing_seed = Blake2b(prog_value || mem_value || loop_counter)
    uint8_t mixing_seed[64];
    {
        Blake2bState mixing_state;
        blake2b_init(&mixing_state, nullptr, 0);
        blake2b_update(&mixing_state, prog_value, 64);
        blake2b_update(&mixing_state, mem_value, 64);
        uint8_t lc_bytes[4];
        store_le32(lc_bytes, vm->loop_counter);
        blake2b_update(&mixing_state, lc_bytes, 4);
        blake2b_final(&mixing_state, mixing_seed);
    }
    
    // Step 4: Generate 8192 bytes of mixing data
    // 8192 = 32 rounds × 32 registers × 8 bytes
    constexpr size_t MIXING_SIZE = 32 * NB_REGS * 8;
    uint8_t mixing_data[MIXING_SIZE];
    argon2_hprime(mixing_data, MIXING_SIZE, mixing_seed, 64);
    
    // Step 5: XOR mixing data into registers (32 rounds)
    size_t offset = 0;
    for (int round = 0; round < 32; ++round) {
        #pragma unroll
        for (int i = 0; i < NB_REGS; ++i) {
            uint64_t mix_value = load_le64(&mixing_data[offset]);
            vm->regs[i] ^= mix_value;
            offset += 8;
        }
    }
    
    // Step 6: Update prog_seed for next loop
    memcpy_device(vm->prog_seed, prog_value, 64);
    
    // Step 7: Increment loop counter
    vm->loop_counter += 1;
}

/*
 * VM Finalization
 * 
 * Produces the final 64-byte hash from VM state.
 * 
 * Algorithm:
 *   output = Blake2b(
 *     prog_digest.finalize() ||
 *     mem_digest.finalize() ||
 *     memory_counter ||
 *     regs[0] || regs[1] || ... || regs[31]
 *   )
 * 
 * @param vm           VM state to finalize
 * @param output_hash  Output buffer (64 bytes)
 */
DEVICE void vm_finalize(VM* vm, uint8_t* output_hash) {
    // Finalize both digests
    uint8_t prog_final[64];
    uint8_t mem_final[64];
    
    blake2b_final(&vm->prog_digest, prog_final);
    blake2b_final(&vm->mem_digest, mem_final);
    
    // Build final hash: Blake2b(prog || mem || mc || all_regs)
    Blake2bState final_state;
    blake2b_init(&final_state, nullptr, 0);
    
    // Add finalized digests
    blake2b_update(&final_state, prog_final, 64);
    blake2b_update(&final_state, mem_final, 64);
    
    // Add memory counter (little-endian)
    uint8_t mc_bytes[4];
    store_le32(mc_bytes, vm->memory_counter);
    blake2b_update(&final_state, mc_bytes, 4);
    
    // Add all registers (little-endian)
    for (int i = 0; i < NB_REGS; ++i) {
        uint8_t reg_bytes[8];
        store_le64(reg_bytes, vm->regs[i]);
        blake2b_update(&final_state, reg_bytes, 8);
    }
    
    // Produce final hash
    blake2b_final(&final_state, output_hash);
}

/*
 * VM Step - Execute One Instruction
 * 
 * Decodes and executes one instruction, updating digests and IP.
 * 
 * @param vm           VM state
 * @param rom_texture  ROM texture object for memory operands
 */
/**
 * Execute one VM step (decode and execute instruction)
 * @param vm          VM state
 * @param rom_texture  ROM texture object for memory operands
 * @param rom_size     ROM size in bytes for address wrapping
 */
__device__ void vm_step(VM* vm, cudaTextureObject_t rom_texture, size_t rom_size) {
    // Get current instruction (20 bytes) with wrapping
    // Match Rust: let start = (i as usize).wrapping_mul(INSTR_SIZE) % self.instructions.len();
    uint32_t program_size = vm->program.nb_instrs * INSTR_SIZE;
    uint32_t start = (vm->ip * INSTR_SIZE) % program_size;
    const uint8_t* instr_bytes = &vm->program.instructions[start];
    
    // Decode instruction
    Instruction instr;
    decode_instruction(&instr, instr_bytes);
    
    // Execute instruction (modifies vm->regs, vm->memory_counter)
    execute_instruction(vm, &instr, rom_texture, rom_size);
    
    // Update prog_digest with instruction bytes
    blake2b_update(&vm->prog_digest, instr_bytes, 20);
    
    // Increment IP (wrapping)
    vm->ip = (vm->ip + 1);
}
