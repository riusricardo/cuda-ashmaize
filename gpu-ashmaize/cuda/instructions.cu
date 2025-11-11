#include "instructions.cuh"
#include "blake2b.cuh"
#include <stdio.h>

/**
 * Decode 20-byte instruction from memory
 * 
 * Matches signature: decode_instruction(Instruction*, const uint8_t*)
 */
DEVICE void decode_instruction(Instruction* instr, const uint8_t* instr_bytes) {
    instr->opcode = instr_bytes[0];
    instr->op1_type = instr_bytes[1] >> 4;
    instr->op2_type = instr_bytes[1] & 0x0f;
    
    uint16_t rs = (static_cast<uint16_t>(instr_bytes[2]) << 8) | instr_bytes[3];
    instr->r1 = static_cast<uint8_t>((rs >> 10) & 0x1f);
    instr->r2 = static_cast<uint8_t>((rs >> 5) & 0x1f);
    instr->r3 = static_cast<uint8_t>(rs & 0x1f);
    
    instr->lit1 = load_le64(&instr_bytes[4]);
    instr->lit2 = load_le64(&instr_bytes[12]);
}

/**
 * Load operand value based on type
 * 
 * Handles all 5 operand types:
 * - Reg (0-4): Read from VM register
 * - Memory (5-8): Read from ROM + update mem_digest + increment memory_counter
 * - Literal (9-12): Use literal value directly
 * - Special1 (13): Finalize prog_digest and read first 8 bytes
 * - Special2 (14-15): Finalize mem_digest and read first 8 bytes
 * 
 * @param vm           VM state
 * @param instr        Decoded instruction
 * @param operand_num  Which operand (1 or 2)
 * @param rom_texture  ROM texture for memory reads
 * @return             64-bit operand value
 */
__device__ uint64_t load_operand(
    VM* vm,
    const Instruction* instr,
    int operand_num,
    cudaTextureObject_t rom_texture,
    size_t rom_size
) {
    uint8_t op_type = (operand_num == 1) ? instr->op1_type : instr->op2_type;
    uint8_t reg_idx = (operand_num == 1) ? instr->r1 : instr->r2;
    uint64_t literal = (operand_num == 1) ? instr->lit1 : instr->lit2;
    
    // Operand type ranges (from SPECS.md):
    // Register: 0-4
    // Memory: 5-8
    // Literal: 9-12
    // Special1: 13
    // Special2: 14-15
    
    if (op_type < 5) {
        // Register operand
        return vm->regs[reg_idx];
    }
    else if (op_type < 9) {
        // Memory operand - read from ROM
        // Use lower 32 bits of literal as ROM address
        uint32_t rom_addr = static_cast<uint32_t>(literal);
        
        // CRITICAL: Match CPU behavior exactly!
        // CPU rom.at(i): start = i % (data.len() / 64), then data[start..start+64]
        // This calculates a BLOCK INDEX (0 to num_blocks-1)
        // But then uses it directly as a BYTE OFFSET!
        // Example: if ROM is 262144 bytes (4096 blocks of 64 bytes)
        //   rom.at(5000) → start = 5000 % 4096 = 904
        //   Returns ROM bytes [904..968] (NOT [904*64..904*64+64])
        
        // Safety: Ensure ROM size is valid and can accommodate at least one 64-byte block
        if (rom_size < 64) {
            // Fallback: zero cache line for invalid ROM
            uint8_t cache_line[64] = {0};
            blake2b_update(&vm->mem_digest, cache_line, 64);
            vm->memory_counter += 1;
            size_t chunk_idx = (vm->memory_counter % 8) * 8;
            return 0;
        }
        
        size_t num_blocks = rom_size / 64;
        uint32_t byte_offset = (rom_addr % num_blocks);  // This gives 0 to num_blocks-1
        
        // Safety: Ensure byte_offset + 64 doesn't exceed ROM bounds
        // This can happen if num_blocks * 64 != rom_size (ROM not aligned to 64 bytes)
        if (byte_offset + 64 > rom_size) {
            byte_offset = rom_size - 64;  // Clamp to last valid 64-byte region
        }
        
        // Read 64-byte cache line from ROM texture with bounds-safe fetching
        uint8_t cache_line[64];
        #pragma unroll
        for (int i = 0; i < 64; i++) {
            // Each fetch is guaranteed to be within [0, rom_size)
            cache_line[i] = tex1Dfetch<uint8_t>(rom_texture, byte_offset + i);
        }
        
        // Update mem_digest with full 64-byte cache line
        blake2b_update(&vm->mem_digest, cache_line, 64);
        
        // Increment memory counter FIRST (matches CPU behavior)
        vm->memory_counter += 1;
        
        // Use memory_counter to select which 8-byte chunk (0-7) to return
        size_t chunk_idx = (vm->memory_counter % 8) * 8;
        uint64_t result = load_le64(&cache_line[chunk_idx]);
        
        return result;
    }
    else if (op_type < 13) {
        // Literal operand
        return literal;
    }
    else if (op_type == 13) {
        // Special1 - finalized prog_digest (first 8 bytes)
        Blake2bState temp_digest;
        blake2b_clone(&temp_digest, &vm->prog_digest);
        uint8_t hash[64];
        blake2b_final(&temp_digest, hash);
        return load_le64(&hash[0]);
    }
    else {
        // Special2 (14-15) - finalized mem_digest (first 8 bytes)
        Blake2bState temp_digest;
        blake2b_clone(&temp_digest, &vm->mem_digest);
        uint8_t hash[64];
        blake2b_final(&temp_digest, hash);
        return load_le64(&hash[0]);
    }
}

/**
 * Execute decoded instruction
 * 
 * Implements all 13 operations:
 * - Add (0-39): dst = src1 + src2
 * - Mul (40-79): dst = src1 * src2
 * - MulH (80-95): dst = (src1 * src2) >> 64 (high 64 bits)
 * - Div (96-111): dst = src1 / src2 (div by 0 → special1)
 * - Mod (112-127): dst = src1 % src2 (BUG: actually does division!)
 * - ISqrt (128-137): dst = floor(sqrt(src1))
 * - BitRev (138-147): dst = reverse_bits(src1)
 * - Xor (148-187): dst = src1 ^ src2
 * - RotL (188-203): dst = rotate_left(src1, r1)
 * - RotR (204-219): dst = rotate_right(src1, r1)
 * - Neg (220-239): dst = ~src1
 * - And (240-247): dst = src1 & src2
 * - Hash[N] (248-255): dst = blake2b(src1 || src2)[N*8..(N+1)*8]
 * 
 * @param vm          VM state (registers, digests, counters)
 * @param instr       Decoded instruction
 * @param rom_texture ROM texture for memory operands
 */
/**
 * Execute a single instruction
 * Updates VM registers and state
 * @param vm   VM state
 * @param instr Decoded instruction
 * @param rom_texture ROM texture for memory operands
 * @param rom_size ROM size in bytes for address wrapping
 */
__device__ void execute_instruction(
    VM* vm,
    const Instruction* instr,
    cudaTextureObject_t rom_texture,
    size_t rom_size
) {
    uint8_t opcode = instr->opcode;
    uint64_t result = 0;
    
    // Determine if this is a 2-operand or 3-operand instruction
    // Op2 instructions: ISqrt, BitRev, RotL, RotR, Neg
    bool is_op2 = (opcode >= 128 && opcode < 138) ||  // ISqrt
                  (opcode >= 138 && opcode < 148) ||  // BitRev
                  (opcode >= 188 && opcode < 240);    // RotL, RotR, Neg
    
    if (is_op2) {
        // 2-operand instruction: only needs src1
        uint64_t src1 = load_operand(vm, instr, 1, rom_texture, rom_size);
        
        if (opcode >= 128 && opcode < 138) {
            // ISqrt
            result = isqrt64(src1);
        }
        else if (opcode >= 138 && opcode < 148) {
            // BitRev
            result = reverse_bits64(src1);
        }
        else if (opcode >= 188 && opcode < 204) {
            // RotL - use r1 as rotation amount
            result = rotl64(src1, instr->r1);
        }
        else if (opcode >= 204 && opcode < 220) {
            // RotR - use r1 as rotation amount
            result = rotr64(src1, instr->r1);
        }
        else {
            // Neg (220-239)
            result = ~src1;
        }
    }
    else {
        // 3-operand instruction: needs src1 and src2
        uint64_t src1 = load_operand(vm, instr, 1, rom_texture, rom_size);
        uint64_t src2 = load_operand(vm, instr, 2, rom_texture, rom_size);
        
        if (opcode < 40) {
            // Add
            result = src1 + src2;
        }
        else if (opcode < 80) {
            // Mul
            // DEBUG: Print mul operands for opcode 71
            if (opcode == 71 && threadIdx.x == 0 && blockIdx.x == 0) {
                printf("GPU Mul (opcode=%u): src1=%016llx src2=%016llx r3=%u\n",
                       opcode, src1, src2, instr->r3);
            }
            result = src1 * src2;
        }
        else if (opcode < 96) {
            // MulH - high 64 bits of 128-bit multiply
            __uint128_t product = static_cast<__uint128_t>(src1) * static_cast<__uint128_t>(src2);
            result = static_cast<uint64_t>(product >> 64);
        }
        else if (opcode < 112) {
            // Div
            if (src2 == 0) {
                // Division by zero: use special1 value
                Blake2bState temp_digest;
                blake2b_clone(&temp_digest, &vm->prog_digest);
                uint8_t hash[64];
                blake2b_final(&temp_digest, hash);
                result = load_le64(&hash[0]);
            } else {
                result = src1 / src2;
            }
        }
        else if (opcode < 128) {
            // Mod - BUG IN SPEC: Actually does division, not modulo!
            // This matches the CPU implementation bug
            if (src2 == 0) {
                // Division by zero: use special1 value
                Blake2bState temp_digest;
                blake2b_clone(&temp_digest, &vm->prog_digest);
                uint8_t hash[64];
                blake2b_final(&temp_digest, hash);
                result = load_le64(&hash[0]);
            } else {
                result = src1 / src2;  // BUG: should be src1 % src2
            }
        }
        else if (opcode < 188) {
            // Xor (148-187)
            result = src1 ^ src2;
        }
        else if (opcode < 248) {
            // And (240-247)
            result = src1 & src2;
        }
        else {
            // Hash[N] (248-255)
            uint8_t n = opcode - 248;  // Hash index 0-7
            
            // Compute Blake2b(LE(src1) || LE(src2))
            uint8_t input[16];
            store_le64(&input[0], src1);
            store_le64(&input[8], src2);
            
            uint8_t hash[64];
            blake2b_hash(hash, input, 16);
            
            // Extract N-th 8-byte chunk
            result = load_le64(&hash[n * 8]);
        }
    }
    
    // Store result in destination register
    vm->regs[instr->r3] = result;
}
