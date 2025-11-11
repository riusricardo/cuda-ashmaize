#ifndef ASHMAIZE_INSTRUCTIONS_CUH
#define ASHMAIZE_INSTRUCTIONS_CUH

#include "common.cuh"
#include "vm.cuh"

// Instruction structure (decoded)
struct Instruction {
    uint8_t opcode;
    uint8_t op1_type;
    uint8_t op2_type;
    uint8_t r1, r2, r3;
    uint64_t lit1, lit2;
};

// Decode 20-byte instruction
DEVICE void decode_instruction(Instruction* instr, const uint8_t* instr_bytes);

// Execute decoded instruction (device-only due to texture)
__device__ void execute_instruction(VM* vm, const Instruction* instr, cudaTextureObject_t rom_texture, size_t rom_size);

// Load operand value (device-only due to texture)
__device__ uint64_t load_operand(
    VM* vm,
    const Instruction* instr,
    int operand_num,
    cudaTextureObject_t rom_texture,
    size_t rom_size
);

#endif // ASHMAIZE_INSTRUCTIONS_CUH
