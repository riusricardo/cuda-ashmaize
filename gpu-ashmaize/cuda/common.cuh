#ifndef ASHMAIZE_COMMON_CUH
#define ASHMAIZE_COMMON_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Device code attributes
#define DEVICE __device__ __host__
#define HOST __host__
#define GLOBAL __global__
#define FORCEINLINE __forceinline__

// Constants
constexpr size_t REGISTER_SIZE = 8;
constexpr size_t NB_REGS = 32;
constexpr size_t INSTR_SIZE = 20;
constexpr size_t ROM_ACCESS_SIZE = 64;
constexpr size_t DIGEST_SIZE = 64;
constexpr size_t BLAKE2B_BLOCK_SIZE = 128;
constexpr size_t BLAKE2B_STATE_SIZE = 64;

// Forward declarations
struct Blake2bState;
struct VM;
struct Instruction;

// Operand types
enum OperandType : uint8_t {
    OPERAND_REG = 0,      // 0-4
    OPERAND_MEMORY = 5,   // 5-8
    OPERAND_LITERAL = 9,  // 9-12
    OPERAND_SPECIAL1 = 13, // 13
    OPERAND_SPECIAL2 = 14  // 14-15
};

// Operation types
enum Operation : uint8_t {
    OP_ADD = 0,
    OP_MUL,
    OP_MULH,
    OP_DIV,
    OP_MOD,
    OP_XOR,
    OP_AND,
    OP_HASH,
    OP_NEG,
    OP_ROTL,
    OP_ROTR,
    OP_ISQRT,
    OP_BITREV
};

// Utility functions
DEVICE FORCEINLINE uint64_t rotr64(uint64_t x, unsigned int n) {
    return (x >> n) | (x << (64 - n));
}

DEVICE FORCEINLINE uint64_t rotl64(uint64_t x, unsigned int n) {
    return (x << n) | (x >> (64 - n));
}

// Integer square root (Newton's method)
DEVICE FORCEINLINE uint64_t isqrt64(uint64_t n) {
    if (n == 0) return 0;
    if (n <= 3) return 1;
    
    uint64_t x = n;
    uint64_t y = (x + 1) / 2;
    
    // Newton iterations (converges quickly)
    while (y < x) {
        x = y;
        y = (x + n / x) / 2;
    }
    
    return x;
}

// Bit reversal using PTX intrinsics
DEVICE FORCEINLINE uint64_t reverse_bits64(uint64_t n) {
    uint32_t lo = static_cast<uint32_t>(n);
    uint32_t hi = static_cast<uint32_t>(n >> 32);
    uint32_t lo_rev, hi_rev;
    
    asm("brev.b32 %0, %1;" : "=r"(lo_rev) : "r"(lo));
    asm("brev.b32 %0, %1;" : "=r"(hi_rev) : "r"(hi));
    
    return (static_cast<uint64_t>(lo_rev) << 32) | hi_rev;
}

// High bits of 64-bit multiplication using PTX
DEVICE FORCEINLINE uint64_t mulhi64(uint64_t a, uint64_t b) {
    uint64_t hi;
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
    return hi;
}

// Memory utilities
DEVICE FORCEINLINE void memcpy_device(void* dst, const void* src, size_t n) {
    uint8_t* d = static_cast<uint8_t*>(dst);
    const uint8_t* s = static_cast<const uint8_t*>(src);
    for (size_t i = 0; i < n; ++i) {
        d[i] = s[i];
    }
}

DEVICE FORCEINLINE void memset_device(void* ptr, int value, size_t n) {
    uint8_t* p = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < n; ++i) {
        p[i] = static_cast<uint8_t>(value);
    }
}

// Load 64-bit value in little-endian
DEVICE FORCEINLINE uint64_t load_le64(const uint8_t* src) {
    uint64_t result;
    result  = static_cast<uint64_t>(src[0]);
    result |= static_cast<uint64_t>(src[1]) << 8;
    result |= static_cast<uint64_t>(src[2]) << 16;
    result |= static_cast<uint64_t>(src[3]) << 24;
    result |= static_cast<uint64_t>(src[4]) << 32;
    result |= static_cast<uint64_t>(src[5]) << 40;
    result |= static_cast<uint64_t>(src[6]) << 48;
    result |= static_cast<uint64_t>(src[7]) << 56;
    return result;
}

// Store 64-bit value in little-endian
DEVICE FORCEINLINE void store_le64(uint8_t* dst, uint64_t value) {
    dst[0] = static_cast<uint8_t>(value);
    dst[1] = static_cast<uint8_t>(value >> 8);
    dst[2] = static_cast<uint8_t>(value >> 16);
    dst[3] = static_cast<uint8_t>(value >> 24);
    dst[4] = static_cast<uint8_t>(value >> 32);
    dst[5] = static_cast<uint8_t>(value >> 40);
    dst[6] = static_cast<uint8_t>(value >> 48);
    dst[7] = static_cast<uint8_t>(value >> 56);
}

// Store 32-bit value in little-endian
DEVICE FORCEINLINE void store_le32(uint8_t* dst, uint32_t value) {
    dst[0] = static_cast<uint8_t>(value);
    dst[1] = static_cast<uint8_t>(value >> 8);
    dst[2] = static_cast<uint8_t>(value >> 16);
    dst[3] = static_cast<uint8_t>(value >> 24);
}

#endif // ASHMAIZE_COMMON_CUH
