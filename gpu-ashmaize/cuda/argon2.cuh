#ifndef ASHMAIZE_ARGON2_CUH
#define ASHMAIZE_ARGON2_CUH

#include "common.cuh"

// Argon2 H-Prime function for variable-length output
// Used for VM initialization, program shuffling, and mixing
DEVICE void argon2_hprime(uint8_t* output, size_t output_len, const uint8_t* input, size_t input_len);

#endif // ASHMAIZE_ARGON2_CUH
