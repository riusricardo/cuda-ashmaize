#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include "../cuda/argon2.cuh"
#include "../cuda/blake2b.cuh"

// Helper to print hex
void print_hex(const char* label, const uint8_t* data, size_t len) {
    printf("%s: ", label);
    for (size_t i = 0; i < len; i++) {
        printf("%02x", data[i]);
    }
    printf("\n");
}

// Helper to compare byte arrays
bool compare_bytes(const uint8_t* a, const uint8_t* b, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (a[i] != b[i]) {
            printf("Mismatch at byte %zu: expected %02x, got %02x\n", i, b[i], a[i]);
            return false;
        }
    }
    return true;
}

// Test kernel wrapper
__global__ void test_argon2_kernel(
    uint8_t* d_output,
    size_t output_len,
    const uint8_t* d_input,
    size_t input_len
) {
    argon2_hprime(d_output, output_len, d_input, input_len);
}

// Helper to run Argon2H' test
bool test_argon2_hprime(
    const char* test_name,
    const uint8_t* input,
    size_t input_len,
    size_t output_len,
    const uint8_t* expected = nullptr  // For verification tests
) {
    printf("\nTest: %s\n", test_name);
    printf("Input length: %zu bytes, Output length: %zu bytes\n", input_len, output_len);
    
    // Allocate device memory
    uint8_t *d_input, *d_output;
    cudaMalloc(&d_input, input_len);
    cudaMalloc(&d_output, output_len);
    
    // Copy input to device
    cudaMemcpy(d_input, input, input_len, cudaMemcpyHostToDevice);
    
    // Run kernel
    test_argon2_kernel<<<1, 1>>>(d_output, output_len, d_input, input_len);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return false;
    }
    
    cudaDeviceSynchronize();
    
    // Copy result back
    uint8_t* output = new uint8_t[output_len];
    cudaMemcpy(output, d_output, output_len, cudaMemcpyDeviceToHost);
    
    // Print first 64 bytes for inspection
    size_t print_len = output_len < 64 ? output_len : 64;
    print_hex("Output (first 64 bytes)", output, print_len);
    
    // Verify if expected provided
    bool success = true;
    if (expected != nullptr) {
        success = compare_bytes(output, expected, output_len);
        if (success) {
            printf("✓ Verification PASSED\n");
        } else {
            printf("✗ Verification FAILED\n");
            print_hex("Expected (first 64)", expected, print_len);
        }
    } else {
        printf("✓ Test completed (no verification)\n");
    }
    
    // Cleanup
    delete[] output;
    cudaFree(d_input);
    cudaFree(d_output);
    
    return success;
}

// Test 1: Empty input, small output
bool test_empty_input() {
    uint8_t input[] = "";
    return test_argon2_hprime(
        "Empty input, 32 bytes output",
        input, 0,
        32
    );
}

// Test 2: Small input, 32 bytes output (exact single hash)
bool test_32_bytes() {
    uint8_t input[] = "hello";
    return test_argon2_hprime(
        "Small input, 32 bytes output",
        input, 5,
        32
    );
}

// Test 3: Small input, 64 bytes output (two iterations)
bool test_64_bytes() {
    uint8_t input[] = "hello world";
    return test_argon2_hprime(
        "Small input, 64 bytes output",
        input, 11,
        64
    );
}

// Test 4: 256 bytes output (typical for VM init buffer)
bool test_256_bytes() {
    uint8_t input[] = "test seed for VM initialization";
    return test_argon2_hprime(
        "256 bytes output (VM init size)",
        input, strlen((char*)input),
        256
    );
}

// Test 5: 5120 bytes output (typical for program instructions)
bool test_5120_bytes() {
    uint8_t input[] = "program shuffle seed";
    return test_argon2_hprime(
        "5120 bytes output (program shuffle size)",
        input, strlen((char*)input),
        5120
    );
}

// Test 6: Large output (256KB ROM)
bool test_256kb() {
    uint8_t input[] = "ROM generation seed";
    size_t rom_size = 256 * 1024;
    return test_argon2_hprime(
        "256KB output (ROM size)",
        input, strlen((char*)input),
        rom_size
    );
}

// Test 7: Zero-length output
bool test_zero_output() {
    uint8_t input[] = "any input";
    return test_argon2_hprime(
        "Zero-length output",
        input, 9,
        0
    );
}

// Test 8: Odd output size (not multiple of 32)
bool test_odd_size_33() {
    uint8_t input[] = "test";
    return test_argon2_hprime(
        "33 bytes output (1 full + 1 partial)",
        input, 4,
        33
    );
}

// Test 9: Odd output size (63 bytes)
bool test_odd_size_63() {
    uint8_t input[] = "test";
    return test_argon2_hprime(
        "63 bytes output (1 full + 31 partial)",
        input, 4,
        63
    );
}

// Test 10: Odd output size (100 bytes)
bool test_odd_size_100() {
    uint8_t input[] = "test";
    return test_argon2_hprime(
        "100 bytes output (3 full + 4 partial)",
        input, 4,
        100
    );
}

// Test 11: Different inputs should produce different outputs
bool test_different_inputs() {
    printf("\nTest: Different inputs produce different outputs\n");
    
    uint8_t input1[] = "seed1";
    uint8_t input2[] = "seed2";
    size_t output_len = 64;
    
    uint8_t *d_input1, *d_input2, *d_output1, *d_output2;
    cudaMalloc(&d_input1, 5);
    cudaMalloc(&d_input2, 5);
    cudaMalloc(&d_output1, output_len);
    cudaMalloc(&d_output2, output_len);
    
    cudaMemcpy(d_input1, input1, 5, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input2, 5, cudaMemcpyHostToDevice);
    
    test_argon2_kernel<<<1, 1>>>(d_output1, output_len, d_input1, 5);
    test_argon2_kernel<<<1, 1>>>(d_output2, output_len, d_input2, 5);
    cudaDeviceSynchronize();
    
    uint8_t output1[64], output2[64];
    cudaMemcpy(output1, d_output1, output_len, cudaMemcpyDeviceToHost);
    cudaMemcpy(output2, d_output2, output_len, cudaMemcpyDeviceToHost);
    
    bool different = false;
    for (size_t i = 0; i < output_len; i++) {
        if (output1[i] != output2[i]) {
            different = true;
            break;
        }
    }
    
    print_hex("Output1", output1, output_len);
    print_hex("Output2", output2, output_len);
    
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output1);
    cudaFree(d_output2);
    
    if (different) {
        printf("✓ Outputs are different (correct)\n");
        return true;
    } else {
        printf("✗ Outputs are identical (WRONG!)\n");
        return false;
    }
}

// Test 12: Same input should produce same output (determinism)
bool test_determinism() {
    printf("\nTest: Determinism check\n");
    
    uint8_t input[] = "determinism test";
    size_t output_len = 256;
    
    uint8_t *d_input, *d_output1, *d_output2;
    cudaMalloc(&d_input, 16);
    cudaMalloc(&d_output1, output_len);
    cudaMalloc(&d_output2, output_len);
    
    cudaMemcpy(d_input, input, 16, cudaMemcpyHostToDevice);
    
    // Run twice
    test_argon2_kernel<<<1, 1>>>(d_output1, output_len, d_input, 16);
    cudaDeviceSynchronize();
    test_argon2_kernel<<<1, 1>>>(d_output2, output_len, d_input, 16);
    cudaDeviceSynchronize();
    
    uint8_t *output1 = new uint8_t[output_len];
    uint8_t *output2 = new uint8_t[output_len];
    cudaMemcpy(output1, d_output1, output_len, cudaMemcpyDeviceToHost);
    cudaMemcpy(output2, d_output2, output_len, cudaMemcpyDeviceToHost);
    
    bool success = compare_bytes(output1, output2, output_len);
    
    if (success) {
        printf("✓ Outputs are identical (deterministic)\n");
    } else {
        printf("✗ Outputs differ (non-deterministic!)\n");
    }
    
    delete[] output1;
    delete[] output2;
    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);
    
    return success;
}

// Test 13: Manual verification of algorithm
bool test_manual_verification() {
    printf("\nTest: Manual algorithm verification\n");
    
    // Use simple input to manually verify
    uint8_t input[] = "abc";
    size_t output_len = 64;  // Should need 2 iterations
    
    // Step 1: Compute V0 = Blake2b(LE32(64) || "abc")
    printf("Computing V0 = Blake2b(LE32(64) || 'abc')\n");
    uint8_t v0_input[7];
    v0_input[0] = 64;  // LE32(64) = 0x40 0x00 0x00 0x00
    v0_input[1] = 0;
    v0_input[2] = 0;
    v0_input[3] = 0;
    v0_input[4] = 'a';
    v0_input[5] = 'b';
    v0_input[6] = 'c';
    
    uint8_t v0[64];
    Blake2bState state;
    blake2b_init(&state, nullptr, 0);
    blake2b_update(&state, v0_input, 7);
    blake2b_final(&state, v0);
    
    print_hex("V0", v0, 64);
    print_hex("V0[0..32]", v0, 32);
    
    // Step 2: Compute V1 = Blake2b(V0)
    printf("\nComputing V1 = Blake2b(V0)\n");
    uint8_t v1[64];
    blake2b_hash(v1, v0, 64);
    print_hex("V1", v1, 64);
    print_hex("V1[0..32]", v1, 32);
    
    // Step 3: Expected output = V0[0..32] || V1[0..32]
    uint8_t expected[64];
    memcpy(expected, v0, 32);
    memcpy(expected + 32, v1, 32);
    
    printf("\nExpected output:\n");
    print_hex("Expected", expected, 64);
    
    // Now test argon2_hprime
    return test_argon2_hprime(
        "Manual verification",
        input, 3,
        64,
        expected
    );
}

int main() {
    printf("=== Argon2H' Test Suite ===\n");
    printf("Testing CUDA implementation of Argon2 H-Prime\n");
    
    int passed = 0;
    int total = 0;
    
    // Basic functionality tests
    total++; if (test_empty_input()) passed++;
    total++; if (test_32_bytes()) passed++;
    total++; if (test_64_bytes()) passed++;
    total++; if (test_zero_output()) passed++;
    
    // Common size tests
    total++; if (test_256_bytes()) passed++;
    total++; if (test_5120_bytes()) passed++;
    total++; if (test_256kb()) passed++;
    
    // Boundary tests
    total++; if (test_odd_size_33()) passed++;
    total++; if (test_odd_size_63()) passed++;
    total++; if (test_odd_size_100()) passed++;
    
    // Property tests
    total++; if (test_different_inputs()) passed++;
    total++; if (test_determinism()) passed++;
    
    // Verification test
    total++; if (test_manual_verification()) passed++;
    
    printf("\n=== Test Results ===\n");
    printf("Passed: %d/%d\n", passed, total);
    
    if (passed == total) {
        printf("✓ All tests PASSED\n");
        return 0;
    } else {
        printf("✗ Some tests FAILED\n");
        return 1;
    }
}
