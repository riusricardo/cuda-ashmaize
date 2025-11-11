/*
 * Blake2b Verification Tests
 * 
 * Compares CUDA implementation against Python hashlib.blake2b reference hashes
 */

#include "../cuda/blake2b.cuh"
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

// Device memcmp
__device__ int dev_memcmp(const void* s1, const void* s2, size_t n) {
    const uint8_t* p1 = (const uint8_t*)s1;
    const uint8_t* p2 = (const uint8_t*)s2;
    for (size_t i = 0; i < n; i++) {
        if (p1[i] != p2[i]) {
            return p1[i] - p2[i];
        }
    }
    return 0;
}

// Helper to parse hex string to bytes
void hex_to_bytes(const char* hex, uint8_t* bytes, size_t len) {
    for (size_t i = 0; i < len; i++) {
        sscanf(hex + i * 2, "%2hhx", &bytes[i]);
    }
}

__global__ void hash_kernel(const uint8_t* input, size_t len, uint8_t* output, const uint8_t* key, size_t keylen) {
    Blake2bState state;
    blake2b_init(&state, key, keylen);
    blake2b_update(&state, input, len);
    blake2b_final(&state, output);
}

bool verify_hash(const char* test_name, const uint8_t* input, size_t input_len, 
                 const uint8_t* key, size_t key_len, const char* expected_hex) {
    uint8_t expected[64];
    hex_to_bytes(expected_hex, expected, 64);
    
    uint8_t* d_input;
    uint8_t* d_key;
    uint8_t* d_output;
    uint8_t h_output[64];
    
    cudaMalloc(&d_input, input_len > 0 ? input_len : 1);
    cudaMalloc(&d_key, key_len > 0 ? key_len : 1);
    cudaMalloc(&d_output, 64);
    
    if (input_len > 0) cudaMemcpy(d_input, input, input_len, cudaMemcpyHostToDevice);
    if (key_len > 0) cudaMemcpy(d_key, key, key_len, cudaMemcpyHostToDevice);
    
    hash_kernel<<<1, 1>>>(d_input, input_len, d_output, key_len > 0 ? d_key : nullptr, key_len);
    cudaError_t err = cudaDeviceSynchronize();
    
    bool success = false;
    if (err == cudaSuccess) {
        cudaMemcpy(h_output, d_output, 64, cudaMemcpyDeviceToHost);
        success = (memcmp(h_output, expected, 64) == 0);
        
        if (success) {
            printf("✅ %s: PASSED\n", test_name);
        } else {
            printf("❌ %s: FAILED\n", test_name);
            printf("  Expected: ");
            for (int i = 0; i < 64; i++) printf("%02x", expected[i]);
            printf("\n  Got:      ");
            for (int i = 0; i < 64; i++) printf("%02x", h_output[i]);
            printf("\n");
        }
    } else {
        printf("❌ %s: CUDA ERROR: %s\n", test_name, cudaGetErrorString(err));
    }
    
    cudaFree(d_input);
    cudaFree(d_key);
    cudaFree(d_output);
    
    return success;
}

int main() {
    printf("=== Blake2b CUDA Verification vs Python hashlib ===\n\n");
    
    int passed = 0;
    int total = 0;
    
    // Prepare test data
    uint8_t data[1024];
    for (int i = 0; i < 1024; i++) {
        data[i] = (uint8_t)i;
    }
    
    // Length variation tests
    printf("--- Length Variation Tests ---\n");
    
    total++; passed += verify_hash("1 byte", data, 1, nullptr, 0,
        "2fa3f686df876995167e7c2e5d74c4c7b6e48f8068fe0e44208344d480f7904c36963e44115fe3eb2a3ac8694c28bcb4f5a0f3276f2e79487d8219057a506e4b");
    
    total++; passed += verify_hash("2 bytes", data, 2, nullptr, 0,
        "1c08798dc641aba9dee435e22519a4729a09b2bfe0ff00ef2dcd8ed6f8a07d15eaf4aee52bbf18ab5608a6190f70b90486c8a7d4873710b1115d3debbb4327b5");
    
    total++; passed += verify_hash("3 bytes", data, 3, nullptr, 0,
        "40a374727302d9a4769c17b5f409ff32f58aa24ff122d7603e4fda1509e919d4107a52c57570a6d94e50967aea573b11f86f473f537565c66f7039830a85d186");
    
    total++; passed += verify_hash("7 bytes", data, 7, nullptr, 0,
        "8f945ba700f2530e5c2a7df7d5dce0f83f9efc78c073fe71ae1f88204a4fd1cf70a073f5d1f942ed623aa16e90a871246c90c45b621b3401a5ddbd9df6264165");
    
    total++; passed += verify_hash("15 bytes", data, 15, nullptr, 0,
        "444b240fe3ed86d0e2ef4ce7d851edde22155582aa0914797b726cd058b6f45932e0e129516876527b1dd88fc66d7119f4ab3bed93a61a0e2d2d2aeac336d958");
    
    total++; passed += verify_hash("31 bytes", data, 31, nullptr, 0,
        "29f8b8c78c80f2fcb4bdf7825ed90a70d625ff785d262677e250c04f3720c888d03f8045e4edf3f5285bd39d928a10a7d0a5df00b8484ac2868142a1e8bea351");
    
    total++; passed += verify_hash("63 bytes", data, 63, nullptr, 0,
        "d10bf9a15b1c9fc8d41f89bb140bf0be08d2f3666176d13baac4d381358ad074c9d4748c300520eb026daeaea7c5b158892fde4e8ec17dc998dcd507df26eb63");
    
    total++; passed += verify_hash("64 bytes", data, 64, nullptr, 0,
        "2fc6e69fa26a89a5ed269092cb9b2a449a4409a7a44011eecad13d7c4b0456602d402fa5844f1a7a758136ce3d5d8d0e8b86921ffff4f692dd95bdc8e5ff0052");
    
    total++; passed += verify_hash("65 bytes", data, 65, nullptr, 0,
        "fcbe8be7dcb49a32dbdf239459e26308b84dff1ea480df8d104eeff34b46fae98627b450c2267d48c0946a697c5b59531452ac0484f1c84e3a33d0c339bb2e28");
    
    total++; passed += verify_hash("127 bytes", data, 127, nullptr, 0,
        "b6292669ccd38d5f01caae96ba272c76a879a45743afa0725d83b9ebb26665b731f1848c52f11972b6644f554c064fa90780dbbbf3a89d4fc31f67df3e5857ef");
    
    total++; passed += verify_hash("128 bytes", data, 128, nullptr, 0,
        "2319e3789c47e2daa5fe807f61bec2a1a6537fa03f19ff32e87eecbfd64b7e0e8ccff439ac333b040f19b0c4ddd11a61e24ac1fe0f10a039806c5dcc0da3d115");
    
    total++; passed += verify_hash("129 bytes", data, 129, nullptr, 0,
        "f59711d44a031d5f97a9413c065d1e614c417ede998590325f49bad2fd444d3e4418be19aec4e11449ac1a57207898bc57d76a1bcf3566292c20c683a5c4648f");
    
    // Keyed hashing tests
    printf("\n--- Keyed Hashing Tests ---\n");
    
    uint8_t key16[16];
    for (int i = 0; i < 16; i++) key16[i] = (uint8_t)i;
    
    uint8_t key32[32];
    for (int i = 0; i < 32; i++) key32[i] = (uint8_t)i;
    
    uint8_t key64[64];
    for (int i = 0; i < 64; i++) key64[i] = (uint8_t)(i * 2);
    
    uint8_t keyed_data[128];
    for (int i = 0; i < 128; i++) keyed_data[i] = (uint8_t)(i ^ 0xAA);
    
    total++; passed += verify_hash("16-byte key, empty data", nullptr, 0, key16, 16,
        "be9ba4d16e01a2bcdfad4ae51984e0bb9fe88a5e34da8387ffd1152bd0bbb7d9118bb368a6e6ddd8dd0adf46d58fcd368b744ca8452c8c20d225eb3741f307bb");
    
    total++; passed += verify_hash("32-byte key, 64 bytes data", keyed_data, 64, key32, 32,
        "fd50fb052eeb7068452ff39633e6429c2e7a7e28c92782006f18f33434fc84400f4c35e78417d802cd3624f2ca53afad2980417088e8a4f297178ffdf29c2568");
    
    total++; passed += verify_hash("64-byte key, 128 bytes data", keyed_data, 128, key64, 64,
        "ddb41c807f51f28dbcb45f3a6093b7bb3096d095b13a54e4467b6ff206460e720f2d7a9007306d9119ae20f82cde979404134b4ee409f03916038789a4397653");
    
    // Large input test
    printf("\n--- Stress Test ---\n");
    
    uint8_t* large_data = new uint8_t[1024 * 1024];
    for (int i = 0; i < 1024 * 1024; i++) {
        large_data[i] = (uint8_t)(i & 0xFF);
    }
    
    total++; passed += verify_hash("1MB input", large_data, 1024 * 1024, nullptr, 0,
        "988b9f48e713f594ab9d5bbac2de5f1f04714169d0ff806844089dfed9d7dcad92d90e5c7d9d0676042e059ab231b6839113f2d2b8abf631ee0a49c593d4491d");
    
    delete[] large_data;
    
    // Summary
    printf("\n=== Summary ===\n");
    printf("Passed: %d/%d\n", passed, total);
    
    if (passed == total) {
        printf("✅ All verification tests passed!\n");
        printf("   CUDA implementation matches Python hashlib.blake2b exactly.\n");
        return 0;
    } else {
        printf("❌ Some verification tests failed\n");
        return 1;
    }
}
