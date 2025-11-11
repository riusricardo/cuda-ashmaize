/**
 * GPU AshMaize Benchmark
 * 
 * Measures GPU mining throughput for various batch sizes
 */

#include "cuda/common.cuh"
#include "cuda/kernel.cu"
#include "cuda/vm.cuh"
#include "cuda/argon2.cuh"
#include "cuda/blake2b.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <algorithm>

// Helper: Create ROM texture from device memory
cudaTextureObject_t create_rom_texture_from_device(const uint8_t* d_rom_data, size_t rom_size) {
    // Create resource descriptor for linear memory
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = (void*)d_rom_data;
    resDesc.res.linear.desc = cudaCreateChannelDesc<uint8_t>();
    resDesc.res.linear.sizeInBytes = rom_size;
    
    // Create texture descriptor
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    
    // Create texture object
    cudaTextureObject_t texObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
    
    return texObj;
}

void destroy_rom_texture(cudaTextureObject_t texObj) {
    CUDA_CHECK(cudaDestroyTextureObject(texObj));
}

// Generate simple ROM for benchmarking
void generate_simple_rom(uint8_t* rom, size_t rom_size, const uint8_t* seed, size_t seed_size) {
    Blake2bState digest;
    blake2b_init(&digest, nullptr, 0);
    
    uint32_t size_le = rom_size;
    blake2b_update(&digest, (const uint8_t*)&size_le, 4);
    blake2b_update(&digest, seed, seed_size);
    
    uint8_t seed_digest[64];
    blake2b_final(&digest, seed_digest);
    
    argon2_hprime(rom, rom_size, seed_digest, 32);
}

void generate_rom_digest(uint8_t* rom_digest, const uint8_t* rom, size_t rom_size) {
    Blake2bState digest;
    blake2b_init(&digest, nullptr, 0);
    blake2b_update(&digest, rom, rom_size);
    blake2b_final(&digest, rom_digest);
}

// Benchmark kernel for batch processing
__global__ void benchmark_kernel(
    const uint8_t* rom_digest,
    uint64_t salt_base,
    uint32_t batch_size,
    uint32_t nb_loops,
    uint32_t nb_instrs,
    cudaTextureObject_t rom_texture,
    uint8_t* outputs
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;
    
    // Generate salt for this thread
    uint64_t salt_value = salt_base + tid;
    uint8_t salt[8];
    for (int i = 0; i < 8; i++) {
        salt[i] = (salt_value >> (i * 8)) & 0xFF;
    }
    
    // Initialize VM
    VM vm;
    vm_init(&vm, rom_digest, salt, 8, nb_instrs);
    
    // Execute nb_loops iterations
    for (uint32_t loop = 0; loop < nb_loops; loop++) {
        vm_execute(&vm, rom_texture, nb_instrs);
    }
    
    // Finalize and write output
    uint8_t* output = &outputs[tid * 64];
    vm_finalize(&vm, output);
}

struct BenchmarkResult {
    double elapsed_ms;
    double throughput_hash_per_sec;
    size_t batch_size;
    int block_size;
    int grid_size;
};

BenchmarkResult benchmark_batch(
    const uint8_t* d_rom_digest,
    cudaTextureObject_t rom_tex,
    size_t batch_size,
    int block_size,
    uint32_t nb_loops,
    uint32_t nb_instrs
) {
    // Allocate device memory
    uint8_t* d_outputs;
    CUDA_CHECK(cudaMalloc(&d_outputs, batch_size * 64));
    
    int grid_size = (batch_size + block_size - 1) / block_size;
    
    // Warm-up run
    benchmark_kernel<<<grid_size, block_size>>>(
        d_rom_digest, 0, batch_size, nb_loops, nb_instrs, rom_tex, d_outputs
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    
    benchmark_kernel<<<grid_size, block_size>>>(
        d_rom_digest, 0, batch_size, nb_loops, nb_instrs, rom_tex, d_outputs
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    
    double elapsed_ms = elapsed.count();
    double throughput = (batch_size * 1000.0) / elapsed_ms;
    
    CUDA_CHECK(cudaFree(d_outputs));
    
    return BenchmarkResult{
        elapsed_ms,
        throughput,
        batch_size,
        block_size,
        grid_size
    };
}

void print_device_info() {
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("GPU Device: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Shared memory per block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("  Warp size: %d\n", prop.warpSize);
    printf("\n");
}

int main(int argc, char** argv) {
    printf("=== AshMaize GPU Performance Benchmark ===\n\n");
    
    print_device_info();
    
    // Parse command line arguments
    size_t rom_size = 64 * 1024 * 1024;  // 64 MB default
    uint32_t nb_loops = 8;
    uint32_t nb_instrs = 256;
    
    if (argc > 1) {
        rom_size = std::strtoull(argv[1], nullptr, 10) * 1024 * 1024;  // Convert MB to bytes
    }
    if (argc > 2) {
        nb_loops = std::strtoul(argv[2], nullptr, 10);
    }
    if (argc > 3) {
        nb_instrs = std::strtoul(argv[3], nullptr, 10);
    }
    
    printf("Parameters:\n");
    printf("  ROM size: %.2f MB\n", rom_size / (1024.0 * 1024.0));
    printf("  Loops: %u\n", nb_loops);
    printf("  Instructions per loop: %u\n", nb_instrs);
    printf("\n");
    
    // Generate ROM
    printf("Generating ROM...\n");
    uint8_t* rom = new uint8_t[rom_size];
    const uint8_t seed[] = "benchmark_seed";
    generate_simple_rom(rom, rom_size, seed, sizeof(seed) - 1);
    
    uint8_t rom_digest[64];
    generate_rom_digest(rom_digest, rom, rom_size);
    
    // Upload ROM
    printf("Uploading ROM to GPU...\n");
    uint8_t* d_rom;
    CUDA_CHECK(cudaMalloc(&d_rom, rom_size));
    CUDA_CHECK(cudaMemcpy(d_rom, rom, rom_size, cudaMemcpyHostToDevice));
    
    cudaTextureObject_t rom_tex = create_rom_texture_from_device(d_rom, rom_size);
    
    uint8_t* d_rom_digest;
    CUDA_CHECK(cudaMalloc(&d_rom_digest, 64));
    CUDA_CHECK(cudaMemcpy(d_rom_digest, rom_digest, 64, cudaMemcpyHostToDevice));
    
    printf("\n=== Running Benchmarks ===\n\n");
    
    // Test different batch sizes and block sizes
    struct TestConfig {
        size_t batch_size;
        int block_size;
        const char* description;
    };
    
    TestConfig configs[] = {
        {256, 256, "Small batch (1 block)"},
        {1024, 256, "Medium batch (4 blocks)"},
        {4096, 256, "Large batch (16 blocks)"},
        {16384, 256, "Very large batch (64 blocks)"},
        {65536, 256, "Huge batch (256 blocks)"},
        {262144, 256, "Massive batch (1024 blocks)"},
        
        // Test different block sizes with same batch
        {16384, 128, "16K batch, 128 threads/block"},
        {16384, 256, "16K batch, 256 threads/block"},
        {16384, 512, "16K batch, 512 threads/block"},
    };
    
    printf("%-40s %10s %10s %15s %8s\n", 
           "Configuration", "Batch", "Time(ms)", "Hash/sec", "Speedup");
    printf("%s\n", std::string(90, '-').c_str());
    
    double baseline_throughput = 0;
    
    for (const auto& config : configs) {
        auto result = benchmark_batch(
            d_rom_digest, rom_tex, config.batch_size, config.block_size,
            nb_loops, nb_instrs
        );
        
        if (baseline_throughput == 0) {
            baseline_throughput = result.throughput_hash_per_sec;
        }
        
        double speedup = result.throughput_hash_per_sec / baseline_throughput;
        
        printf("%-40s %10zu %10.2f %15.2f %8.2fx\n",
               config.description,
               config.batch_size,
               result.elapsed_ms,
               result.throughput_hash_per_sec,
               speedup);
    }
    
    printf("\n");
    
    // Run extended benchmark for best configuration
    printf("=== Extended Benchmark (Best Configuration) ===\n\n");
    
    size_t best_batch = 65536;
    int best_block = 256;
    int num_iterations = 10;
    
    printf("Running %d iterations with batch size %zu...\n", num_iterations, best_batch);
    
    double total_throughput = 0;
    double min_throughput = 1e9;
    double max_throughput = 0;
    
    for (int i = 0; i < num_iterations; i++) {
        auto result = benchmark_batch(
            d_rom_digest, rom_tex, best_batch, best_block,
            nb_loops, nb_instrs
        );
        
        total_throughput += result.throughput_hash_per_sec;
        min_throughput = std::min(min_throughput, result.throughput_hash_per_sec);
        max_throughput = std::max(max_throughput, result.throughput_hash_per_sec);
        
        printf("  Iteration %2d: %10.2f hash/sec (%.2f ms)\n", 
               i + 1, result.throughput_hash_per_sec, result.elapsed_ms);
    }
    
    double avg_throughput = total_throughput / num_iterations;
    
    printf("\n");
    printf("Results:\n");
    printf("  Average: %.2f hash/sec\n", avg_throughput);
    printf("  Min:     %.2f hash/sec\n", min_throughput);
    printf("  Max:     %.2f hash/sec\n", max_throughput);
    printf("  Std dev: %.2f%%\n", 
           100.0 * (max_throughput - min_throughput) / avg_throughput);
    
    // Cleanup
    destroy_rom_texture(rom_tex);
    CUDA_CHECK(cudaFree(d_rom));
    CUDA_CHECK(cudaFree(d_rom_digest));
    delete[] rom;
    
    printf("\n=== Benchmark Complete ===\n");
    printf("Best throughput: %.2f hashes/second\n", max_throughput);
    
    return 0;
}
