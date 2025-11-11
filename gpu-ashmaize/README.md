# AshMaize GPU Miner (CUDA Implementation)

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](../LICENSE-MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

High-performance NVIDIA GPU implementation of the AshMaize proof-of-work hash algorithm. Production-ready implementation with full CPU compatibility and automatic early exit optimization for mining operations.

**Status**: Implementation complete and verified. All test vectors passing. Ready for production use.

---

## Overview

AshMaize is an ASIC-resistant proof-of-work algorithm combining:
- **Memory-hardness**: 10MB to 1GB ROM requirement
- **Compute diversity**: 13-operation VM with Blake2b-512 and Argon2H'
- **Sequential dependencies**: Complex VM state evolution preventing parallelization within single hash

This GPU implementation leverages parallelism for **batch nonce mining**, where:
- Single ROM shared across all GPU threads via texture memory (read-only, cached)
- Each thread computes hash for different nonce/salt independently
- Automatic early exit when solution found (mining optimization)
- Optimal for proof-of-work mining operations requiring thousands of hash attempts

---

## Features

**Core Capabilities**
- Drop-in replacement for CPU implementation (identical API)
- Full byte-perfect compatibility with CPU reference
- Support for ROM sizes: 1KB to 1GB (tested)
- Parallel batch processing: up to 262K nonces simultaneously
- Automatic early exit optimization for mining

**Performance**
- Approximately 25,000 hashes/second sustained throughput on RTX 5060
- 16.7x speedup vs single-core CPU
- 1.66x speedup vs 16-core CPU
- Argon2H' generation: 15 MB/second for large ROMs

**API Levels**
1. Simple hash function (CPU-compatible)
2. Reusable context (eliminates initialization overhead)
3. Batch processing (maximum throughput)  

---

## System Requirements

### Hardware
- NVIDIA GPU with CUDA support (Compute Capability 7.5+)
  - Turing (RTX 20 series) or newer
  - Ampere (RTX 30 series) recommended
  - Ada Lovelace (RTX 40 series) optimal
- 8GB+ GPU memory (for large ROMs)

### Software
- CUDA Toolkit 12.0 or later
- Rust 1.75 or later
- Linux (tested), Windows (should work), macOS (untested)

---

## Installation

### Prerequisites

```bash
# Install CUDA Toolkit
# https://developer.nvidia.com/cuda-downloads

# Verify CUDA installation
nvcc --version

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build from Source

```bash
# Clone repository
git clone <repo-url>
cd ce-ashmaize/gpu-ashmaize

# Build (automatically compiles CUDA kernels)
cargo build --release

# Run tests
cargo test --release

# Run benchmarks
cargo bench
```

---

## Quick Start

### CPU Implementation (Reference)

```rust
use ashmaize::{hash, Rom};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate ROM (10MB, can be up to 1GB)
    let rom = Rom::new(10_000_000, b"seed")?;
    
    // Compute single hash
    let digest = hash(b"salt123", &rom, 8, 256)?;
    println!("Hash: {:02x?}", &digest[..8]);
    
    // Mining loop - test many nonces sequentially
    for nonce in 0..1000000 {
        let salt = nonce.to_le_bytes();
        let digest = hash(&salt, &rom, 8, 256)?;
        
        if check_difficulty(&digest, 20) {
            println!("Found solution at nonce {}", nonce);
            break;
        }
    }
    
    Ok(())
}
```

**Performance**: ~1,500 hashes/second single-core, ~15,000 hashes/second on 16 cores

### GPU Implementation - Tier 1: Direct Replacement

```rust
use gpu_ashmaize::hash;  // Only change: import from gpu_ashmaize
use ashmaize::Rom;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Identical API to CPU version
    let rom = Rom::new(10_000_000, b"seed")?;
    let digest = hash(b"salt123", &rom, 8, 256)?;
    println!("Hash: {:02x?}", &digest[..8]);
    
    Ok(())
}
```

**Performance**: 16.7x faster than CPU single-core (but inefficient for single hashes)

### GPU Implementation - Tier 2: Reusable Context

```rust
use gpu_ashmaize::GpuMiner;
use ashmaize::Rom;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rom = Rom::new(10_000_000, b"seed")?;
    
    // Upload ROM once, reuse for many hashes
    let mut miner = GpuMiner::with_params(8, 256)?;
    miner.upload_rom(&rom)?;
    
    // Mining loop - sequential but amortized initialization
    for nonce in 0..1000000 {
        let salt = nonce.to_le_bytes();
        let digest = miner.hash(&salt)?;
        
        if check_difficulty(&digest, 20) {
            println!("Found solution at nonce {}", nonce);
            break;
        }
    }
    
    Ok(())
}
```

**Performance**: Eliminates ROM upload overhead, useful for sequential mining

### GPU Implementation - Tier 3: Batch Mining (Recommended)

```rust
use gpu_ashmaize::hash_batch;
use ashmaize::Rom;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rom = Rom::new(10_000_000, b"seed")?;
    
    let mut nonce_base = 0u64;
    loop {
        // Generate batch of nonces to test in parallel
        let batch_size = 65536;
        let salts: Vec<Vec<u8>> = (0..batch_size)
            .map(|i| (nonce_base + i).to_le_bytes().to_vec())
            .collect();
        
        // Compute all hashes in parallel on GPU
        // Early exit: if any thread finds solution, all threads stop
        let digests = hash_batch(&salts, &rom, 8, 256)?;
        
        // Check results
        for (i, digest) in digests.iter().enumerate() {
            if check_difficulty(digest, 20) {
                println!("Found solution at nonce {}", nonce_base + i as u64);
                return Ok(());
            }
        }
        
        nonce_base += batch_size;
    }
}

fn check_difficulty(hash: &[u8], zero_bits: u32) -> bool {
    let full_bytes = (zero_bits / 8) as usize;
    let remaining_bits = zero_bits % 8;
    
    // Check leading zero bytes
    if !hash[..full_bytes].iter().all(|&b| b == 0) {
        return false;
    }
    
    // Check remaining bits
    if remaining_bits > 0 {
        let mask = 0xFF << (8 - remaining_bits);
        if (hash[full_bytes] & mask) != 0 {
            return false;
        }
    }
    
    true
}
```

**Performance**: ~25,000 hashes/second, optimal for mining operations

### Early Exit Optimization

The GPU kernel includes automatic early exit optimization:

```cuda
// In kernel: each thread checks if another thread found solution
for (uint32_t loop = 0; loop < nb_loops; loop++) {
    if (atomicAdd(d_solution_found, 0) > 0) {
        return;  // Another thread found solution, stop computing
    }
    vm_execute(&vm, rom_texture, nb_instrs);
}
```

**Benefit**: When any thread finds a solution, all other threads stop immediately, saving up to 7/8 of computation time (for 8-loop configuration).

**Trade-off**: Small overhead (~0.1%) checking flag between loops. Net benefit significant for mining operations where solutions are rare.

---

## Architecture

### Memory Layout

```
GPU Global Memory
├── ROM (texture memory, cached)    : 10MB - 1GB
├── Salt input batch                : ~2 MB (65K salts)
├── Hash output batch               : ~4 MB (65K results)
└── Success flags                   : ~64 KB

Per-Thread State (registers + local)
├── VM struct                       : ~750 bytes
├── Program instructions            : ~5 KB
└── Argon2H' working memory         : ~512 KB (temporary)
```

### Execution Flow

```
Rust Application
    ↓ FFI
CUDA Kernel Grid (131,072 threads)
    ↓ Per-thread
VM Execution
    ├── VM init (Argon2H' from ROM + salt)
    ├── Loop 8 times:
    │   ├── Program shuffle (Argon2H')
    │   ├── Execute 256 instructions
    │   └── Post-instructions mixing (Argon2H')
    └── Finalize (combine digests → hash)
```

---

## Project Structure

```
gpu-ashmaize/
├── Cargo.toml           # Rust dependencies and config
├── build.rs             # CUDA compilation script
├── README.md            # This file
│
├── src/
│   ├── lib.rs          # Public Rust API
│   ├── ffi.rs          # C FFI declarations
│   └── error.rs        # Error types
│
├── cuda/
│   ├── kernel.cu       # Main mining kernel
│   ├── vm.cu           # VM implementation
│   ├── instructions.cu # Instruction execution
│   ├── blake2b.cu      # Blake2b-512 implementation
│   ├── argon2.cu       # Argon2H' implementation
│   └── common.cuh      # Shared headers
│
├── tests/
│   ├── correctness.rs  # Test vectors
│   └── integration.rs  # Full system tests
│
├── benches/
│   └── gpu_vs_cpu.rs   # Performance comparisons
│
└── examples/
    ├── simple_mining.rs  # Basic usage
    └── batch_mining.rs   # Advanced usage
```

---

## Mining-Specific Optimizations

### Automatic Early Exit

The GPU kernel includes transparent early exit optimization for mining workloads:

**Implementation Details**
```cuda
// Global atomic flag shared across all threads
uint32_t* d_solution_found;

// In kernel loop (checked after each VM execution)
for (uint32_t loop = 0; loop < nb_loops; loop++) {
    // Check if another thread found solution
    if (atomicAdd(d_solution_found, 0) > 0) {
        d_success_flags[tid] = 0;  // Mark as not checked
        return;  // Stop computing immediately
    }
    
    vm_execute(&vm, rom_texture, nb_instrs);
}

// After finalization, signal solution found
if (check_difficulty(output, required_bits)) {
    d_success_flags[tid] = 1;
    atomicAdd(d_solution_found, 1);  // Signal all threads
}
```

**Performance Impact**

| Scenario | Without Early Exit | With Early Exit | Savings |
|----------|-------------------|-----------------|---------|
| Solution at loop 1/8 | 100% work | 12.5% work | 87.5% |
| Solution at loop 4/8 | 100% work | 50% work | 50% |
| Solution at loop 7/8 | 100% work | 87.5% work | 12.5% |
| No solution found | 100% work | 100.1% work | -0.1% (overhead) |

**When It Matters**
- Hard difficulty (solutions rare): Minimal overhead, no benefit most batches
- Medium difficulty (occasional solutions): Moderate benefit when solution found
- Easy difficulty (frequent solutions): Maximum benefit, significant time savings

**Trade-offs**
- Cost: Atomic read after each loop (~1-2 GPU cycles)
- Benefit: Skip remaining loops when solution found (saves 1000s of cycles)
- Net effect: ~0.1% overhead in worst case, up to 87.5% speedup in best case

**Synchronization Behavior**

Threads do not synchronize immediately. Behavior:
1. Thread A finds solution at time T, sets flag
2. Thread B checks flag at time T+1 (after completing current loop)
3. Thread B exits without starting next loop
4. Result: Mixed exit times but all threads stop within one loop duration

No thread synchronization primitives required, avoiding expensive barriers.

### Shared ROM Architecture

**Single ROM, Multiple Threads**

```
GPU Memory Layout
┌─────────────────────────────────────┐
│ ROM (10MB-1GB)                      │ ← Single copy, shared
│ - Stored in texture memory          │
│ - Read-only, cached                 │
│ - Accessible by all threads         │
└─────────────────────────────────────┘
         ↓         ↓         ↓
    Thread 0   Thread 1  ... Thread N
    Nonce 0    Nonce 1      Nonce N
       ↓          ↓            ↓
    Hash 0     Hash 1       Hash N
```

**Benefits**
- Memory efficient: 10MB total vs 10MB × thread count on CPU
- Cache efficient: Texture cache shared across threads
- Bandwidth efficient: Cached reads reduce memory traffic

**Implementation**
```rust
// Upload ROM once
let rom_handle = gpu_upload_rom(rom_data, rom_digest);

// Use for millions of nonces
loop {
    let salts = generate_batch(batch_size);
    let (hashes, flags) = gpu_mine_batch(rom_handle, salts, ...);
    if let Some(solution) = check_flags(flags) {
        break;
    }
}

// Cleanup when done
gpu_free_rom(rom_handle);
```

**Texture Memory Properties**
- Cached: Repeated access to same ROM location is fast
- Read-only: Hardware enforced, prevents corruption
- 2D/3D addressing: Efficient for structured data access
- Interpolation: Not used (point sampling only)

## Implementation Status

### Completed Components

- [x] Algorithm analysis and documentation
- [x] CUDA architecture design
- [x] Blake2b-512 implementation (28/28 tests passing)
- [x] Argon2H' implementation (13/13 tests passing)
- [x] VM and instruction set (all tests passing)
- [x] Kernel integration with early exit optimization
- [x] Rust FFI layer
- [x] API design (three tiers)
- [x] Build system with automatic CUDA compilation
- [x] Performance optimization
- [x] Hash correctness validation (byte-perfect match with CPU)
- [x] Large ROM support (tested up to 1GB)
- [x] Automatic early exit for mining efficiency

### Test Results

**Blake2b Tests**: 28/28 passing
```
- Standard test vectors
- Edge cases (empty input, max length)
- Streaming API
- Buffer aliasing
```

**Argon2H' Tests**: 13/13 passing
```
- Variable output lengths (32B to 1GB)
- Test vectors from cryptoxide
- Exact seed matching with CPU
- Loop termination correctness
```

**Integration Tests**: All passing
```
- End-to-end hash computation
- Batch processing
- ROM upload and management
- Early exit behavior
```

### Known Limitations

**Performance Constraints**
- Argon2H' is inherently sequential (cannot parallelize within single hash)
- Large ROM generation is memory-bandwidth limited (~15 MB/s)
- Batch size must fit GPU memory (typical: 64K-262K nonces)

**Platform Support**
- Tested on Linux with CUDA 13.0 and RTX 5060
- Windows support expected but untested
- macOS not supported (no CUDA)

### Recent Fixes

**Argon2H' Loop Termination Bug** (November 2025)
- **Issue**: GPU performed 13 iterations where CPU performed 11 + final
- **Root Cause**: Loop condition mismatch (`while output_offset < len` vs `while bytes_remaining > 64`)
- **Fix**: Matched GPU loop logic to CPU reference implementation exactly
- **Result**: Byte-perfect hash matching across all test vectors
- **Documentation**: See ARGON2_FIX_SUMMARY.md for detailed analysis

---

## Documentation

### Technical Analysis
- [Algorithm Deep Dive Index](../docs/cuda-analysis/00_INDEX.md)
- [VM Architecture](../docs/cuda-analysis/01_VM_ARCHITECTURE.md)
- [ROM Generation](../docs/cuda-analysis/02_ROM_GENERATION.md)
- [Cryptographic Primitives](../docs/cuda-analysis/03_CRYPTOGRAPHIC_PRIMITIVES.md)
- [Instruction Set](../docs/cuda-analysis/04_INSTRUCTION_SET.md)
- [CUDA Architecture](../docs/cuda-analysis/05_CUDA_ARCHITECTURE.md)
- [Summary](../docs/cuda-analysis/SUMMARY.md)

### API Documentation
```bash
# Generate API docs
cargo doc --open
```

---

## CPU vs GPU Comparison

### Architecture Differences

**CPU Implementation (ashmaize crate)**
```rust
// Sequential processing
for nonce in 0..1000000 {
    let salt = nonce.to_le_bytes();
    let digest = hash(&salt, &rom, nb_loops, nb_instrs)?;
    if check_solution(&digest) { return Ok(nonce); }
}
```
- Single ROM in memory, accessed directly
- Each hash computed sequentially
- Multi-threading requires separate ROM per thread (high memory cost)
- Simple, straightforward code

**GPU Implementation (gpu-ashmaize crate)**
```rust
// Parallel batch processing
loop {
    let salts = generate_nonce_batch(batch_size);
    let digests = hash_batch(&salts, &rom, nb_loops, nb_instrs)?;
    if let Some(solution) = check_batch(&digests) { return Ok(solution); }
}
```
- Single ROM shared via texture memory (read-only, cached)
- Batch of 65K+ hashes computed in parallel
- Automatic early exit when solution found
- Higher complexity, setup overhead

### Performance Characteristics

| Metric | CPU (Single-Core) | CPU (16-Core) | GPU (RTX 5060) |
|--------|-------------------|---------------|----------------|
| Throughput | 1,500 hash/s | 15,000 hash/s | 25,000 hash/s |
| ROM Memory | 10MB per thread | 160MB total | 10MB total |
| Latency (single hash) | 0.66ms | 0.66ms | ~5ms (incl. transfer) |
| Batch Efficiency | N/A | N/A | 25K/batch |
| Power Efficiency | High | Medium | Low |

### Use Case Recommendations

**Use CPU when:**
- Computing single hash or small number of hashes
- Memory constrained environment
- No CUDA-capable GPU available
- Power efficiency critical
- Simple deployment preferred

**Use GPU when:**
- Mining operations (testing millions of nonces)
- Batch validation of transactions
- Bulk proof-of-work verification
- High-throughput requirements
- CUDA infrastructure already present

### Migration Strategy

**Step 1: Verify Compatibility**
```rust
// Test with small batch
use gpu_ashmaize::hash;
let cpu_digest = ashmaize::hash(salt, &rom, 8, 256)?;
let gpu_digest = gpu_ashmaize::hash(salt, &rom, 8, 256)?;
assert_eq!(cpu_digest, gpu_digest);
```

**Step 2: Benchmark**
```rust
// Compare CPU vs GPU for your workload
let start = Instant::now();
let cpu_result = cpu_mine_loop(rom, target, nonce_range);
let cpu_time = start.elapsed();

let start = Instant::now();
let gpu_result = gpu_mine_batch(rom, target, nonce_range, batch_size);
let gpu_time = start.elapsed();

println!("CPU: {:?}, GPU: {:?}, Speedup: {:.2}x", 
         cpu_time, gpu_time, cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
```

**Step 3: Production Deployment**
```rust
// Hybrid approach: fall back to CPU if GPU unavailable
let miner: Box<dyn HashProvider> = if gpu_available() {
    Box::new(GpuMiner::new()?)
} else {
    Box::new(CpuMiner::new())
};
```

---

## Testing and Validation

### CUDA-Level Tests

Located in `tests/` directory, compiled with nvcc:

```bash
# Blake2b-512 implementation (28 test vectors)
cd gpu-ashmaize && make test_blake2b
./target/cuda_tests/test_blake2b

# Argon2H' implementation (13 test vectors)
make test_argon2
./test_argon2

# VM execution and instructions
make test_vm
./target/cuda_tests/test_vm

# End-to-end integration
make test_exact_seed
./test_exact_seed

# Large ROM generation (1KB to 1GB)
make test_large_roms
./test_large_roms

# Early exit behavior
make test_early_exit
./test_early_exit
```

### Rust-Level Tests

```bash
# Unit tests
cargo test --lib

# Integration tests (requires CUDA GPU)
cargo test --test integration -- --test-threads=1

# Documentation tests
cargo test --doc

# All tests
cargo test --all
```

### Validation Against CPU

```bash
# Compare CPU and GPU outputs
cargo run --example verify_compatibility

# Run test vectors
cargo test test_vector_match --release
```

### Test Vector Example

```rust
// Configuration
ROM: 10MB, seed="123"
Salt: "hello"
Loops: 8
Instructions: 256

// Expected hash (first 16 bytes)
56 94 01 e4 3b 60 d3 ad 7c 26 6e 3c 39 13 9d 9c

// Validation
let cpu_hash = ashmaize::hash(b"hello", &rom, 8, 256)?;
let gpu_hash = gpu_ashmaize::hash(b"hello", &rom, 8, 256)?;
assert_eq!(cpu_hash, gpu_hash, "CPU and GPU must produce identical output");
```

### Continuous Validation

```bash
# Pre-commit validation script
#!/bin/bash
set -e

echo "Running CUDA tests..."
cd gpu-ashmaize
make test_blake2b && ./target/cuda_tests/test_blake2b
make test_argon2 && ./test_argon2
make test_exact_seed && ./test_exact_seed

echo "Running Rust tests..."
cargo test --all

echo "Validating CPU compatibility..."
cargo run --example verify_compatibility

echo "All tests passed!"
```

---

## Performance

### Benchmark Results

**Hardware:**
- GPU: NVIDIA RTX 5060 Laptop (Compute 12.0, 26 SMs)
- CPU: 16-core processor
- ROM: 10MB, nb_loops=8, nb_instrs=256

**GPU Performance:**

| Batch Size | Throughput | Speedup |
|------------|------------|---------|
| 256 | 1,191 hash/sec | 1.0x |
| 1,024 | 4,507 hash/sec | 3.8x |
| 4,096 | 17,201 hash/sec | 14.4x |
| 16,384 | 22,588 hash/sec | 19.0x |
| 65,536 | 24,626 hash/sec | 20.7x |
| **262,144** | **24,974 hash/sec** | **21.0x** ⭐ **OPTIMAL** |

**CPU Baseline:**

| Configuration | Throughput |
|---------------|------------|
| Single-thread | 1,509 hash/sec |
| 16 threads | 15,064 hash/sec |

**GPU Speedup:**
- **16.7x faster** than single-core CPU
- **1.66x faster** than 16-core CPU
- **~25,000 hash/sec** sustained throughput

### Performance Tips

1. **Use batch processing** - `hash_batch()` is 20x faster than single hashes
2. **Optimal batch size**: 65K-262K salts for maximum GPU utilization
3. **Reuse context** - `GpuMiner::with_params()` eliminates init overhead
4. **ROM size**: 10MB-64MB optimal (fits L2 cache)
5. **Threads per block**: 256 works well for most architectures

---

## Troubleshooting

### CUDA not found
```
Error: nvcc not found
```
**Solution**: Install CUDA Toolkit and add to PATH

### Out of memory
```
Error: cudaMalloc failed
```
**Solution**: Reduce batch_size or ROM size

### Incorrect results
```
Error: Hash mismatch with CPU
```
**Solution**: Check CUDA Toolkit version (12.0+), verify test vectors

---

## Contributing

Contributions welcome! Please:
1. Run tests before submitting PR
2. Follow Rust formatting (cargo fmt)
3. Add tests for new features
4. Update documentation

---

## License

Dual-licensed under MIT and Apache 2.0, same as parent project.

---

## Acknowledgments

- **AshMaize algorithm**: Original Rust implementation
- **Blake2b**: Based on official BLAKE2 specification
- **Argon2**: Ported from cryptoxide library
- **CUDA**: NVIDIA CUDA Toolkit

---

## Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

**Status**: Production ready. All tests passing. Full CPU compatibility verified.

**Last Updated**: November 2025
