# AshMaize GPU Implementation# AshMaize GPU Implementation



CUDA implementation of AshMaize with byte-perfect CPU compatibility.High-performance CUDA implementation of the AshMaize proof-of-work hash algorithm with byte-perfect CPU compatibility.



## Overview**Status**: Production ready. All tests passing. CPU/GPU hash equivalence verified.



Parallel batch nonce mining optimized for NVIDIA GPUs:## Overview



- Single ROM shared via texture memory (cached, read-only)CUDA implementation of AshMaize optimized for parallel batch nonce mining:

- Each thread computes independent hash (different salt/nonce)

- No inter-thread communication required- Single ROM shared via texture memory (cached, read-only)

- Optimal for proof-of-work mining workloads- Each thread computes independent hash (different salt/nonce)

- No inter-thread communication required

**Key Characteristics:**- Optimal for proof-of-work mining workloads

- Sequential bottleneck: Argon2H' (64% of time, cannot parallelize within hash)

- Parallelization: Thread-level (many independent hashes)**Key Characteristics:**

- Memory efficient: Single ROM shared across all threads- Sequential bottleneck: Argon2H' (64% of execution time, cannot parallelize within hash)

- Parallelization strategy: Many independent hash computations

## Features- Memory efficient: Single 256MB ROM shared across all threads vs 256MB per CPU core



- Drop-in replacement for CPU `hash()` function## Features

- Byte-perfect compatibility with CPU

- ROM sizes: 64 bytes to 1GB- Drop-in replacement for CPU `hash()` function (identical signature)

- Batch processing: up to 262K salts per launch- Byte-perfect compatibility with CPU implementation

- Three API tiers: simple, reusable, batch- ROM sizes: 64 bytes to 1GB (tested)

- Batch processing: up to 262K salts per kernel launch

**Performance (RTX 5060, 256MB ROM, 8 loops, 256 instrs):**- Three API tiers: simple, reusable context, batch processing

- Single hash: ~370ms (includes upload)

- Batch: ~25,000 hash/sec**Performance (RTX 5060, 256MB ROM, 8 loops, 256 instructions):**

- Speedup: 16-18x vs single-core CPU- Single hash: ~370ms (includes upload overhead)

- Batch throughput: ~25,000 hash/sec

## Requirements- Speedup: 16-18x vs single-core CPU, 1.15x vs 16-core CPU



**Hardware:**## Requirements

- NVIDIA GPU with Compute Capability 7.5+

- 8GB+ GPU memory recommended**Hardware:**

- NVIDIA GPU with Compute Capability 7.5+ (Turing/Ampere/Ada)

**Software:**- 8GB+ GPU memory (for large ROMs)

- CUDA Toolkit 12.0+

- Rust 1.75+**Software:**

- nvcc in PATH- CUDA Toolkit 12.0+

- Rust 1.75+

## Installation- nvcc compiler in PATH

- Linux (tested), Windows/macOS (untested)

```bash

# Verify CUDA## Installation

nvcc --version

```bash

# Build# Prerequisites: CUDA Toolkit 12.0+, nvcc in PATH

cd gpu-ashmaizenvcc --version

cargo build --release

# Build

# Testcd gpu-ashmaize

cargo test --releasecargo build --release

cargo run --release --example minimal_test

```# Run tests

cargo test --release

## Usagecargo run --release --example minimal_test

```

### API Tier 1: Drop-in Replacement

## Usage

```rust

use gpu_ashmaize::hash;### API Tier 1: Drop-in Replacement

use ashmaize::{Rom, RomGenerationType};

```rust

const MB: usize = 1024 * 1024;use gpu_ashmaize::hash;  // Only change from CPU version

use ashmaize::{Rom, RomGenerationType};

let rom = Rom::new(

    b"seed",const MB: usize = 1024 * 1024;

    RomGenerationType::TwoStep {

        pre_size: 16 * 1024,let rom = Rom::new(

        mixing_numbers: 4    b"seed",

    },    RomGenerationType::TwoStep { pre_size: 16*1024, mixing_numbers: 4 },

    256 * MB    256 * MB

););



let digest = hash(b"nonce", &rom, 8, 256);let digest = hash(b"nonce", &rom, 8, 256);

``````



Identical to CPU `ashmaize::hash()`. Includes ROM upload overhead per call.Identical signature to CPU `ashmaize::hash()`. Includes ROM upload overhead per call.



### API Tier 2: Reusable Context### API Tier 2: Reusable Context



```rust```rust

use gpu_ashmaize::GpuMiner;use gpu_ashmaize::GpuMiner;

use ashmaize::{Rom, RomGenerationType};use ashmaize::{Rom, RomGenerationType};



let rom = Rom::new(b"seed", RomGenerationType::TwoStep { ... }, 256*MB);let rom = Rom::new(b"seed", RomGenerationType::TwoStep { ... }, 256*MB);



let mut miner = GpuMiner::with_params(8, 256)?;let mut miner = GpuMiner::with_params(8, 256)?;

miner.upload_rom(&rom)?;miner.upload_rom(&rom)?;  // Upload once



for nonce in 0..1_000_000 {for nonce in 0..1_000_000 {

    let salt = nonce.to_le_bytes();    let salt = nonce.to_le_bytes();

    let digest = miner.hash(&salt)?;    let digest = miner.hash(&salt)?;

    // Check solution...    // Check solution...

}}

``````



Amortizes ROM upload across many computations.Amortizes ROM upload cost across many hash computations.



### API Tier 3: Batch Processing (Recommended)### API Tier 3: Batch Processing (Recommended)



```rust```rust

use gpu_ashmaize::hash_batch;use gpu_ashmaize::hash_batch;

use ashmaize::{Rom, RomGenerationType};use ashmaize::{Rom, RomGenerationType};



let rom = Rom::new(b"seed", RomGenerationType::TwoStep { ... }, 256*MB);let rom = Rom::new(b"seed", RomGenerationType::TwoStep { ... }, 256*MB);



let salts: Vec<Vec<u8>> = (0..65536)// Generate 65K nonces

    .map(|i| i.to_le_bytes().to_vec())let salts: Vec<Vec<u8>> = (0..65536)

    .collect();    .map(|i| i.to_le_bytes().to_vec())

    .collect();

let digests = hash_batch(&salts, &rom, 8, 256)?;

// Compute all in parallel

for (i, digest) in digests.iter().enumerate() {let digests = hash_batch(&salts, &rom, 8, 256)?;

    // Process result...

}for (i, digest) in digests.iter().enumerate() {

```    // Check each result...

}

Maximum throughput for mining operations.```



## ArchitectureMaximum throughput (~25,000 hash/sec on RTX 5060).



### Memory Layout### Early Exit Optimization



```The GPU kernel includes automatic early exit optimization:

GPU Global Memory:

  ROM: Texture memory (256MB, cached, shared)```cuda

  Input: Flattened salts (~2MB for 65K)// In kernel: each thread checks if another thread found solution

  Output: Hash results (~4MB for 65K)for (uint32_t loop = 0; loop < nb_loops; loop++) {

    if (atomicAdd(d_solution_found, 0) > 0) {

Per-Thread (registers + local):        return;  // Another thread found solution, stop computing

  VM registers: 32 x 64-bit    }

  Blake2b contexts: 2 x 232 bytes    vm_execute(&vm, rom_texture, nb_instrs);

  Program buffer: 5KB (256 instrs)}

  Argon2H' working: ~512KB temporary```

  Total: ~518KB per thread

```**Benefit**: When any thread finds a solution, all other threads stop immediately, saving up to 7/8 of computation time (for 8-loop configuration).



### Execution Flow**Trade-off**: Small overhead (~0.1%) checking flag between loops. Net benefit significant for mining operations where solutions are rare.



```---

Rust → FFI → CUDA Kernel Grid

              ↓ Per-thread## Architecture

              VM Execution:

                1. Init (Argon2H')### Memory Layout

                2. Loop 8x:

                   - Shuffle program (Argon2H')```

                   - Execute 256 instructionsGPU Global Memory

                   - Post-mixing (Argon2H')├── ROM (texture memory, cached)    : 10MB - 1GB

                3. Finalize → 64-byte hash├── Salt input batch                : ~2 MB (65K salts)

```├── Hash output batch               : ~4 MB (65K results)

└── Success flags                   : ~64 KB

### Parallelization

Per-Thread State (registers + local)

- Each thread: independent hash├── VM struct                       : ~750 bytes

- No synchronization required├── Program instructions            : ~5 KB

- Single ROM shared (texture memory)└── Argon2H' working memory         : ~512 KB (temporary)

- Bottleneck: Sequential Argon2H' (accept and compensate with thread parallelism)```



## Project Structure### Execution Flow



``````

gpu-ashmaize/Rust Application

├── src/    ↓ FFI

│   ├── lib.rs          # Public APICUDA Kernel Grid (131,072 threads)

│   ├── ffi.rs          # CUDA FFI    ↓ Per-thread

│   └── error.rs        # Error typesVM Execution

├── cuda/    ├── VM init (Argon2H' from ROM + salt)

│   ├── blake2b.cu/cuh  # Blake2b-512    ├── Loop 8 times:

│   ├── argon2.cu/cuh   # Argon2H'    │   ├── Program shuffle (Argon2H')

│   ├── vm.cu/cuh       # VM state    │   ├── Execute 256 instructions

│   ├── instructions.cu/cuh # Execution    │   └── Post-instructions mixing (Argon2H')

│   └── kernel.cu/cuh   # Main kernel    └── Finalize (combine digests → hash)

├── examples/           # Usage examples```

├── tests/              # CUDA tests (*.cu)

└── build.rs            # CUDA compilation---

```

## Project Structure

## Testing

```

### CUDA Unit Testsgpu-ashmaize/

├── Cargo.toml           # Rust dependencies and config

```bash├── build.rs             # CUDA compilation script

cd gpu-ashmaize├── README.md            # This file

│

# All CUDA tests├── src/

make test│   ├── lib.rs          # Public Rust API

│   ├── ffi.rs          # C FFI declarations

# Individual primitives│   └── error.rs        # Error types

make test-blake2b   # 44/44 passing│

make test-argon2    # 13/13 passing├── cuda/

```│   ├── kernel.cu       # Main mining kernel

│   ├── vm.cu           # VM implementation

### Rust Integration Tests│   ├── instructions.cu # Instruction execution

│   ├── blake2b.cu      # Blake2b-512 implementation

```bash│   ├── argon2.cu       # Argon2H' implementation

# Quick verification│   └── common.cuh      # Shared headers

cargo run --release --example minimal_test│

├── tests/

# Systematic validation│   ├── correctness.rs  # Test vectors

cargo run --release --example systematic_debug│   └── integration.rs  # Full system tests

│

# Large ROM stress test (256MB, 512MB, 1GB)├── benches/

cargo run --release --example test_large_roms│   └── gpu_vs_cpu.rs   # Performance comparisons

│

# CPU/GPU equivalence└── examples/

cargo test --release    ├── simple_mining.rs  # Basic usage

```    └── batch_mining.rs   # Advanced usage

```

### Test Results

---

**Blake2b-512:** 44/44 tests passing

- RFC 7693 test vectors## Mining-Specific Optimizations

- Incremental hashing

- Context cloning### Automatic Early Exit



**Argon2H':** 13/13 tests passingThe GPU kernel includes transparent early exit optimization for mining workloads:

- Variable-length output

- Byte-perfect with cryptoxide**Implementation Details**

```cuda

**VM Execution:** All passing// Global atomic flag shared across all threads

- ROM: 64KB to 1GBuint32_t* d_solution_found;

- CPU/GPU hash equivalence verified

- Multiple parameter combinations// In kernel loop (checked after each VM execution)

for (uint32_t loop = 0; loop < nb_loops; loop++) {

## Performance    // Check if another thread found solution

    if (atomicAdd(d_solution_found, 0) > 0) {

### Benchmark Results (RTX 5060, 256MB ROM, 8 loops, 256 instrs)        d_success_flags[tid] = 0;  // Mark as not checked

        return;  // Stop computing immediately

**ROM Generation (one-time):**    }

- TwoStep: ~430ms    

- Upload to GPU: ~50ms    vm_execute(&vm, rom_texture, nb_instrs);

}

**Hash Computation:**

// After finalization, signal solution found

| Configuration | Throughput | Notes |if (check_difficulty(output, required_bits)) {

|---------------|------------|-------|    d_success_flags[tid] = 1;

| CPU single-core | ~1,370 hash/sec | Reference |    atomicAdd(d_solution_found, 1);  // Signal all threads

| CPU 16-core | ~21,700 hash/sec | Parallel |}

| GPU single hash | ~370ms | Includes upload overhead |```

| GPU batch 1K | ~4,500 hash/sec | Small batch |

| GPU batch 64K | ~25,000 hash/sec | Optimal |**Performance Impact**



**Speedup:**| Scenario | Without Early Exit | With Early Exit | Savings |

- 16-18x vs single-core CPU|----------|-------------------|-----------------|---------|

- 1.15x vs 16-core CPU| Solution at loop 1/8 | 100% work | 12.5% work | 87.5% |

- Best for large batches| Solution at loop 4/8 | 100% work | 50% work | 50% |

| Solution at loop 7/8 | 100% work | 87.5% work | 12.5% |

### Performance Notes| No solution found | 100% work | 100.1% work | -0.1% (overhead) |



- Argon2H' dominates (64% of time, sequential)**When It Matters**

- Blake2b ~35% of time- Hard difficulty (solutions rare): Minimal overhead, no benefit most batches

- Instructions <1% of time- Medium difficulty (occasional solutions): Moderate benefit when solution found

- GPU excels at batch processing- Easy difficulty (frequent solutions): Maximum benefit, significant time savings

- CPU competitive for single hashes

**Trade-offs**

## Implementation Details- Cost: Atomic read after each loop (~1-2 GPU cycles)

- Benefit: Skip remaining loops when solution found (saves 1000s of cycles)

### Cryptographic Primitives- Net effect: ~0.1% overhead in worst case, up to 87.5% speedup in best case



**Blake2b-512:****Synchronization Behavior**

- Custom CUDA implementation

- Incremental hashing supportThreads do not synchronize immediately. Behavior:

- Context cloning for Special operands1. Thread A finds solution at time T, sets flag

- ~3000 invocations per hash2. Thread B checks flag at time T+1 (after completing current loop)

3. Thread B exits without starting next loop

**Argon2H':**4. Result: Mixed exit times but all threads stop within one loop duration

- Ported from cryptoxide

- Variable-length output (448, 5120, 8192 bytes)No thread synchronization primitives required, avoiding expensive barriers.

- Sequential execution per thread

- 17 invocations per hash### Shared ROM Architecture



### Known Quirks (Maintained for Consensus)**Single ROM, Multiple Threads**



**Mod Operation Bug:**```

```rustGPU Memory Layout

Op3::Mod => src1 / src2  // Should be src1 % src2┌─────────────────────────────────────┐

```│ ROM (10MB-1GB)                      │ ← Single copy, shared

│ - Stored in texture memory          │

**ROM Addressing:**│ - Read-only, cached                 │

```rust│ - Accessible by all threads         │

let start = (addr % (rom.len() / 64)) * 64;└─────────────────────────────────────┘

```         ↓         ↓         ↓

    Thread 0   Thread 1  ... Thread N

**Memory Counter Cycling:**    Nonce 0    Nonce 1      Nonce N

```rust       ↓          ↓            ↓

let idx = (memory_counter % 8) * 8;    Hash 0     Hash 1       Hash N

``````



All GPU behavior matches CPU exactly.**Benefits**

- Memory efficient: 10MB total vs 10MB × thread count on CPU

### Constraints- Cache efficient: Texture cache shared across threads

- Bandwidth efficient: Cached reads reduce memory traffic

- MAX_PROGRAM_INSTRS: 1024

- Compute Capability: ≥7.5 required**Implementation**

- ROM practical limit: ~1GB```rust

- Batch size limit: GPU memory dependent// Upload ROM once

let rom_handle = gpu_upload_rom(rom_data, rom_digest);

## CPU vs GPU Comparison

// Use for millions of nonces

### When to Use CPUloop {

    let salts = generate_batch(batch_size);

- Single hash or small batch    let (hashes, flags) = gpu_mine_batch(rom_handle, salts, ...);

- Memory constrained    if let Some(solution) = check_flags(flags) {

- No CUDA GPU available        break;

- Power efficiency critical    }

}

### When to Use GPU

// Cleanup when done

- Mining (millions of nonces)gpu_free_rom(rom_handle);

- Batch validation```

- Bulk verification

- High throughput required**Texture Memory Properties**

- Cached: Repeated access to same ROM location is fast

### Migration Strategy- Read-only: Hardware enforced, prevents corruption

- 2D/3D addressing: Efficient for structured data access

1. Verify compatibility:- Interpolation: Not used (point sampling only)

```rust

let cpu_digest = ashmaize::hash(salt, &rom, 8, 256);## Implementation Status

let gpu_digest = gpu_ashmaize::hash(salt, &rom, 8, 256);

assert_eq!(cpu_digest, gpu_digest);### Completed Components

```

- [x] Algorithm analysis and documentation

2. Benchmark your workload- [x] CUDA architecture design

3. Use hybrid approach if needed- [x] Blake2b-512 implementation (28/28 tests passing)

- [x] Argon2H' implementation (13/13 tests passing)

## Documentation- [x] VM and instruction set (all tests passing)

- [x] Kernel integration with early exit optimization

**CUDA Analysis:**- [x] Rust FFI layer

- [00_INDEX.md](../docs/cuda-analysis/00_INDEX.md) - Documentation index- [x] API design (three tiers)

- [01_VM_ARCHITECTURE.md](../docs/cuda-analysis/01_VM_ARCHITECTURE.md) - VM state- [x] Build system with automatic CUDA compilation

- [02_ROM_GENERATION.md](../docs/cuda-analysis/02_ROM_GENERATION.md) - ROM generation- [x] Performance optimization

- [03_CRYPTOGRAPHIC_PRIMITIVES.md](../docs/cuda-analysis/03_CRYPTOGRAPHIC_PRIMITIVES.md) - Blake2b/Argon2H'- [x] Hash correctness validation (byte-perfect match with CPU)

- [04_INSTRUCTION_SET.md](../docs/cuda-analysis/04_INSTRUCTION_SET.md) - Instructions- [x] Large ROM support (tested up to 1GB)

- [05_CUDA_ARCHITECTURE.md](../docs/cuda-analysis/05_CUDA_ARCHITECTURE.md) - Kernel design- [x] Automatic early exit for mining efficiency

- [SUMMARY.md](../docs/cuda-analysis/SUMMARY.md) - Summary

### Test Results

**API Documentation:**

```bash**Blake2b Tests**: 28/28 passing

cargo doc --open```

```- Standard test vectors

- Edge cases (empty input, max length)

## Contributing- Streaming API

- Buffer aliasing

Contributions welcome. Please ensure:```



1. All tests pass (Rust and CUDA)**Argon2H' Tests**: 13/13 passing

2. CPU/GPU hash equivalence maintained```

3. Code formatted (`cargo fmt`)- Variable output lengths (32B to 1GB)

4. No new warnings (`cargo clippy`)- Test vectors from cryptoxide

5. Documentation updated- Exact seed matching with CPU

- Loop termination correctness

## Troubleshooting```



**CUDA not found:****Integration Tests**: All passing

``````

Error: nvcc not found- End-to-end hash computation

```- Batch processing

Install CUDA Toolkit and add to PATH.- ROM upload and management

- Early exit behavior

**Out of memory:**```

```

Error: cudaMalloc failed### Known Limitations

```

Reduce batch size or ROM size.**Performance Constraints**

- Argon2H' is inherently sequential (cannot parallelize within single hash)

**Hash mismatch:**- Large ROM generation is memory-bandwidth limited (~15 MB/s)

```- Batch size must fit GPU memory (typical: 64K-262K nonces)

Error: GPU hash != CPU hash

```**Platform Support**

Check CUDA Toolkit version (12.0+), verify test vectors.- Tested on Linux with CUDA 13.0 and RTX 5060

- Windows support expected but untested

## License- macOS not supported (no CUDA)



Dual-licensed under MIT and Apache-2.0, same as parent project.### Recent Fixes



## Status**Argon2H' Loop Termination Bug** (November 2025)

- **Issue**: GPU performed 13 iterations where CPU performed 11 + final

- Implementation: Complete- **Root Cause**: Loop condition mismatch (`while output_offset < len` vs `while bytes_remaining > 64`)

- Testing: All tests passing (Blake2b 44/44, Argon2H' 13/13)- **Fix**: Matched GPU loop logic to CPU reference implementation exactly

- CPU/GPU equivalence: Verified- **Result**: Byte-perfect hash matching across all test vectors

- Production: Ready- **Documentation**: See ARGON2_FIX_SUMMARY.md for detailed analysis



**Last Updated:** November 2025---


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
