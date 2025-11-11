**This project is currently a work in progress. It is provided as-is, without
any warranty of correctness, functionality, or fitness for any particular
purpose. There is no guarantee that it works as intended, and it may contain
bugs, incomplete features, or incorrect cryptographic behavior.**

**Do not use this software for security-critical or production purposes. Use at
your own risk.**

# AshMaize

AshMaize is a Proof-of-Work (PoW) algorithm designed to be ASIC-resistant while remaining simple to implement. It combines cryptographic primitives (Blake2b-512, Argon2H') with a Random VM execution model to create a memory-hard, compute-intensive hash function.

## Key Features

- **🔐 Memory-Hard**: Uses Argon2H' for key derivation and program shuffling
- **🎲 Random VM**: Executes randomized instruction sequences on a 32-register virtual machine
- **📊 Large ROM**: Operates on configurable ROM sizes (64KB to 1GB+) for dataset access
- **🚀 GPU Accelerated**: CUDA implementation with verified correctness against CPU reference
- **🌐 WebAssembly**: Browser-compatible implementation for web mining
- **⚡ ASIC Resistant**: Complex memory access patterns and dynamic instruction execution

## Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │         AshMaize Hash               │
                    └─────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            ┌───────────────┐             ┌──────────────┐
            │  ROM Generation│             │  VM Hashing  │
            │   (One-time)   │             │  (Per Salt)  │
            └───────────────┘             └──────────────┘
                    │                               │
            ┌───────┴────────┐          ┌──────────┴──────────┐
            ▼                ▼          ▼                     ▼
      ┌─────────┐    ┌─────────┐  ┌────────┐         ┌──────────┐
      │ Blake2b │    │Argon2H' │  │VM Init │         │VM Execute│
      │  Seed   │───▶│Expansion│  │(Argon2)│────────▶│(n loops) │
      └─────────┘    └─────────┘  └────────┘         └──────────┘
                           │             │                   │
                           ▼             ▼                   ▼
                     ┌─────────┐   ┌─────────┐       ┌──────────┐
                     │ROM Data │   │32 Regs  │       │ Shuffle  │
                     │256KB-1GB│   │Digests  │       │ (Argon2) │
                     └─────────┘   └─────────┘       └──────────┘
                                                            │
                                                            ▼
                                                   ┌─────────────────┐
                                                   │Execute Instrs   │
                                                   │(Add,Mul,ISqrt,  │
                                                   │ RotL,Xor,ROM...)│
                                                   └─────────────────┘
                                                            │
                                                            ▼
                                                   ┌─────────────────┐
                                                   │Post-Instr Mixing│
                                                   │(Blake2b+Argon2) │
                                                   └─────────────────┘
                                                            │
                                                            ▼
                                                     ┌──────────┐
                                                     │Final Hash│
                                                     │ (64 bytes)│
                                                     └──────────┘
```

## Project Structure

```
ce-ashmaize/
├── src/                    # Core Rust CPU implementation
│   ├── lib.rs             # Main hash function and VM
│   └── rom.rs             # ROM generation (FullRandom, TwoStep)
├── gpu-ashmaize/          # CUDA GPU implementation
│   ├── cuda/              # CUDA kernels and device code
│   │   ├── blake2b.cu     # Blake2b-512 implementation
│   │   ├── argon2.cu      # Argon2H' implementation
│   │   ├── vm.cu          # VM initialization and execution
│   │   ├── instructions.cu # Instruction decode/execute
│   │   └── kernel.cu      # Main mining kernel
│   ├── src/lib.rs         # Rust FFI wrapper
│   ├── examples/          # GPU test examples
│   └── tests/             # CUDA unit tests
├── crates/
│   ├── ashmaize-web/      # WebAssembly bindings
│   └── ashmaize-webdemo/  # Web demo application
├── examples/              # CPU examples and benchmarks
├── benches/               # Performance benchmarks
└── docs/                  # Technical documentation
    ├── cuda-analysis/     # Detailed CUDA analysis
    ├── SPECS.md           # Algorithm specification
    ├── ARCHITECTURE_DIAGRAMS.md
    └── GPU_*.md           # GPU implementation details
```

## Quick Start

### Prerequisites

**For CPU-only:**
- Rust 1.70+ (2024 edition)
- Cargo

**For GPU acceleration:**
- NVIDIA GPU (Compute Capability 7.5+, tested on sm_90/RTX 5060)
- CUDA Toolkit 12.0+ (tested with CUDA 13.0)
- nvcc compiler

### Installation

```bash
# Clone the repository
git clone https://github.com/input-output-hk/ce-ashmaize.git
cd ce-ashmaize

# Build CPU version
cargo build --release

# Build with GPU support
cd gpu-ashmaize
cargo build --release
```

### Basic Usage

```rust
use ashmaize::{hash, Rom, RomGenerationType};

// Generate ROM (one-time setup)
const MB: usize = 1024 * 1024;
let rom = Rom::new(
    b"my_seed",
    RomGenerationType::TwoStep {
        pre_size: 16 * MB,
        mixing_numbers: 4,
    },
    256 * MB,  // 256MB ROM
);

// Hash with salt
let salt = b"my_nonce_12345";
let nb_loops = 8;      // Number of VM execution loops
let nb_instrs = 256;   // Instructions per loop
let hash = hash(salt, &rom, nb_loops, nb_instrs);

println!("Hash: {:?}", &hash[..16]);
```

### GPU Usage

```rust
use gpu_ashmaize;
use ashmaize::{Rom, RomGenerationType};

let rom = Rom::new(b"seed", RomGenerationType::FullRandom, 256 * 1024 * 1024);
let salt = b"test_nonce";
let hash = gpu_ashmaize::hash(salt, &rom, 8, 256);
```

## Testing

### CPU Tests
```bash
# Run all CPU tests
cargo test --release

# Run specific test
cargo test --release test_eq

# Run benchmarks
cargo bench
```

### GPU Tests

**Low-Level Primitives (CUDA):**
```bash
cd gpu-ashmaize
make test              # Run all CUDA tests
make test-blake2b      # Blake2b tests (28 tests)
make test-argon2       # Argon2H' tests (13 tests)
```

**High-Level Integration (Rust):**
```bash
cd gpu-ashmaize

# Minimal verification
cargo run --release --example minimal_test

# Systematic testing (different parameters)
cargo run --release --example systematic_debug

# Large ROM testing (256MB, 512MB, 1GB)
cargo run --release --example test_large_roms

# CPU vs GPU equivalence
cargo test --release
```

### Test Coverage

✅ **Blake2b-512**: 44/44 tests passing
  - 28 CUDA unit tests
  - 16 Python reference verification tests

✅ **Argon2H'**: 13/13 CUDA tests passing
  - Byte-perfect match between CPU and GPU

✅ **VM Execution**: All integration tests passing
  - Different ROM sizes (64KB to 1GB)
  - Different parameters (loops: 2-16, instrs: 256-512)
  - Multiple salts and edge cases

## Performance

**Typical Performance (RTX 5060, 256MB ROM, 8 loops, 256 instrs):**
- ROM Generation: ~430ms (one-time)
- CPU Hash: ~730µs per hash
- GPU Hash: ~370ms per hash (includes transfer overhead)

**Note:** GPU shows significant speedup with batch processing (multiple salts).

## Algorithm Details

### VM Specification

- **Registers**: 32 × 64-bit registers
- **Instructions**: 20 bytes each, 16 operation types:
  - Arithmetic: Add, Sub, Mul (wrapping)
  - Bitwise: And, Or, Xor, Neg
  - Rotation: RotL, RotR
  - Special: ISqrt, BitRev
  - Memory: ROM access (64-byte cache lines)
  - Control: Special1 (prog_digest), Special2 (mem_digest)

- **Operand Types**: Register, Memory (ROM), Literal, Special
- **Program Counter**: Wrapping increment
- **Memory Counter**: Increments on ROM access

### Cryptographic Primitives

- **Blake2b-512**: Used for ROM generation seed, digest finalization
- **Argon2H'**: Custom variant for:
  - VM initialization (448 bytes → 32 regs + 2 digests + prog_seed)
  - Program shuffling (generates random instruction bytes)
  - Post-instruction mixing (8KB mixing data)

### Mixing Strategy

Each loop iteration:
1. Shuffle program bytes using Argon2H'(prog_seed)
2. Execute instructions (register operations, ROM access)
3. Post-instruction mixing:
   - Sum all registers
   - Finalize digests with sum
   - Generate 8KB mixing data via Argon2H'
   - XOR 32 rounds × 32 registers with mixing data
4. Update prog_seed for next loop

## Memory Safety & Bounds Checking

The GPU implementation includes comprehensive safety measures:

- ✅ **Input Validation**: All parameters validated before kernel launch
- ✅ **ROM Bounds Checking**: Clamping to prevent out-of-bounds reads
- ✅ **Program Buffer Limits**: MAX_PROGRAM_INSTRS = 1024 (20KB per VM)
- ✅ **NULL Pointer Checks**: Defensive validation at entry points
- ✅ **Range Validation**: Sanity checks on all numeric inputs
- ✅ **Graceful Degradation**: Clamps to limits rather than crashing

**Tested with:**
- ROMs up to 1GB
- Instruction counts up to 512
- Various loop counts (2-16)
- Multiple salts and edge cases

**No memory leaks, no crashes, no illegal access.**

## Documentation

- **[SPECS.md](SPECS.md)**: Complete algorithm specification
- **[ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md)**: Visual architecture
- **[GPU_ACCELERATION_ANALYSIS.md](GPU_ACCELERATION_ANALYSIS.md)**: GPU design analysis
- **[TECHNICAL_ANALYSIS.md](TECHNICAL_ANALYSIS.md)**: Implementation details
- **[LOW_LEVEL_TEST_SUMMARY.md](gpu-ashmaize/LOW_LEVEL_TEST_SUMMARY.md)**: Test verification
- **[docs/cuda-analysis/](docs/cuda-analysis/)**: Detailed CUDA implementation guide

## For AI Coding Assistants

### Context Summary

**Project Type**: Cryptographic PoW algorithm with CPU (Rust) and GPU (CUDA) implementations

**Key Characteristics:**
- Memory-hard algorithm (Argon2H' based)
- Random VM execution model (32 registers, 16 instruction types)
- Large dataset (ROM) access patterns
- Verified CPU/GPU equivalence (byte-perfect match)

**Critical Implementation Details:**

1. **ROM Addressing Quirk** (⚠️ Important):
   ```rust
   // CPU: rom.at(i) does modulo THEN uses result as byte offset
   let start = i % (data.len() / 64);  // Block index 0-4095
   &data[start..start+64]              // Uses as BYTE offset!
   ```
   GPU must match this exact behavior (see `gpu-ashmaize/cuda/instructions.cu:70`)

2. **Memory Bounds**:
   - MAX_PROGRAM_INSTRS = 1024 (gpu-ashmaize/cuda/vm.cuh)
   - ROM min size: 64 bytes, max: 1GB
   - Instruction size: 20 bytes each

3. **Verified Correctness**:
   - All low-level tests passing (Blake2b 44/44, Argon2H' 13/13)
   - CPU/GPU hashes match byte-for-byte
   - Tested with ROMs up to 1GB

**Common Operations:**

```rust
// Generate ROM
let rom = Rom::new(seed, RomGenerationType::TwoStep { 
    pre_size: 16*MB, mixing_numbers: 4 
}, 256*MB);

// Hash (CPU)
let hash = ashmaize::hash(salt, &rom, nb_loops, nb_instrs);

// Hash (GPU)
let hash = gpu_ashmaize::hash(salt, &rom, nb_loops, nb_instrs);
```

**Testing Strategy:**
1. Low-level primitives first (make test-blake2b, make test-argon2)
2. VM integration tests (cargo run --example minimal_test)
3. Large ROM stress tests (cargo run --example test_large_roms)

**File Modification Guidelines:**
- CPU VM: `src/lib.rs` (VM struct, execute(), hash())
- GPU VM: `gpu-ashmaize/cuda/{vm.cu,instructions.cu,kernel.cu}`
- Crypto: `gpu-ashmaize/cuda/{blake2b.cu,argon2.cu}`
- Tests: `gpu-ashmaize/examples/*.rs` and `gpu-ashmaize/tests/*.cu`

**Known Constraints:**
- CUDA requires sm_75+ (tested on sm_90)
- Program instructions limited to 1024 per VM instance
- ROM size must be ≥64 bytes for proper operation
- GPU transfers have overhead (batching recommended)

**Debug Output Locations:**
- GPU kernel: `gpu-ashmaize/cuda/kernel.cu` (printf statements present)
- GPU instructions: `gpu-ashmaize/cuda/instructions.cu` (Mul opcode logging)
- CPU VM: `src/lib.rs` (eprintln! statements present)

⚠️ **Debug output should be removed/conditionalized before production**

### Quick Reference Commands

```bash
# Build everything
cargo build --release

# Test CPU
cargo test --release

# Test GPU low-level
cd gpu-ashmaize && make test

# Test GPU high-level
cd gpu-ashmaize && cargo run --release --example minimal_test

# Clean build
cargo clean && cd gpu-ashmaize && make clean
```

## Contributing

Contributions welcome! Please ensure:
1. All tests pass (CPU and GPU if applicable)
2. Code follows Rust style guidelines
3. CUDA code includes proper bounds checking
4. New features include tests
5. Performance implications documented

## License

This project is licensed under either of the following licenses:

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
  http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or
  http://opensource.org/licenses/MIT)
