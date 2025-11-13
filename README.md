# AshMaize

AshMaize is a memory-hard, ASIC-resistant proof-of-work hash algorithm combining cryptographic primitives (Blake2b-512, Argon2H') with random VM execution. Designed for portability across CPU, GPU, and WebAssembly environments.

## Key Features

- **Memory-Hard**: Argon2H' for key derivation and program shuffling prevents memory optimization attacks
- **Random VM**: 32-register virtual machine with 13 operations executing randomized instruction sequences
- **Large ROM**: Configurable dataset sizes (1KB to 1GB+) with cache-friendly 64-byte access patterns
- **GPU Accelerated**: Production CUDA implementation (40-60x speedup over CPU for batch mining)
- **WebAssembly**: Browser-compatible implementation for web-based applications
- **ASIC Resistant**: Complex memory dependencies and sequential Argon2H' execution prevent specialized hardware advantages
- **Deterministic**: Byte-perfect reproducibility across all implementations (CPU, GPU, WASM)

## Algorithm Overview

```
ROM Generation (one-time)          VM Execution (per salt)
─────────────────────────          ────────────────────────

    Blake2b Seed                   ROM Digest + Salt
         ↓                                 ↓
    Argon2H' (TwoStep)              Argon2H' Init
    - Pre-memory (16KB)             - 32 x 64-bit registers
    - Expansion (10MB-1GB)          - 2 Blake2b contexts
         ↓                          - Program seed (64B)
   ROM Data (read-only)                     ↓
         │                          Loop (8x default):
         │                            1. Argon2H' shuffle program
         │                            2. Execute 256 instructions:
         │                               - Arithmetic (Add,Mul,MulH)
         │                               - Bitwise (Xor,And,Neg)
         │                               - Rotation (RotL,RotR)
         └──────────────────────────────► - Memory (ROM access)
                                           - Special (ISqrt,BitRev,Hash)
                                         3. Update Blake2b digests
                                         4. Post-instruction mixing
                                            (Argon2H' + XOR registers)
                                                 ↓
                                         Finalize (Blake2b combine)
                                                 ↓
                                           64-byte hash
```

## Repository Structure

```
ce-ashmaize/
├── src/                      # Core Rust implementation
│   ├── lib.rs               # VM, hash function, instruction execution
│   └── rom.rs               # ROM generation (FullRandom, TwoStep)
├── gpu-ashmaize/            # CUDA GPU implementation
│   ├── cuda/
│   │   ├── blake2b.cu/cuh   # Blake2b-512 (44 tests passing)
│   │   ├── argon2.cu/cuh    # Argon2H' (13 tests passing)
│   │   ├── vm.cu/cuh        # VM state management
│   │   ├── instructions.cu/cuh # Instruction decode/execute
│   │   └── kernel.cu/cuh    # Main mining kernel
│   ├── src/
│   │   ├── lib.rs           # Public Rust API
│   │   ├── ffi.rs           # CUDA FFI bindings
│   │   └── error.rs         # Error types
│   ├── examples/            # GPU examples (minimal_test, systematic_debug, etc.)
│   ├── tests/               # CUDA unit tests (*.cu files)
│   └── README.md            # GPU-specific documentation
├── crates/
│   ├── ashmaize-web/        # WebAssembly bindings
│   └── ashmaize-webdemo/    # Web demo (Leptos framework)
├── examples/                # CPU examples (hash.rs, benchmarks)
├── benches/                 # Criterion benchmarks
└── docs/
    └── cuda-analysis/       # CUDA implementation analysis (6 documents)
```

## Prerequisites

**CPU Implementation:**
- Rust 1.70+ (edition 2024)
- Cargo

**GPU Implementation:**
- NVIDIA GPU (Compute Capability 7.5+)
- CUDA Toolkit 12.0+
- nvcc compiler in PATH

**WebAssembly:**
- wasm-pack
- Node.js (for web demo)

## Installation

```bash
git clone https://github.com/input-output-hk/ce-ashmaize.git
cd ce-ashmaize

# CPU only
cargo build --release

# GPU (requires CUDA)
cd gpu-ashmaize
cargo build --release

# WebAssembly
cd crates/ashmaize-web
wasm-pack build --target web
```

## Usage

### CPU Implementation

```rust
use ashmaize::{hash, Rom, RomGenerationType};

const KB: usize = 1024;
const MB: usize = 1024 * KB;

// Generate ROM (one-time, ~430ms for 256MB)
let rom = Rom::new(
    b"my_seed",
    RomGenerationType::TwoStep {
        pre_size: 16 * KB,
        mixing_numbers: 4,
    },
    256 * MB,
);

// Compute hash (~730µs per hash)
let salt = b"nonce_12345";
let digest = hash(salt, &rom, 8, 256);
println!("Hash: {}", hex::encode(&digest[..8]));
```

### GPU Implementation

```rust
use gpu_ashmaize::{hash, GpuMiner};
use ashmaize::{Rom, RomGenerationType};

const MB: usize = 1024 * 1024;

// Generate ROM on CPU
let rom = Rom::new(
    b"my_seed",
    RomGenerationType::TwoStep {
        pre_size: 16 * 1024,
        mixing_numbers: 4,
    },
    256 * MB,
);

// Option 1: Drop-in replacement (includes upload overhead)
let digest = hash(b"nonce", &rom, 8, 256);

// Option 2: Reusable context (amortizes upload cost)
let mut miner = GpuMiner::with_params(8, 256)?;
miner.upload_rom(&rom)?;

for nonce in 0..1000 {
    let salt = nonce.to_le_bytes();
    let digest = miner.hash(&salt)?;
    // Check solution...
}

// Option 3: Batch processing (maximum throughput)
let salts: Vec<Vec<u8>> = (0..65536)
    .map(|i| i.to_le_bytes().to_vec())
    .collect();
let digests = gpu_ashmaize::hash_batch(&salts, &rom, 8, 256)?;
```

### WebAssembly

```javascript
import init, { Rom } from './pkg/ashmaize_web.js';

await init();

const rom = Rom.builder()
    .key(new Uint8Array([1, 2, 3]))
    .size(1024 * 1024)  // 1MB
    .gen_two_steps(16384, 4)
    .build();

const hash = rom.hash(new Uint8Array([0, 1, 2, 3]), 8, 256);
console.log('Hash:', Array.from(hash.slice(0, 8)));
```

## Testing

### CPU Tests

```bash
# All tests
cargo test --release

# Specific test
cargo test --release test_eq

# Benchmarks
cargo bench

# Example: verify hash determinism
cargo run --release --example hash
```

### GPU Tests

**CUDA Unit Tests:**
```bash
cd gpu-ashmaize

# All CUDA tests
make test

# Specific primitives
make test-blake2b   # 44/44 tests passing
make test-argon2    # 13/13 tests passing
```

**Rust Integration Tests:**
```bash
cd gpu-ashmaize

# Quick verification
cargo run --release --example minimal_test

# Comprehensive validation
cargo run --release --example systematic_debug

# Large ROM stress test (256MB, 512MB, 1GB)
cargo run --release --example test_large_roms

# CPU/GPU equivalence
cargo test --release
```

### Test Results

**Blake2b-512**: 44/44 tests passing
- RFC 7693 test vectors
- Incremental hashing
- Context cloning

**Argon2H'**: 13/13 tests passing
- Variable-length output (32B to 1GB)
- Byte-perfect match with cryptoxide

**VM Execution**: All tests passing
- ROM sizes: 64KB to 1GB
- Multiple parameter combinations
- CPU/GPU hash equivalence verified

## Performance

### Typical Performance (256MB ROM, 8 loops, 256 instructions)

**ROM Generation (one-time):**
- TwoStep: ~430ms
- FullRandom: ~50-60 seconds (not recommended)

**CPU (Rust native):**
- Single-core: ~730µs per hash (~1,370 hash/sec)
- 16-core: ~46µs per hash (~21,700 hash/sec)

**GPU (CUDA, RTX 5060):**
- Single hash: ~370ms (includes upload overhead)
- Batch (65K salts): ~2.6 seconds (~25,000 hash/sec)
- Speedup: ~16-18x over single-core CPU, ~1.15x over 16-core CPU

**WebAssembly (browser):**
- ~2-3ms per hash (depends on browser and hardware)

### Performance Notes

- GPU excels at batch processing (parallel nonce mining)
- CPU competitive for single hashes or small batches
- ROM upload cost amortized across batch in GPU implementation
- Argon2H' dominates execution time (64%), cannot parallelize within single hash

## Algorithm Details

### Virtual Machine

**State (per hash computation):**
- 32 x 64-bit registers
- Program counter (wrapping)
- Memory counter (ROM access tracking)
- Loop counter
- 2 Blake2b-512 contexts (prog_digest, mem_digest)
- Program seed (64 bytes)
- Program buffer (nb_instrs × 20 bytes)

**Instructions (20 bytes each):**
- Opcode (1 byte) → 13 operation types
- Operands (1 byte) → 5 operand types
- Register indices (2 bytes) → r1, r2, r3 (5 bits each)
- Literal values (16 bytes) → two 64-bit immediates

**Operations:**
- Arithmetic: Add, Mul, MulH, Div, Mod (note: Mod bug - uses division)
- Bitwise: Xor, And, Neg
- Rotation: RotL, RotR
- Math: ISqrt, BitRev
- Cryptographic: Hash (Blake2b-512)

**Operand Types:**
- Reg: Register value
- Memory: ROM access (64-byte blocks)
- Literal: Immediate 64-bit value
- Special1: prog_digest finalization (expensive)
- Special2: mem_digest finalization (expensive)

### Cryptographic Primitives

**Blake2b-512 (35% of execution time):**
- ROM seed generation
- Incremental digest updates (~3000 per hash)
- Special operand values
- Final hash combination

**Argon2H' (64% of execution time):**
- VM initialization: 448 bytes output
- Program shuffling: 5120 bytes per loop (8x)
- Post-instruction mixing: 8192 bytes per loop (8x)
- Sequential bottleneck (cannot parallelize)

### Execution Flow

**Per hash computation:**
1. VM Init: Argon2H'(rom_digest + salt) → registers, digests, prog_seed
2. Loop (nb_loops times, typically 8):
   a. Program shuffle: Argon2H'(prog_seed) → program instructions
   b. Execute nb_instrs instructions (typically 256)
   c. Post-instruction mixing:
      - Sum all registers
      - Update digests with sum
      - Argon2H'(digests) → 8KB mixing data
      - XOR registers 32 times with mixing data
      - Update prog_seed for next loop
3. Finalize: Blake2b(prog_digest + mem_digest + regs) → 64-byte hash

## Implementation Notes

### ROM Generation

**TwoStep (recommended):**
1. Generate pre-memory (16KB) using Argon2H'
2. Compute offset arrays (differential addressing)
3. Expand to full size by XOR-combining pre-memory chunks
4. Blake2b digest of entire ROM

**FullRandom (not recommended):**
- Single Argon2H' pass for entire ROM
- 100-200x slower than TwoStep
- No practical advantage

### Known Quirks

**Mod Operation Bug:**
```rust
Op3::Mod => src1 / src2  // Should be src1 % src2
```
Maintained for consensus across all implementations.

**ROM Addressing:**
```rust
// CPU implementation uses byte offset directly
let start = (addr % (rom.len() / 64)) * 64;
&rom[start..start+64]
```
GPU implementation matches this exact behavior using texture memory.

**Memory Counter Cycling:**
Within each 64-byte ROM block, extracts 8-byte chunks cyclically:
```rust
let idx = (memory_counter % 8) * 8;  // 0, 8, 16, 24, 32, 40, 48, 56
```

### GPU Implementation Details

**Memory Layout:**
- ROM: Texture memory (cached, read-only, hardware bounds checking)
- VM state: Registers + local memory (~518KB per thread)
- Working memory: Argon2H' temporary buffers (reused)

**Parallelism:**
- Each thread: independent hash computation
- No inter-thread communication required
- Optimal for batch nonce mining

**Limitations:**
- MAX_PROGRAM_INSTRS = 1024 (20KB program buffer)
- ROM size: practical limit ~1GB (GPU memory dependent)
- Argon2H' is sequential (accept as bottleneck)

## Documentation

**Core Algorithm:**
- [docs/cuda-analysis/00_INDEX.md](docs/cuda-analysis/00_INDEX.md) - Documentation index
- [docs/cuda-analysis/01_VM_ARCHITECTURE.md](docs/cuda-analysis/01_VM_ARCHITECTURE.md) - VM state machine
- [docs/cuda-analysis/02_ROM_GENERATION.md](docs/cuda-analysis/02_ROM_GENERATION.md) - Memory-hard component
- [docs/cuda-analysis/03_CRYPTOGRAPHIC_PRIMITIVES.md](docs/cuda-analysis/03_CRYPTOGRAPHIC_PRIMITIVES.md) - Blake2b/Argon2H'
- [docs/cuda-analysis/04_INSTRUCTION_SET.md](docs/cuda-analysis/04_INSTRUCTION_SET.md) - 13 operations detailed
- [docs/cuda-analysis/05_CUDA_ARCHITECTURE.md](docs/cuda-analysis/05_CUDA_ARCHITECTURE.md) - GPU kernel design
- [docs/cuda-analysis/SUMMARY.md](docs/cuda-analysis/SUMMARY.md) - Implementation summary

**GPU Implementation:**
- [gpu-ashmaize/README.md](gpu-ashmaize/README.md) - GPU miner documentation
- [gpu-ashmaize/examples/](gpu-ashmaize/examples/) - Usage examples

**API Documentation:**
```bash
# Generate and open API docs
cargo doc --open
```

## Development

### Building

```bash
# Full workspace
cargo build --release

# CPU only
cargo build --release --package ashmaize

# GPU (requires CUDA)
cd gpu-ashmaize && cargo build --release

# WebAssembly
cd crates/ashmaize-web && wasm-pack build --target web

# Web demo
cd crates/ashmaize-webdemo && trunk serve
```

### Testing

```bash
# CPU tests
cargo test --release

# GPU CUDA tests
cd gpu-ashmaize && make test

# GPU Rust tests
cd gpu-ashmaize && cargo test --release

# Benchmarks
cargo bench
```

### Project Layout

**Core Implementation:**
- `src/lib.rs`: VM, hash function, instruction execution (~537 lines)
- `src/rom.rs`: ROM generation (FullRandom, TwoStep) (~369 lines)

**GPU Implementation:**
- `gpu-ashmaize/cuda/*.cu`: CUDA kernels (Blake2b, Argon2H', VM, instructions)
- `gpu-ashmaize/src/*.rs`: Rust FFI wrapper and public API

**WebAssembly:**
- `crates/ashmaize-web/src/lib.rs`: WASM bindings (~135 lines)
- `crates/ashmaize-webdemo/`: Leptos-based web demo

**Documentation:**
- `docs/cuda-analysis/`: 6 comprehensive documents (3700+ lines)

### Implementation Notes for Developers

**ROM Addressing (maintain exactly):**
```rust
let start = (addr % (rom.len() / 64)) * 64;
```

**Mod Bug (do not fix for consensus):**
```rust
Op3::Mod => src1 / src2  // Intentional: uses division
```

**Memory Counter Cycling:**
```rust
let idx = (memory_counter % 8) * 8;
```

**Test Vector (validation):**
- ROM: seed="123", TwoStep(16KB, 4), 10MB
- Salt: "hello"
- Parameters: 8 loops, 256 instructions
- Expected: `[56, 148, 1, 228, 59, 96, ...]`

**GPU Constraints:**
- MAX_PROGRAM_INSTRS = 1024
- Compute Capability ≥ 7.5 required
- ROM practical limit ~1GB

## Contributing

Contributions are welcome. Please ensure:

1. All tests pass (CPU and GPU if applicable)
2. Code follows Rust formatting (`cargo fmt`)
3. No new compiler warnings (`cargo clippy`)
4. New features include tests and documentation
5. GPU changes maintain CPU/GPU hash equivalence
6. Performance implications documented

## License

Dual-licensed under:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
