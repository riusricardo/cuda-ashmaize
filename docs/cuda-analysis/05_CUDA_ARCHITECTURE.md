# CUDA Architecture Design for AshMaize GPU Mining

## Overview
This document defines the complete CUDA implementation architecture for AshMaize PoW mining, translating the Rust algorithm into production-ready GPU code optimized for parallel salt search.

---

## Design Goals

### Primary Objectives
1. **Correctness**: Byte-for-byte match with CPU implementation
2. **Determinism**: Identical results across runs and platforms
3. **Performance**: 40-60x speedup through parallel mining
4. **Maintainability**: Clean, professional, well-documented code

### Non-Goals
- Single-hash optimization (algorithm is sequential-heavy)
- Cross-GPU compatibility (NVIDIA CUDA only)
- Dynamic difficulty adjustment (handle in Rust layer)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Rust Application                       │
│  (Mining coordinator, ROM management, result validation)    │
└────────────────┬─────────────────┬──────────────────────────┘
                 │                 │
                 │ FFI boundary    │ Results
                 ↓                 ↑
┌─────────────────────────────────────────────────────────────┐
│                    Rust FFI Layer                           │
│  - ROM transfer to GPU                                      │
│  - Salt batch generation                                    │
│  - Kernel launch configuration                              │
│  - Result collection                                        │
└────────────────┬───────────────────────────────────────────┬┘
                 │ cudaMemcpy      │ cudaMemcpy              │
                 ↓                 ↑                         │
┌────────────────────────────────────────────────────────────┴┐
│                      GPU Global Memory                       │
│  - ROM data (texture-bound, read-only)                      │
│  - Salt batch input                                          │
│  - Hash results output                                       │
│  - Success flags                                             │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│                    CUDA Kernel Grid                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Thread Block (256 threads)                           │   │
│  │ ┌──────────┐ ┌──────────┐       ┌──────────┐        │   │
│  │ │ Thread 0 │ │ Thread 1 │  ...  │Thread 255│        │   │
│  │ │  Salt 0  │ │  Salt 1  │       │ Salt 255 │        │   │
│  │ └──────────┘ └──────────┘       └──────────┘        │   │
│  └──────────────────────────────────────────────────────┘   │
│  ... (Multiple blocks across SMs)                            │
└─────────────────────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│              Per-Thread Execution (VM Instance)              │
│  1. VM initialization (Argon2H' from ROM + salt)            │
│  2. Loop (nb_loops times, default 8):                       │
│     a. Program shuffle (Argon2H' prog_seed → instructions)  │
│     b. Execute nb_instrs instructions (default 256)         │
│     c. Post-instructions (Argon2H' mixing + XOR)            │
│  3. Finalize (combine digests → final hash)                 │
│  4. Check difficulty, store result if valid                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Memory Architecture

### GPU Memory Layout

```
GPU Global Memory (~8-16 GB typical)
├── ROM Data (10MB - 1GB, depends on config)
│   └── Bound to texture object for cached access
├── Salt Input Buffer (batch size × salt size)
│   └── Example: 65536 salts × 32 bytes = 2 MB
├── Output Hash Buffer (batch size × 64 bytes)
│   └── Example: 65536 × 64 = 4 MB
├── Success Flags (batch size × 1 byte)
│   └── Example: 65536 bytes = 64 KB
└── Device-side working memory (per-thread, allocated on demand)
    └── Argon2H' working buffers (not pre-allocated)

Total: ROM + ~6 MB per 65K salt batch
```

### Per-Thread Memory (Registers + Local)

```
Thread-local State (~518 KB per thread)
├── VM Struct (~750 bytes)
│   ├── Registers: 32 × u64 = 256 bytes
│   ├── Counters: ip (4), memory_counter (4), loop_counter (4) = 12 bytes
│   ├── prog_digest: Blake2bState = 232 bytes
│   ├── mem_digest: Blake2bState = 232 bytes
│   └── prog_seed: 64 bytes
├── Program Instructions: 256 × 20 = 5120 bytes
├── Argon2H' Working Memory: ~512 KB (temporary, reusable)
└── Stack/locals: ~100 bytes

Optimization: Keep VM struct in registers/L1, program in local memory
```

### Texture Memory (ROM)

```cuda
// ROM texture configuration
cudaTextureObject_t rom_texture;
cudaResourceDesc res_desc;
res_desc.resType = cudaResourceTypeLinear;
res_desc.res.linear.devPtr = d_rom;
res_desc.res.linear.sizeInBytes = rom_size;
res_desc.res.linear.desc = cudaCreateChannelDesc<uint4>();  // 128-bit access

cudaTextureDesc tex_desc;
tex_desc.readMode = cudaReadModeElementType;
tex_desc.filterMode = cudaFilterModePoint;
tex_desc.addressMode[0] = cudaAddressModeWrap;  // Modulo behavior

cudaCreateTextureObject(&rom_texture, &res_desc, &tex_desc, NULL);
```

**Benefits:**
- L1/L2 cache utilization
- Read-only optimization
- Hardware-accelerated modulo (address wrap mode)
- Coalesced access for nearby threads

---

## Kernel Design

### Kernel 1: ashmaize_mine (Main Mining Kernel)

**Purpose**: Compute hashes for batch of salts, find valid PoW solutions

**Launch configuration:**
```cuda
dim3 grid(num_blocks);       // Example: 256 blocks
dim3 block(threads_per_block); // Example: 256 threads
size_t shared_mem = 0;       // No shared memory needed

ashmaize_mine<<<grid, block, shared_mem>>>(
    rom_texture,              // Texture object for ROM
    d_salts,                  // Input: Salt array
    d_hashes,                 // Output: Hash results
    d_success_flags,          // Output: Success indicators
    difficulty_target,        // PoW difficulty threshold
    nb_loops,                 // Number of execution loops (8)
    nb_instrs,                // Instructions per loop (256)
    batch_size                // Number of salts to process
);
```

**Thread mapping:**
```
thread_id = blockIdx.x * blockDim.x + threadIdx.x
if (thread_id < batch_size) {
    salt = d_salts[thread_id]
    hash = ashmaize_hash(salt, rom_texture, nb_loops, nb_instrs)
    d_hashes[thread_id] = hash
    d_success_flags[thread_id] = (hash < difficulty_target) ? 1 : 0
}
```

**Signature:**
```cuda
__global__ void ashmaize_mine(
    cudaTextureObject_t rom_texture,
    const uint8_t* __restrict__ d_salts,
    uint8_t* __restrict__ d_hashes,
    uint8_t* __restrict__ d_success_flags,
    const uint8_t* __restrict__ difficulty_target,
    uint32_t nb_loops,
    uint32_t nb_instrs,
    uint32_t batch_size
);
```

---

### Kernel 2: ashmaize_verify (Optional Verification)

**Purpose**: Verify pre-computed hash matches expected result

**Use case**: Testing, validation, debugging

**Signature:**
```cuda
__global__ void ashmaize_verify(
    cudaTextureObject_t rom_texture,
    const uint8_t* __restrict__ d_salt,
    const uint8_t* __restrict__ d_expected_hash,
    uint8_t* __restrict__ d_result,  // 1 = match, 0 = mismatch
    uint32_t nb_loops,
    uint32_t nb_instrs
);
```

---

## Core Functions

### 1. VM Initialization

```cuda
__device__ void vm_init(
    VM* vm,
    const uint8_t* rom_digest,
    const uint8_t* salt,
    size_t salt_len,
    uint32_t nb_instrs
) {
    // Step 1: Prepare Argon2H' input (ROM digest + salt)
    uint8_t input[128];  // Max reasonable salt size
    memcpy(input, rom_digest, 64);
    memcpy(input + 64, salt, salt_len);
    
    // Step 2: Derive 448 bytes (registers + digests + seed)
    uint8_t init_buffer[448];
    argon2_hprime(init_buffer, 448, input, 64 + salt_len);
    
    // Step 3: Parse init_buffer
    // Bytes 0-255: Registers
    for (int i = 0; i < 32; i++) {
        vm->regs[i] = *((uint64_t*)&init_buffer[i * 8]);
    }
    
    // Bytes 256-319: prog_digest init
    blake2b_init_from_data(&vm->prog_digest, &init_buffer[256], 64);
    
    // Bytes 320-383: mem_digest init
    blake2b_init_from_data(&vm->mem_digest, &init_buffer[320], 64);
    
    // Bytes 384-447: prog_seed
    memcpy(vm->prog_seed, &init_buffer[384], 64);
    
    // Step 4: Initialize counters
    vm->ip = 0;
    vm->memory_counter = 0;
    vm->loop_counter = 0;
    
    // Step 5: Allocate program (zeroed)
    // Note: program.instructions must be allocated in local memory
    memset(vm->program.instructions, 0, nb_instrs * 20);
}
```

---

### 2. Program Shuffle

```cuda
__device__ void program_shuffle(
    Program* program,
    const uint8_t* seed
) {
    // Argon2H' to generate pseudo-random instruction bytes
    size_t program_size = program->nb_instrs * 20;
    argon2_hprime(
        program->instructions,
        program_size,
        seed,
        64
    );
}
```

---

### 3. Instruction Execution

```cuda
__device__ void execute_one_instruction(
    VM* vm,
    cudaTextureObject_t rom_texture
) {
    // Fetch instruction
    uint8_t instr_bytes[20];
    program_at(&vm->program, vm->ip, instr_bytes);
    
    // Decode
    Instruction instr;
    decode_instruction(instr_bytes, &instr);
    
    // Load operands
    uint64_t src1 = load_operand(vm, rom_texture, &instr, OPERAND_1);
    uint64_t src2 = load_operand(vm, rom_texture, &instr, OPERAND_2);
    
    // Execute
    uint64_t result = execute_operation(&instr, src1, src2, vm);
    
    // Store result
    vm->regs[instr.r3] = result;
    
    // Update prog_digest
    blake2b_update(&vm->prog_digest, instr_bytes, 20);
    
    // Increment IP
    vm->ip++;
}
```

---

### 4. Operand Loading

```cuda
__device__ uint64_t load_operand(
    VM* vm,
    cudaTextureObject_t rom_texture,
    const Instruction* instr,
    int operand_num  // 1 or 2
) {
    uint8_t op_type = (operand_num == 1) ? instr->op1_type : instr->op2_type;
    uint8_t reg_idx = (operand_num == 1) ? instr->r1 : instr->r2;
    uint64_t literal = (operand_num == 1) ? instr->lit1 : instr->lit2;
    
    switch (op_type) {
        case OPERAND_REG:
            return vm->regs[reg_idx];
        
        case OPERAND_MEMORY:
            return load_from_rom(vm, rom_texture, literal);
        
        case OPERAND_LITERAL:
            return literal;
        
        case OPERAND_SPECIAL1: {
            Blake2bState temp;
            blake2b_clone(&temp, &vm->prog_digest);
            uint8_t digest[64];
            blake2b_final(&temp, digest);
            return *((uint64_t*)digest);
        }
        
        case OPERAND_SPECIAL2: {
            Blake2bState temp;
            blake2b_clone(&temp, &vm->mem_digest);
            uint8_t digest[64];
            blake2b_final(&temp, digest);
            return *((uint64_t*)digest);
        }
    }
    return 0;  // Unreachable
}
```

---

### 5. ROM Access

```cuda
__device__ uint64_t load_from_rom(
    VM* vm,
    cudaTextureObject_t rom_texture,
    uint64_t address
) {
    // Fetch 64-byte chunk from texture
    uint32_t chunk_idx = (uint32_t)address;  // Modulo handled by texture wrap
    
    uint64_t chunk[8];  // 64 bytes = 8 × uint64_t
    
    // Read 4 × uint4 (4 × 16 bytes = 64 bytes)
    for (int i = 0; i < 4; i++) {
        uint4 data = tex1Dfetch<uint4>(rom_texture, chunk_idx * 4 + i);
        chunk[i*2 + 0] = ((uint64_t)data.y << 32) | data.x;
        chunk[i*2 + 1] = ((uint64_t)data.w << 32) | data.z;
    }
    
    // Update mem_digest
    blake2b_update(&vm->mem_digest, (uint8_t*)chunk, 64);
    
    // Increment counter
    vm->memory_counter++;
    
    // Extract 8 bytes based on counter
    int idx = (vm->memory_counter % 8);
    return chunk[idx];
}
```

---

### 6. Post-Instructions Mixing

```cuda
__device__ void post_instructions(VM* vm) {
    // Step 1: Sum all registers
    uint64_t sum = 0;
    for (int i = 0; i < 32; i++) {
        sum += vm->regs[i];
    }
    
    // Step 2: Finalize digests with sum
    Blake2bState temp_prog, temp_mem;
    blake2b_clone(&temp_prog, &vm->prog_digest);
    blake2b_clone(&temp_mem, &vm->mem_digest);
    
    blake2b_update(&temp_prog, (uint8_t*)&sum, 8);
    blake2b_update(&temp_mem, (uint8_t*)&sum, 8);
    
    uint8_t prog_value[64];
    uint8_t mem_value[64];
    blake2b_final(&temp_prog, prog_value);
    blake2b_final(&temp_mem, mem_value);
    
    // Step 3: Generate mixing seed
    Blake2bState mixing_state;
    blake2b_init(&mixing_state, NULL, 0);
    blake2b_update(&mixing_state, prog_value, 64);
    blake2b_update(&mixing_state, mem_value, 64);
    blake2b_update(&mixing_state, (uint8_t*)&vm->loop_counter, 4);
    
    uint8_t mixing_value[64];
    blake2b_final(&mixing_state, mixing_value);
    
    // Step 4: Generate mixing data (8192 bytes via Argon2H')
    uint8_t mixing_out[8192];
    argon2_hprime(mixing_out, 8192, mixing_value, 64);
    
    // Step 5: XOR registers with mixing data (32 rounds)
    for (int round = 0; round < 32; round++) {
        uint64_t* mixing_chunk = (uint64_t*)&mixing_out[round * 256];
        for (int i = 0; i < 32; i++) {
            vm->regs[i] ^= mixing_chunk[i];
        }
    }
    
    // Step 6: Update state
    memcpy(vm->prog_seed, prog_value, 64);
    vm->loop_counter++;
}
```

---

### 7. Finalization

```cuda
__device__ void vm_finalize(
    VM* vm,
    uint8_t* output_hash
) {
    // Finalize digests
    uint8_t prog_digest[64];
    uint8_t mem_digest[64];
    
    Blake2bState temp_prog, temp_mem;
    blake2b_clone(&temp_prog, &vm->prog_digest);
    blake2b_clone(&temp_mem, &vm->mem_digest);
    
    blake2b_final(&temp_prog, prog_digest);
    blake2b_final(&temp_mem, mem_digest);
    
    // Combine into final hash
    Blake2bState final_state;
    blake2b_init(&final_state, NULL, 0);
    blake2b_update(&final_state, prog_digest, 64);
    blake2b_update(&final_state, mem_digest, 64);
    blake2b_update(&final_state, (uint8_t*)&vm->memory_counter, 4);
    
    // Add all registers
    for (int i = 0; i < 32; i++) {
        blake2b_update(&final_state, (uint8_t*)&vm->regs[i], 8);
    }
    
    // Finalize to output
    blake2b_final(&final_state, output_hash);
}
```

---

### 8. Main Hash Function

```cuda
__device__ void ashmaize_hash(
    uint8_t* output_hash,
    const uint8_t* salt,
    size_t salt_len,
    const uint8_t* rom_digest,
    cudaTextureObject_t rom_texture,
    uint32_t nb_loops,
    uint32_t nb_instrs
) {
    // Allocate VM (local memory)
    VM vm;
    
    // Initialize
    vm_init(&vm, rom_digest, salt, salt_len, nb_instrs);
    
    // Execute loops
    for (uint32_t loop = 0; loop < nb_loops; loop++) {
        // Shuffle program
        program_shuffle(&vm.program, vm.prog_seed);
        
        // Execute instructions
        for (uint32_t i = 0; i < nb_instrs; i++) {
            execute_one_instruction(&vm, rom_texture);
        }
        
        // Post-processing
        post_instructions(&vm);
    }
    
    // Finalize
    vm_finalize(&vm, output_hash);
}
```

---

## Thread Organization

### Occupancy Analysis

**Per-thread resources:**
- Registers: ~50-60 (VM state, temporaries)
- Local memory: ~518 KB (VM + program + Argon2 working memory)
- Shared memory: 0 bytes

**GPU limits (Typical Ampere/Ada):**
- Max threads per SM: 2048
- Registers per SM: 65536
- Local memory per thread: No hard limit (backed by global memory)

**Occupancy calculation:**
```
Threads per block: 256
Blocks per SM: Limited by registers and local memory

Register limit: 65536 / 50 = ~1310 threads per SM
Theoretical occupancy: 1310 / 2048 = 64%

With 256 threads/block: 1310 / 256 = ~5 blocks per SM
Actual occupancy: 5 * 256 / 2048 = 62.5%
```

**Recommendation:**
- **Threads per block**: 256 (good balance)
- **Blocks per SM**: 4-5 (depends on register usage)
- **Total threads**: 2048+ (limited by GPU memory, not compute)

---

### Grid Configuration

**Strategy**: Maximize throughput, minimize latency

```cuda
// Calculate optimal grid size
int device;
cudaGetDevice(&device);

cudaDeviceProp props;
cudaGetDeviceProperties(&props, device);

int sm_count = props.multiProcessorCount;
int threads_per_block = 256;
int blocks_per_sm = 4;  // Conservative for occupancy
int total_threads = sm_count * blocks_per_sm * threads_per_block;

// Example: RTX 4090 (128 SMs)
// total_threads = 128 * 4 * 256 = 131,072 threads
// With 52ms per hash: ~2500 hashes/second

// Launch kernel
int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
ashmaize_mine<<<num_blocks, threads_per_block>>>(/* args */);
```

---

## Rust FFI Interface

### Structures

```rust
// src/gpu.rs

use std::os::raw::c_void;

#[repr(C)]
pub struct GpuRom {
    texture_object: u64,  // cudaTextureObject_t
    digest: [u8; 64],
    size: usize,
}

#[repr(C)]
pub struct GpuMinerConfig {
    nb_loops: u32,
    nb_instrs: u32,
    batch_size: u32,
    threads_per_block: u32,
}

#[repr(C)]
pub struct GpuMiningResult {
    hash: [u8; 64],
    salt: Vec<u8>,
    success: bool,
}
```

### API Functions

```rust
pub struct GpuMiner {
    rom: Option<GpuRom>,
    config: GpuMinerConfig,
}

impl GpuMiner {
    pub fn new(config: GpuMinerConfig) -> Result<Self, GpuError> {
        // Initialize CUDA, allocate resources
        unsafe { gpu_init() }?;
        Ok(Self { rom: None, config })
    }
    
    pub fn upload_rom(&mut self, rom: &Rom) -> Result<(), GpuError> {
        // Transfer ROM to GPU, create texture object
        let gpu_rom = unsafe {
            gpu_upload_rom(
                rom.data.as_ptr(),
                rom.data.len(),
                rom.digest.0.as_ptr()
            )
        }?;
        self.rom = Some(gpu_rom);
        Ok(())
    }
    
    pub fn mine_batch(
        &self,
        salts: &[Vec<u8>]
    ) -> Result<Vec<GpuMiningResult>, GpuError> {
        let rom = self.rom.as_ref().ok_or(GpuError::NoRom)?;
        
        // Prepare input
        let mut flat_salts = Vec::new();
        // ... flatten salts
        
        // Allocate output
        let mut hashes = vec![0u8; salts.len() * 64];
        let mut flags = vec![0u8; salts.len()];
        
        // Launch kernel
        unsafe {
            gpu_mine_batch(
                rom,
                flat_salts.as_ptr(),
                hashes.as_mut_ptr(),
                flags.as_mut_ptr(),
                self.config
            )
        }?;
        
        // Collect results
        let mut results = Vec::new();
        for (i, flag) in flags.iter().enumerate() {
            if *flag != 0 {
                results.push(GpuMiningResult {
                    hash: hashes[i*64..(i+1)*64].try_into().unwrap(),
                    salt: salts[i].clone(),
                    success: true,
                });
            }
        }
        
        Ok(results)
    }
}

impl Drop for GpuMiner {
    fn drop(&mut self) {
        // Cleanup CUDA resources
        unsafe { gpu_cleanup() };
    }
}
```

### C FFI Declarations

```rust
#[link(name = "ashmaize_cuda")]
extern "C" {
    fn gpu_init() -> i32;
    fn gpu_cleanup() -> i32;
    
    fn gpu_upload_rom(
        data: *const u8,
        size: usize,
        digest: *const u8
    ) -> GpuRom;
    
    fn gpu_mine_batch(
        rom: *const GpuRom,
        salts: *const u8,
        hashes: *mut u8,
        flags: *mut u8,
        config: GpuMinerConfig
    ) -> i32;
}
```

---

## Build System

### Project Structure

```
gpu-ashmaize/
├── Cargo.toml
├── build.rs                  # Build script for CUDA compilation
├── src/
│   ├── lib.rs               # Rust API
│   ├── ffi.rs               # C FFI declarations
│   └── error.rs             # Error handling
├── cuda/
│   ├── kernel.cu            # Main mining kernel
│   ├── vm.cu                # VM implementation
│   ├── instructions.cu      # Instruction execution
│   ├── blake2b.cu           # Blake2b implementation
│   ├── argon2.cu            # Argon2H' implementation
│   ├── common.cuh           # Common headers
│   └── Makefile             # CUDA build rules
├── tests/
│   ├── correctness.rs       # Test vectors
│   ├── performance.rs       # Benchmarks
│   └── integration.rs       # Full system tests
├── benches/
│   └── gpu_vs_cpu.rs        # Comparative benchmarks
├── examples/
│   ├── simple_mining.rs     # Basic usage
│   └── batch_mining.rs      # Advanced usage
└── README.md

```

### build.rs (CUDA Compilation)

```rust
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Compile CUDA code
    let cuda_dir = PathBuf::from("cuda");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    // Run nvcc
    let status = Command::new("nvcc")
        .args(&[
            "-arch=sm_75",  // Turing+
            "-std=c++17",
            "-O3",
            "-use_fast_math",
            "-lineinfo",
            "--ptxas-options=-v",
            "-Xcompiler", "-fPIC",
            "-o", out_dir.join("libashmaize_cuda.a").to_str().unwrap(),
            cuda_dir.join("kernel.cu").to_str().unwrap(),
            cuda_dir.join("vm.cu").to_str().unwrap(),
            cuda_dir.join("instructions.cu").to_str().unwrap(),
            cuda_dir.join("blake2b.cu").to_str().unwrap(),
            cuda_dir.join("argon2.cu").to_str().unwrap(),
        ])
        .status()
        .expect("Failed to compile CUDA code");
    
    assert!(status.success());
    
    // Link CUDA library
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=ashmaize_cuda");
    println!("cargo:rustc-link-lib=cudart");
    
    // Rerun if CUDA sources change
    println!("cargo:rerun-if-changed=cuda/");
}
```

---

## Testing Strategy

### Unit Tests (Per Component)

1. **Blake2b**
   - Empty input
   - Single block
   - Multi-block
   - Incremental vs one-shot
   - Test vectors from BLAKE2 spec

2. **Argon2H'**
   - Various output sizes (448, 5120, 8192 bytes)
   - Match cryptoxide outputs
   - Determinism (same input → same output)

3. **Instructions**
   - Each operation individually
   - Edge cases (div by zero, special operands)
   - All operand types

4. **VM**
   - Initialization
   - Instruction execution loop
   - Post-instructions mixing
   - Finalization

### Integration Tests

```rust
#[test]
fn test_vector_match() {
    let rom = Rom::new(
        b"123",
        RomGenerationType::TwoStep {
            pre_size: 16 * 1024,
            mixing_numbers: 4,
        },
        10 * 1024 * 1024,
    );
    
    let mut miner = GpuMiner::new(GpuMinerConfig::default()).unwrap();
    miner.upload_rom(&rom).unwrap();
    
    let salt = b"hello";
    let results = miner.mine_batch(&[salt.to_vec()]).unwrap();
    
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].hash, EXPECTED_HASH);
}
```

### Performance Tests

```rust
#[bench]
fn bench_gpu_mining(b: &mut Bencher) {
    let miner = setup_gpu_miner();
    let salts = generate_random_salts(1024);
    
    b.iter(|| {
        miner.mine_batch(&salts).unwrap()
    });
}
```

---

## Performance Optimization

### Phase 1: Correctness First
- Get any working implementation
- Validate against test vectors
- No optimization focus

### Phase 2: Profile-Guided
- Use Nsight Compute for profiling
- Identify hotspots
- Optimize top 3 bottlenecks

### Phase 3: Micro-Optimizations
- PTX intrinsics (brev, mul.hi)
- Loop unrolling
- Register allocation tuning
- Memory access coalescing

### Expected Performance

**CPU baseline (Rust):**
- Single hash: ~52ms
- Throughput: ~19 hashes/second

**GPU target (CUDA, RTX 4090):**
- Single hash: ~285ms per thread (5-6x slower)
- Parallel threads: 131,072
- Throughput: ~460 hashes/second per thread × 131K = **~60M hashes/hour**
- Speedup: **~40-60x** for mining workload

---

## Summary

### Architecture Highlights

✅ **Memory**: ROM in texture memory, VM state in registers/local
✅ **Parallelism**: Embarrassingly parallel salt mining
✅ **Crypto**: Custom Blake2b, ported Argon2H'
✅ **FFI**: Clean Rust interface, CUDA backend
✅ **Testing**: Comprehensive unit and integration tests
✅ **Build**: Automated CUDA compilation via build.rs

### Implementation Phases

1. **Foundation**: Project structure, build system
2. **Crypto**: Blake2b + Argon2H' implementations
3. **VM**: Core execution engine
4. **Integration**: Kernel + FFI + testing
5. **Optimization**: Profile and improve
6. **Production**: Documentation, packaging, release

### Next Steps

1. Create `gpu-ashmaize/` directory structure
2. Implement Blake2b (CUDA)
3. Port Argon2H' (CUDA)
4. Implement VM and instructions (CUDA)
5. Create Rust FFI layer
6. Test against CPU implementation
7. Optimize hot paths
8. Production release

**Status**: Architecture design complete, ready for implementation.
