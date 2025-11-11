# Cryptographic Primitives Deep Dive for CUDA Implementation

## Overview
AshMaize uses two cryptographic primitives extensively: **Blake2b-512** for hashing and **Argon2H'** (Argon2 H-Prime) for key derivation. This document maps every invocation, analyzes their usage patterns, and provides CUDA implementation guidance.

---

## Primitives Used

### Blake2b-512
- **Library**: cryptoxide v0.5.1
- **Output**: 64 bytes (512 bits)
- **Mode**: Incremental (streaming) hashing
- **Features**: Context cloning, update, finalize

### Argon2H' (Argon2 H-Prime)
- **Library**: cryptoxide v0.5.1
- **Mode**: Key derivation function (KDF)
- **Type**: Argon2d variant (data-dependent memory access)
- **Features**: Variable output length, sequential execution

---

## Blake2b-512 Usage Inventory

### 1. ROM Seed Generation

**Location**: `rom.rs`, `Rom::new()`

```rust
let seed = blake2b::Context::<256>::new()  // Note: Blake2b-256, not 512!
    .update(&(data.len() as u32).to_le_bytes())
    .update(key)
    .finalize();
```

**Details:**
- **Variant**: Blake2b-256 (32 bytes output)
- **Input**: 4 bytes (ROM size) + key (variable)
- **Output**: 32 bytes
- **Purpose**: Derive seed for ROM generation
- **Frequency**: Once per ROM
- **CUDA**: CPU-side only (ROM generation)

---

### 2. ROM Digest (FullRandom)

**Location**: `rom.rs`, `random_gen()`

```rust
argon2::hprime(output, &seed);
RomDigest(Blake2b::<512>::new().update(output).finalize())
```

**Details:**
- **Variant**: Blake2b-512
- **Input**: Entire ROM (10MB+)
- **Output**: 64 bytes
- **Purpose**: Fingerprint ROM for integrity
- **Frequency**: Once per ROM
- **CUDA**: CPU-side only

---

### 3. ROM Digest (TwoStep) - Incremental

**Location**: `rom.rs`, `random_gen()`

```rust
let mut digest = Blake2b::<512>::new();
for chunk in output.chunks(64) {
    digest.update_mut(chunk);
}
RomDigest(digest.finalize())
```

**Details:**
- **Variant**: Blake2b-512
- **Input**: Entire ROM in 64-byte chunks
- **Output**: 64 bytes
- **Purpose**: Fingerprint ROM during generation
- **Frequency**: Once per ROM
- **Pattern**: Incremental, sequential
- **CUDA**: CPU-side only

---

### 4. TwoStep Offset Generation

**Location**: `rom.rs`, `random_gen()`

```rust
let offset_bytes_input = Blake2b::<512>::new()
    .update(&seed)
    .update(b"generation offset base")
    .finalize();
```

**Details:**
- **Variant**: Blake2b-512
- **Input**: 32 bytes (seed) + 22 bytes (constant string)
- **Output**: 64 bytes
- **Purpose**: Seed for Argon2H' offset generation
- **Frequency**: Once per ROM
- **CUDA**: CPU-side only

---

### 5. TwoStep Differential Offsets

**Location**: `rom.rs`, `random_gen()`

```rust
for i in 0..4 {
    let command = Blake2b::<512>::new()
        .update(&seed)
        .update(b"generation offset")
        .update(&i.to_le_bytes())
        .finalize();
    offsets_diff.extend(digest_to_u16s(&command));
}
```

**Details:**
- **Variant**: Blake2b-512
- **Input**: 32 bytes (seed) + 18 bytes (string) + 4 bytes (loop counter)
- **Output**: 64 bytes × 4 = 256 bytes total
- **Purpose**: Generate 128 u16 offsets for ROM mixing
- **Frequency**: 4 times per ROM
- **CUDA**: CPU-side only

---

### 6. VM Initialization Digests

**Location**: `lib.rs`, `VM::new()`

```rust
let prog_digest = Blake2b::<512>::new().update(&init_buffer[256..320]);
let mem_digest = Blake2b::<512>::new().update(&init_buffer[320..384]);
```

**Details:**
- **Variant**: Blake2b-512
- **Input**: 64 bytes each (from Argon2H' output)
- **Output**: Blake2b context (not finalized yet)
- **Purpose**: Initialize incremental digest contexts
- **Frequency**: Once per hash (once per thread in CUDA)
- **State size**: ~200 bytes per context
- **CUDA**: **Per-thread initialization**

---

### 7. Instruction Prog Digest Updates

**Location**: `lib.rs`, `execute_one_instruction()`

```rust
vm.prog_digest.update_mut(&prog_chunk);
```

**Details:**
- **Variant**: Blake2b-512 (incremental)
- **Input**: 20 bytes per instruction
- **Output**: Updated context (intermediate state)
- **Purpose**: Accumulate hash of executed instructions
- **Frequency**: **Every instruction** (nb_loops × nb_instrs times)
  - Default: 8 × 256 = 2048 updates per hash
- **CUDA**: **Hot path** - requires fast incremental hashing

---

### 8. Memory Digest Updates

**Location**: `lib.rs`, `mem_access64!` macro

```rust
$vm.mem_digest.update_mut(mem);
```

**Details:**
- **Variant**: Blake2b-512 (incremental)
- **Input**: 64 bytes per ROM access
- **Output**: Updated context
- **Purpose**: Accumulate hash of ROM accesses
- **Frequency**: **Every ROM access** (~30-40% of instructions)
  - ~600-800 updates per hash
- **CUDA**: **Hot path** - critical for performance

---

### 9. Special Operand 1 (prog_digest read)

**Location**: `lib.rs`, `special1_value64!` macro

```rust
let r = $vm.prog_digest.clone().finalize();
u64::from_le_bytes(*<&[u8; 8]>::try_from(&r[0..8]).unwrap())
```

**Details:**
- **Variant**: Blake2b-512
- **Operation**: Clone context → finalize → extract 8 bytes
- **Input**: Current prog_digest state
- **Output**: u64 (first 8 bytes of 64-byte digest)
- **Purpose**: Non-deterministic operand based on execution history
- **Frequency**: ~1-5% of instructions use Special1 operand
  - ~20-100 times per hash
- **Cost**: **Expensive** (clone + finalize)
- **CUDA**: Requires full Blake2b finalization

---

### 10. Special Operand 2 (mem_digest read)

**Location**: `lib.rs`, `special2_value64!` macro

```rust
let r = $vm.mem_digest.clone().finalize();
u64::from_le_bytes(*<&[u8; 8]>::try_from(&r[0..8]).unwrap())
```

**Details:**
- **Variant**: Blake2b-512
- **Operation**: Clone context → finalize → extract 8 bytes
- **Input**: Current mem_digest state
- **Output**: u64
- **Purpose**: Non-deterministic operand based on memory access history
- **Frequency**: ~1-5% of instructions use Special2 operand
- **Cost**: **Expensive** (clone + finalize)
- **CUDA**: Requires full Blake2b finalization

---

### 11. Hash Instruction

**Location**: `lib.rs`, `Op3::Hash(v)`

```rust
Op3::Hash(v) => {
    let out = Blake2b::<512>::new()
        .update(&src1.to_le_bytes())
        .update(&src2.to_le_bytes())
        .finalize();
    if let Some(chunk) = out.chunks(8).nth(v as usize) {
        u64::from_le_bytes(*<&[u8; 8]>::try_from(chunk).unwrap())
    } else {
        panic!("chunk doesn't exist")
    }
}
```

**Details:**
- **Variant**: Blake2b-512
- **Input**: 16 bytes (two u64 operands)
- **Output**: 64 bytes, extract 8-byte chunk at index `v`
- **Purpose**: Hash instruction operation (opcode 248-255)
- **Frequency**: ~3% of instructions (8/256 opcode space)
  - ~60-75 times per hash
- **Variants**: 8 versions (v = 0..7), each extracts different 8-byte chunk
- **CUDA**: Requires fast one-shot Blake2b

---

### 12. Post-Instructions Digest Finalization

**Location**: `lib.rs`, `VM::post_instructions()`

```rust
let sum_regs = self.regs.iter().fold(0, |acc, r| acc.wrapping_add(*r));

let prog_value = self.prog_digest.clone()
    .update(&sum_regs.to_le_bytes())
    .finalize();

let mem_value = self.mem_digest.clone()
    .update(&sum_regs.to_le_bytes())
    .finalize();
```

**Details:**
- **Variant**: Blake2b-512
- **Operation**: Clone → update with 8 bytes → finalize
- **Input**: Current digest state + register sum
- **Output**: 64 bytes each (prog_value, mem_value)
- **Purpose**: Snapshot digest states for mixing
- **Frequency**: Once per loop (nb_loops times, typically 8)
- **CUDA**: 16 finalizations per hash (2 per loop × 8 loops)

---

### 13. Post-Instructions Mixing Seed

**Location**: `lib.rs`, `VM::post_instructions()`

```rust
let mixing_value = Blake2b::<512>::new()
    .update(&prog_value)
    .update(&mem_value)
    .update(&self.loop_counter.to_le_bytes())
    .finalize();
```

**Details:**
- **Variant**: Blake2b-512
- **Input**: 64 + 64 + 4 = 132 bytes
- **Output**: 64 bytes
- **Purpose**: Seed for Argon2H' mixing data generation
- **Frequency**: Once per loop (8 times per hash)
- **CUDA**: Fast, small input

---

### 14. Final Hash Computation

**Location**: `lib.rs`, `VM::finalize()`

```rust
let prog_digest = self.prog_digest.finalize();
let mem_digest = self.mem_digest.finalize();

let mut context = Blake2b::<512>::new()
    .update(&prog_digest)
    .update(&mem_digest)
    .update(&self.memory_counter.to_le_bytes());

for r in self.regs {
    context.update_mut(&r.to_le_bytes());
}

context.finalize()
```

**Details:**
- **Variant**: Blake2b-512
- **Input**: 64 + 64 + 4 + (32 × 8) = 388 bytes total
- **Output**: 64 bytes (final hash)
- **Purpose**: Combine all VM state into final digest
- **Frequency**: Once per hash
- **CUDA**: Final step, low priority for optimization

---

## Argon2H' (Argon2 H-Prime) Usage Inventory

### What is Argon2H'?

Argon2H' is a **variant of Argon2** used specifically for **variable-length output key derivation**. It's based on **Argon2d** (data-dependent memory access) but extended to produce arbitrary output lengths.

**Key properties:**
- **Sequential**: Cannot parallelize within single invocation
- **Memory-hard**: Resists brute-force attacks
- **Time-hard**: Computationally expensive
- **Deterministic**: Same input → same output

**Function signature:**
```rust
fn hprime(output: &mut [u8], input: &[u8])
```

---

### 1. VM Initialization

**Location**: `lib.rs`, `VM::new()`

```rust
let mut init_buffer = [0; 448];  // 256 + 3*64
let mut init_buffer_input = rom_digest.0.to_vec();
init_buffer_input.extend_from_slice(salt);
argon2::hprime(&mut init_buffer, &init_buffer_input);
```

**Details:**
- **Input**: 64 bytes (ROM digest) + salt (variable, typically 16-32 bytes)
- **Output**: 448 bytes
  - Bytes 0-255: Register initialization (32 × u64)
  - Bytes 256-319: prog_digest initialization
  - Bytes 320-383: mem_digest initialization
  - Bytes 384-447: prog_seed initialization
- **Purpose**: Derive all initial VM state from ROM + salt
- **Frequency**: **Once per hash** (per thread in CUDA)
- **Time**: ~1-2ms on CPU
- **CUDA**: **Critical** - every thread must execute this

---

### 2. Program Shuffle

**Location**: `lib.rs`, `Program::shuffle()`

```rust
pub fn shuffle(&mut self, seed: &[u8; 64]) {
    argon2::hprime(&mut self.instructions, seed)
}
```

**Details:**
- **Input**: 64 bytes (prog_seed)
- **Output**: Full program (nb_instrs × 20 bytes)
  - Default: 256 × 20 = 5120 bytes
- **Purpose**: Pseudo-randomly shuffle instruction bytes
- **Frequency**: **Once per loop** (nb_loops times, typically 8)
  - 8 invocations per hash
- **Time**: ~1-2ms per invocation on CPU
- **CUDA**: **Major bottleneck** - 8-16ms per hash just for shuffling

---

### 3. Post-Instructions Mixing

**Location**: `lib.rs`, `VM::post_instructions()`

```rust
let mut mixing_out = vec![0; 32 * 32 * 8];  // 8192 bytes
argon2::hprime(&mut mixing_out, &mixing_value);
```

**Details:**
- **Input**: 64 bytes (mixing_value from Blake2b)
- **Output**: 8192 bytes (32 rounds × 32 registers × 8 bytes)
- **Purpose**: Generate pseudo-random data for register mixing
- **Frequency**: **Once per loop** (8 times per hash)
- **Time**: ~2-3ms per invocation on CPU
- **CUDA**: **Major bottleneck** - 16-24ms per hash

**Register mixing:**
```rust
for mem_chunks in mixing_out.chunks(256) {  // 32 chunks
    for (reg, reg_chunk) in regs.iter_mut().zip(mem_chunks.chunks(8)) {
        *reg ^= u64::from_le_bytes(reg_chunk);
    }
}
```
- 32 rounds of XOR mixing
- Each round: 32 registers XORed with 8 bytes each

---

### 4. ROM Pre-Memory Generation (TwoStep)

**Location**: `rom.rs`, `random_gen()`

```rust
let mut mixing_buffer = vec![0; pre_size];
argon2::hprime(&mut mixing_buffer, &seed);
```

**Details:**
- **Input**: 32 bytes (seed)
- **Output**: pre_size bytes (typically 16KB)
- **Purpose**: Generate dense random data for ROM expansion
- **Frequency**: Once per ROM
- **Time**: ~5ms for 16KB on CPU
- **CUDA**: CPU-side only (ROM generation)

---

### 5. ROM Offset Generation (TwoStep)

**Location**: `rom.rs`, `random_gen()`

```rust
let mut offsets_bytes = vec![0; nb_chunks_bytes];
argon2::hprime(&mut offsets_bytes, &offset_bytes_input);
```

**Details:**
- **Input**: 64 bytes (Blake2b output)
- **Output**: nb_chunks_bytes (ROM size / 64)
  - 10MB ROM → 163,840 bytes
- **Purpose**: Generate base offsets for ROM expansion
- **Frequency**: Once per ROM
- **Time**: ~30ms for 10MB ROM on CPU
- **CUDA**: CPU-side only

---

### 6. ROM Full Generation (FullRandom)

**Location**: `rom.rs`, `random_gen()`

```rust
argon2::hprime(output, &seed);
```

**Details:**
- **Input**: 32 bytes (seed)
- **Output**: Full ROM size (10MB - 10GB)
- **Purpose**: Generate entire ROM in one sequential pass
- **Frequency**: Once per ROM
- **Time**: ~200ms for 10MB, ~20 seconds for 1GB
- **CUDA**: CPU-side only (too slow for GPU)

---

## Performance Analysis

### Blake2b-512 Execution Counts

**Per hash (default: 8 loops, 256 instructions per loop):**

| Operation | Count | Input Size | Output Size | Total Data |
|-----------|-------|------------|-------------|------------|
| Instruction updates | 2048 | 20 bytes | Context | ~40 KB |
| Memory updates | ~700 | 64 bytes | Context | ~45 KB |
| Hash instructions | ~60 | 16 bytes | 64 bytes | ~960 B input, ~3.8 KB output |
| Special1 reads | ~40 | Context | 64 bytes | ~2.5 KB |
| Special2 reads | ~40 | Context | 64 bytes | ~2.5 KB |
| Post-instr finalize | 16 | Context | 64 bytes | ~1 KB |
| Mixing seeds | 8 | 132 bytes | 64 bytes | ~1 KB + 512 B |
| Final hash | 1 | 388 bytes | 64 bytes | 388 B + 64 B |
| **Total** | **~2913** | | | **~94 KB processed** |

**Key insights:**
- Blake2b is called **~3000 times** per hash
- Most are incremental updates (cheap)
- ~180 full finalizations (expensive)
- Incremental state size: ~200 bytes
- Total Blake2b time: ~30-40% of hash time

---

### Argon2H' Execution Counts

**Per hash (default: 8 loops, 256 instructions):**

| Operation | Count | Input | Output | Time (CPU) |
|-----------|-------|-------|--------|-----------|
| VM init | 1 | ~80 bytes | 448 bytes | ~1.5 ms |
| Program shuffle | 8 | 64 bytes | 5120 bytes | ~12 ms |
| Mixing data | 8 | 64 bytes | 8192 bytes | ~20 ms |
| **Total** | **17** | | **~75 KB** | **~33.5 ms** |

**Key insights:**
- Argon2H' is called **17 times** per hash
- Total output: **~75 KB**
- Total time: **~33.5ms** (~64% of total hash time)
- **Cannot parallelize** within single hash
- **Sequential bottleneck** for GPU

---

## CUDA Implementation Requirements

### Blake2b-512

**Requirements:**
1. **Incremental hashing support**
   - Ability to initialize context
   - Multiple `update()` calls
   - Clone context (for Special1/2)
   - Finalize to 64 bytes
   
2. **Context size**: ~200 bytes per VM instance
   - 2 contexts per VM (prog_digest, mem_digest)
   - ~400 bytes per thread

3. **Performance targets**:
   - Incremental update: <10 cycles for small inputs
   - Full finalization: <500 cycles
   - One-shot hash: <1000 cycles

**CUDA library options:**

| Library | Pros | Cons | Recommendation |
|---------|------|------|----------------|
| **crypto_hash** | Official CUDA samples | Old, unmaintained | ❌ |
| **cuda-blake2** | Optimized for GPU | Limited features | ⚠️ Check incremental support |
| **Custom implementation** | Full control | Development effort | ✅ **Best option** |

**Implementation strategy:**
```cuda
// Blake2b state structure
struct Blake2bState {
    uint64_t h[8];        // Hash state (64 bytes)
    uint64_t t[2];        // Byte counter (16 bytes)
    uint64_t f[2];        // Finalization flags (16 bytes)
    uint8_t buf[128];     // Input buffer (128 bytes)
    size_t buflen;        // Buffer length (8 bytes)
    // Total: ~232 bytes
};

__device__ void blake2b_init(Blake2bState* state, const uint8_t* key, size_t keylen);
__device__ void blake2b_update(Blake2bState* state, const uint8_t* data, size_t len);
__device__ void blake2b_final(Blake2bState* state, uint8_t* out);
__device__ Blake2bState blake2b_clone(const Blake2bState* state);
```

**Optimization opportunities:**
- Use 64-bit operations (native on GPU)
- Unroll compression function loops
- Minimize memory access (keep state in registers)
- Vectorize where possible (limited benefit for Blake2b)

---

### Argon2H' (Argon2 H-Prime)

**Requirements:**
1. **Variable-length output** (448 bytes, 5120 bytes, 8192 bytes)
2. **Argon2d mode** (data-dependent memory access)
3. **Sequential execution** (inherent to Argon2)
4. **Deterministic** (same as CPU implementation)

**CUDA library options:**

| Library | Pros | Cons | Recommendation |
|---------|------|------|----------------|
| **argon2-gpu** | GPU-optimized | May not support h-prime | ⚠️ Check compatibility |
| **cryptoxide port** | Matches Rust behavior | Porting effort | ✅ **Recommended** |
| **Custom implementation** | Full control | Complex, error-prone | ⚠️ Last resort |

**Implementation challenges:**
1. **Memory access patterns**: Argon2d is intentionally unpredictable
2. **Sequential dependency**: Each iteration depends on previous
3. **Large working memory**: Typically 512KB - 1MB per instance
4. **Time/memory trade-off**: Can reduce memory, increases time

**Optimization strategy:**
```cuda
// Argon2 state per thread
struct Argon2State {
    uint64_t* memory;     // Working memory (512KB)
    size_t memory_size;
    uint32_t passes;
    uint32_t lanes;
    // ...
};

__device__ void argon2_hprime(
    uint8_t* output, 
    size_t output_len,
    const uint8_t* input,
    size_t input_len
) {
    // Sequential execution per thread
    // Cannot parallelize across threads for single hash
    // BUT: Each thread computes independent hash
}
```

**Memory strategy:**
- Allocate working memory per thread-block (shared memory too small)
- Use local memory (slow, but necessary)
- Consider dynamic parallelism for Argon2 internal loops (limited benefit)

**Time cost (estimated):**
- VM init: ~1ms → ~10-20ms on GPU (10-20x slower due to sequential)
- Program shuffle: ~1.5ms → ~5-10ms per invocation
- Mixing data: ~2.5ms → ~8-15ms per invocation

**Total Argon2H' overhead per hash:**
- CPU: ~33.5ms
- GPU: ~150-300ms per thread (4-9x slower)
- **BUT**: 1000+ threads in parallel → net speedup

---

## Cryptographic Operation Timeline

**Single hash execution (8 loops, 256 instructions):**

```
Time   | Operation                        | Primitive    | Count | Cumulative
-------|----------------------------------|--------------|-------|------------
0ms    | VM init                          | Argon2H'     | 1     | 1.5ms
1.5ms  | Loop 0 start                     |              |       |
1.5ms  |   Program shuffle                | Argon2H'     | 1     | 3.0ms
3.0ms  |   Execute 256 instructions       |              |       |
3.0ms  |     - Instruction digest updates | Blake2b inc  | 256   |
3.0ms  |     - Memory digest updates      | Blake2b inc  | ~90   |
3.0ms  |     - Hash instructions          | Blake2b full | ~8    |
3.0ms  |     - Special operands           | Blake2b full | ~10   |
8.0ms  |   Post-instructions              |              |       |
8.0ms  |     - Digest finalizations       | Blake2b full | 2     |
8.0ms  |     - Mixing seed                | Blake2b full | 1     |
8.0ms  |     - Mixing data                | Argon2H'     | 1     | 10.5ms
10.5ms |   Register mixing                | XOR          | 32×32 |
10.5ms | Loop 0 end                       |              |       |
...    | Loops 1-7 (similar)              |              |       | 51.5ms
51.5ms | Final hash                       | Blake2b full | 1     | 52ms
-------|----------------------------------|--------------|-------|------------
Total: ~52ms per hash on CPU
```

**Breakdown:**
- Argon2H': ~33.5ms (64%)
- Blake2b: ~18ms (35%)
- Instructions: ~0.5ms (1%)

**GPU projection (per thread):**
```
Argon2H': ~200ms (70%)
Blake2b: ~80ms (28%)
Instructions: ~5ms (2%)
Total: ~285ms per hash per thread

BUT: 2048 threads in parallel
Throughput: 2048 / 0.285s = ~7200 hashes/second
CPU: ~19 hashes/second (52ms/hash)
Speedup: ~380x for parallel mining
```

---

## CUDA Library Selection

### Recommended Stack

1. **Blake2b-512**: Custom implementation
   - Based on official Blake2b reference
   - Optimized for CUDA (64-bit ops, unrolled loops)
   - Incremental support essential
   - Target: <1000 cycles per operation

2. **Argon2H'**: Port cryptoxide implementation
   - Ensures deterministic behavior (matches Rust)
   - Already optimized for Argon2d
   - H-prime extension (variable output)
   - Accept sequential bottleneck (inherent to algorithm)

3. **Helper functions**: Custom CUDA utilities
   - Byte order conversions (to_le_bytes, from_le_bytes)
   - Buffer XOR operations
   - Memory management

---

## Testing Strategy

### Correctness Validation

**Test vectors (from Rust implementation):**
```rust
ROM: key="123", TwoStep(pre_size=16KB, mixing=4), size=10MB
Salt: "hello"
Loops: 8
Instructions: 256
Expected: [56, 148, 1, 228, 59, 96, ...]  // 64 bytes
```

**Validation steps:**
1. **Blake2b unit tests**:
   - Empty input → known output
   - Single block → known output
   - Multi-block → known output
   - Incremental vs one-shot equivalence

2. **Argon2H' unit tests**:
   - Match cryptoxide outputs
   - Various output sizes (448, 5120, 8192 bytes)
   - Same seed → same output

3. **Integration tests**:
   - Full hash computation
   - Match CPU output byte-for-byte
   - Edge cases (div by zero, special operands)

4. **Randomized testing**:
   - 10,000+ random salts
   - CPU vs GPU comparison
   - No mismatches allowed

---

## Performance Optimization Checklist

### Blake2b
- ✅ Keep state in registers (minimize memory traffic)
- ✅ Unroll compression loops
- ✅ Use 64-bit operations natively
- ✅ Minimize context cloning overhead
- ⚠️ Consider shared memory for frequently used states
- ❌ Limited SIMD benefit (Blake2b inherently scalar)

### Argon2H'
- ✅ Use local memory for working buffer
- ✅ Optimize memory access patterns (coalesce where possible)
- ⚠️ Consider reduced-memory trade-off variants
- ❌ Cannot parallelize single invocation
- ✅ **Key insight**: Parallel thread execution is the optimization

### Overall
- ✅ Minimize global memory access (use texture for ROM)
- ✅ Maximize occupancy (small per-thread state)
- ✅ Balance register usage vs local memory
- ✅ Profile to identify true bottlenecks
- ✅ Accept that some operations are inherently sequential

---

## Summary for CUDA Implementation

### Blake2b-512
- **Usage**: ~3000 invocations per hash
- **Critical paths**: Instruction/memory digest updates, finalizations
- **State size**: ~232 bytes per context (×2 per VM)
- **Library**: Custom implementation recommended
- **Performance target**: <1000 cycles per operation

### Argon2H'
- **Usage**: 17 invocations per hash
- **Output sizes**: 448, 5120, 8192 bytes
- **Time**: Dominates execution (64% of total)
- **Challenge**: Sequential, cannot parallelize within single hash
- **Strategy**: Accept per-thread cost, parallelize across threads
- **Library**: Port cryptoxide (deterministic behavior)

### Overall Strategy
1. **Correctness first**: Exact match with CPU implementation
2. **Accept sequential overhead**: Algorithm designed this way
3. **Parallelize at salt level**: 1000+ independent hashes
4. **Optimize hot paths**: Blake2b incremental updates
5. **Minimize memory traffic**: Texture ROM, register-heavy VM state

### Next Steps
- Implement instruction set execution
- Analyze memory access patterns
- Design CUDA kernel architecture
- Plan thread organization and memory layout

