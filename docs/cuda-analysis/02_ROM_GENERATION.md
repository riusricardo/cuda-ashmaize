# ROM Generation Deep Dive for CUDA Implementation

## Overview
The ROM (Read-Only Memory) is the memory-hard component of AshMaize, designed to resist ASIC optimization through large memory requirements and sequential generation. This document analyzes ROM generation mechanisms for CUDA implementation planning.

---

## ROM Structure

```rust
pub struct Rom {
    pub(crate) digest: RomDigest,  // 64-byte hash of entire ROM
    data: Vec<u8>,                  // The actual ROM data
}

pub(crate) struct RomDigest(pub(crate) [u8; 64]);
```

### ROM Properties
- **Size**: Configurable (typically 10MB - 10GB)
- **Access unit**: 64 bytes (DATASET_ACCESS_SIZE)
- **Generation**: Deterministic based on key
- **Usage**: Read-only during hash computation
- **Purpose**: ASIC resistance through memory bandwidth requirements

---

## Generation Types

### 1. FullRandom

```rust
RomGenerationType::FullRandom
```

**Algorithm:**
```rust
// Single sequential Argon2H' call
argon2::hprime(output, &seed);
RomDigest(Blake2b::<512>::new().update(output).finalize())
```

**Characteristics:**
- **Simplest approach**: Single Argon2H' invocation
- **Slowest**: Processes entire ROM size sequentially
- **Most memory-hard**: Every byte depends on all previous bytes
- **Generation time**: Proportional to ROM size (e.g., 10MB = ~1-2 seconds)

**CUDA considerations:**
- Cannot parallelize generation
- Must generate on CPU or suffer GPU sequential bottleneck
- Best to generate once, cache, reuse

---

### 2. TwoStep (Recommended)

```rust
RomGenerationType::TwoStep {
    pre_size: usize,         // Must be power of 2
    mixing_numbers: usize,   // Number of chunks to XOR (typically 4)
}
```

**Algorithm:**

```
Phase 1: Generate pre-memory buffer (pre_size bytes)
    mixing_buffer = argon2::hprime(pre_size, seed)

Phase 2: Generate offsets
    # Base offsets for each output chunk
    offsets_bytes[output.len()/64] = argon2::hprime(seed || "generation offset base")
    
    # Differential offsets for mixing
    offsets_diff[128] = 4 loops of Blake2b(seed || "generation offset" || i)
                        → 32 u16s per loop × 4 = 128 offsets

Phase 3: Expand to full ROM size
    for each 64-byte chunk in output:
        # Copy base chunk from pre-memory
        idx0 = (chunk_index % nb_source_chunks)
        chunk = mixing_buffer[idx0 * 64..(idx0+1) * 64]
        
        # XOR with (mixing_numbers - 1) additional chunks
        start_idx = offsets_bytes[chunk_index % offsets_bytes.len()] % nb_source_chunks
        for d in 1..mixing_numbers:
            idx = (start_idx + offsets_diff[(d-1) % 128]) % nb_source_chunks
            chunk ^= mixing_buffer[idx * 64..(idx+1) * 64]
        
        output[chunk_index] = chunk
        digest.update(chunk)
```

**Characteristics:**
- **Faster**: Pre-memory is smaller (e.g., 16KB vs 10MB)
- **Parallelizable expansion**: Phase 3 can be GPU-accelerated
- **Memory-hard**: Still requires full ROM for verification
- **Quality**: Statistically equivalent to FullRandom

**Example parameters:**
```rust
RomGenerationType::TwoStep {
    pre_size: 16 * 1024,    // 16 KB
    mixing_numbers: 4,       // XOR 4 chunks together
}
```

---

## Detailed TwoStep Analysis

### Phase 1: Pre-Memory Generation

```rust
let mut mixing_buffer = vec![0; pre_size];
argon2::hprime(&mut mixing_buffer, &seed);
```

**Input:**
- `seed`: 32 bytes from Blake2b-256 of key + size
- `pre_size`: Power of 2 (typically 16KB)

**Output:**
- `mixing_buffer`: Sequential Argon2H' output

**Time complexity**: O(pre_size)
**Memory**: pre_size bytes
**Parallelization**: None (sequential Argon2H')

**CUDA strategy**: 
- Generate on CPU (fast for 16KB)
- Upload to GPU constant/texture memory
- Reuse across all salt computations

---

### Phase 2: Offset Generation

#### 2a. Base Offsets

```rust
let mut offsets_bytes = vec![0; output.len() / 64];
let offset_bytes_input = Blake2b::<512>::new()
    .update(&seed)
    .update(b"generation offset base")
    .finalize();
argon2::hprime(&mut offsets_bytes, &offset_bytes_input);
```

**Purpose**: Determines starting index for each output chunk
**Size**: 1 byte per 64-byte output chunk
- 10MB ROM → 163,840 chunks → 163,840 bytes (~160KB)
**Type**: u8 values (0-255)
**Usage**: `start_idx = offsets_bytes[i % offsets_bytes.len()] % nb_source_chunks`

**CUDA strategy**: 
- Generate on CPU
- Upload to GPU texture memory
- Random access during expansion

#### 2b. Differential Offsets

```rust
const OFFSET_LOOPS: u32 = 4;
fn digest_to_u16s(digest: &[u8; 64]) -> impl Iterator<Item = u16> {
    digest.chunks(2)
        .map(|c| u16::from_le_bytes(...))
}

let mut offsets_diff = vec![];
for i in 0..OFFSET_LOOPS {
    let command = Blake2b::<512>::new()
        .update(&seed)
        .update(b"generation offset")
        .update(&i.to_le_bytes())
        .finalize();
    offsets_diff.extend(digest_to_u16s(&command))
}
// offsets_diff.len() = 32 * 4 = 128
```

**Purpose**: Additional indices for mixing chunks
**Size**: 128 u16 values (256 bytes)
**Type**: u16 (0-65535)
**Usage**: Indexed by `(d - 1) % 128` where d is mixing iteration
**Generation**: 4 × Blake2b-512 → 4 × 32 u16s = 128 offsets

**CUDA strategy**: 
- Tiny (256 bytes), can be in constant memory
- Generated on CPU
- Fast access pattern

---

### Phase 3: ROM Expansion

```rust
let nb_source_chunks = (pre_size / 64) as u32;
for (i, chunk) in output.chunks_mut(64).enumerate() {
    // Step 1: Compute base index
    let idx0 = (i as u32) % nb_source_chunks;
    let offset = idx0 * 64;
    chunk.copy_from_slice(&mixing_buffer[offset..offset + 64]);
    
    // Step 2: Compute starting position for mixing
    let start_idx = offsets_bytes[i % offsets_bytes.len()] as u32 % nb_source_chunks;
    
    // Step 3: XOR with additional chunks
    for d in 1..mixing_numbers {
        let idx = (start_idx + offsets_diff[(d - 1) % 128] as u32) % nb_source_chunks;
        let offset = idx * 64;
        xorbuf(chunk, &mixing_buffer[offset..offset + 64]);
    }
    
    // Step 4: Update digest
    digest.update_mut(chunk);
}
```

**Per-chunk operations:**
1. **Load base chunk**: Simple modulo indexing (sequential pattern)
2. **Compute mixing indices**: 2 lookups + modulo operations
3. **XOR mixing**: (mixing_numbers - 1) × 64-byte XOR operations
4. **Update digest**: Incremental Blake2b (sequential)

**Parallelization potential:**
- Steps 1-3 are **independent per chunk** → GPU parallelizable
- Step 4 (digest) is **sequential** → Must run on CPU or sequentially on GPU

**CUDA strategy:**
```
Hybrid approach:
1. Generate mixing_buffer on CPU (sequential Argon2H')
2. Upload mixing_buffer, offsets to GPU
3. GPU kernel: Parallel expansion (skip digest update)
4. CPU: Stream digest updates as chunks complete
OR: Accept non-parallelized digest (low overhead)
```

---

## xorbuf() Optimization

```rust
fn xorbuf(out: &mut [u8], input: &[u8]) {
    let input = input.as_ptr() as *const u64;
    let out = out.as_mut_ptr() as *mut u64;
    unsafe {
        *out.offset(0) ^= *input.offset(0);
        *out.offset(1) ^= *input.offset(1);
        *out.offset(2) ^= *input.offset(2);
        *out.offset(3) ^= *input.offset(3);
        *out.offset(4) ^= *input.offset(4);
        *out.offset(5) ^= *input.offset(5);
        *out.offset(6) ^= *input.offset(6);
        *out.offset(7) ^= *input.offset(7);
    }
}
```

**Analysis:**
- Processes 64 bytes as 8 × u64 XOR operations
- Unsafe pointer arithmetic for performance
- Unrolled loop for compiler optimization

**CUDA equivalent:**
```cuda
__device__ void xorbuf_gpu(uint64_t* out, const uint64_t* input) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i] ^= input[i];
    }
}
```

Or vectorized:
```cuda
__device__ void xorbuf_gpu(uint2* out, const uint2* input) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        out[i].x ^= input[i].x;
        out[i].y ^= input[i].y;
    }
}
```

---

## ROM Access During Hashing

### Access Interface

```rust
pub(crate) fn at(&self, i: u32) -> &[u8; DATASET_ACCESS_SIZE] {
    let start = i as usize % (self.data.len() / DATASET_ACCESS_SIZE);
    <&[u8; DATASET_ACCESS_SIZE]>::try_from(
        &self.data[start..start + DATASET_ACCESS_SIZE]
    ).unwrap()
}
```

**Access pattern:**
- Input: u32 index (typically from u64 literal in instruction)
- Output: 64-byte aligned chunk
- Wrapping: Modulo number of chunks (not number of bytes)
- Alignment: Always 64-byte aligned

**Example:**
```
ROM size: 10,485,760 bytes (10 MB)
Chunks: 10,485,760 / 64 = 163,840 chunks
Access: rom.at(500000) → chunk 500000 % 163840 = 8480 → bytes 542720..542784
```

### Memory Access Patterns in VM

```rust
macro_rules! mem_access64 {
    ($vm:ident, $rom:ident, $addr:ident) => {{
        let mem = rom.at($addr as u32);  // Get 64-byte chunk
        $vm.mem_digest.update_mut(mem);  // Update digest
        $vm.memory_counter = $vm.memory_counter.wrapping_add(1);
        
        // Extract 8 bytes based on access counter
        let idx = (($vm.memory_counter % (64 / 8)) as usize) * 8;
        u64::from_le_bytes(*<&[u8; 8]>::try_from(&mem[idx..idx + 8]).unwrap())
    }};
}
```

**Access characteristics:**
1. **Chunk selection**: Based on instruction literal (random)
2. **Byte selection**: Cycles through 8-byte chunks (0, 8, 16, 24, 32, 40, 48, 56)
3. **Side effects**: 
   - Updates mem_digest (64 bytes)
   - Increments counter
4. **Returns**: 8 bytes as u64

**CUDA memory strategy:**

| Strategy | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **Global memory** | Simple, large capacity | Slow (400-800 cycles) | ❌ Too slow |
| **Texture memory** | Cached, read-only | Limited size, cache thrashing | ✅ **Best choice** |
| **Constant memory** | Fast, broadcast | 64KB limit | ❌ Too small for ROM |
| **Shared memory** | Very fast | 48-96KB limit per SM | ❌ Too small for ROM |

**Texture memory configuration:**
```cuda
// Create texture object
cudaTextureObject_t rom_texture;
cudaResourceDesc rom_res;
rom_res.resType = cudaResourceTypeLinear;
rom_res.res.linear.devPtr = d_rom;
rom_res.res.linear.sizeInBytes = rom_size;
rom_res.res.linear.desc = cudaCreateChannelDesc<uint4>();  // 128-bit reads

cudaTextureDesc rom_desc;
rom_desc.readMode = cudaReadModeElementType;
cudaCreateTextureObject(&rom_texture, &rom_res, &rom_desc, NULL);

// Access in kernel
uint4 chunk = tex1Dfetch<uint4>(rom_texture, addr * 4);  // 4 × uint4 = 64 bytes
```

**Benefits:**
- L1/L2 cache automatically utilized
- Coalesced reads if multiple threads access nearby addresses
- Read-only optimization
- Handles large sizes (up to GPU memory limit)

---

## ROM Digest

```rust
pub(crate) struct RomDigest(pub(crate) [u8; 64]);
```

**Purpose**: 
- Fingerprint of entire ROM
- Used to initialize VM state (via Argon2H')
- Ensures ROM integrity

**Generation:**
- **FullRandom**: Blake2b-512 of entire output
- **TwoStep**: Incremental Blake2b during expansion phase

**CUDA usage:**
- ROM digest computed on CPU during ROM generation
- Passed to GPU kernels as constant parameter
- Each thread uses it for VM initialization

---

## Seed Generation

```rust
let seed = blake2b::Context::<256>::new()
    .update(&(data.len() as u32).to_le_bytes())  // 4 bytes: ROM size
    .update(key)                                   // Variable: User key
    .finalize();                                   // 32 bytes output
```

**Inputs:**
- ROM size (4 bytes, little-endian u32)
- Key (arbitrary bytes)

**Output**: 32-byte Blake2b-256 digest

**Properties:**
- Different keys → different ROMs
- Same key + size → same ROM (deterministic)
- Size included to prevent cross-size attacks

---

## TwoStep Parameters Analysis

### pre_size

**Constraints:**
- Must be power of 2
- Typically 16KB - 256KB

**Trade-offs:**

| pre_size | Generation time | Security | GPU expansion benefit |
|----------|----------------|----------|----------------------|
| 4 KB | Very fast (~1ms) | Lower mixing | Minimal benefit |
| 16 KB | Fast (~5ms) | Good | ✅ **Optimal** |
| 64 KB | Medium (~20ms) | Better | Good |
| 256 KB | Slow (~80ms) | Best | Diminishing returns |

**Recommendation**: **16 KB** (power of 2, fast CPU generation, sufficient mixing)

### mixing_numbers

**Effect**: Number of chunks XORed together per output chunk

**Trade-offs:**

| mixing_numbers | Generation time | Memory hardness | Pattern complexity |
|----------------|----------------|-----------------|-------------------|
| 2 | Fast | Low | Simple |
| 4 | Medium | Good | ✅ **Optimal** |
| 8 | Slower | Better | High |
| 16 | Slow | Best | Overkill |

**Recommendation**: **4** (balanced complexity, good mixing, reasonable speed)

---

## CUDA ROM Generation Strategy

### Option 1: CPU Generation (Recommended)

```
Flow:
1. CPU: Generate full ROM (TwoStep, 16KB pre-size, 4 mixing)
2. CPU: Compute RomDigest
3. GPU: Upload ROM to texture memory (one-time cost)
4. GPU: All threads share same ROM
5. Mining: Compute millions of salts against same ROM
```

**Pros:**
- Simple implementation
- ROM generation optimized (Rust/CPU)
- One-time upload cost amortized over many hashes
- Texture cache efficiency

**Cons:**
- Large GPU memory usage (10MB - 1GB)
- Upload time (~50ms for 10MB over PCIe)

**Best for**: PoW mining (many hashes per ROM)

### Option 2: Hybrid Generation

```
Flow:
1. CPU: Generate pre-memory (16KB)
2. CPU: Generate offsets (160KB + 256 bytes)
3. GPU: Upload pre-memory and offsets
4. GPU: Parallel expansion kernel (skip digest)
5. CPU: Compute digest incrementally
```

**Pros:**
- Smaller upload (16KB vs 10MB)
- Faster ROM switching
- GPU utilization during expansion

**Cons:**
- More complex implementation
- Digest computation tricky
- Marginal benefit if ROM rarely changes

**Best for**: Frequent ROM changes (e.g., per-user ROMs)

### Option 3: Pure GPU Generation

```
Flow:
1. GPU: Sequential Argon2H' for pre-memory
2. GPU: Parallel expansion
3. GPU: Sequential digest computation
```

**Pros:**
- No data uploads (except seed)
- Maximum flexibility

**Cons:**
- Complex GPU Argon2H' implementation
- Sequential bottleneck on GPU (inefficient)
- No performance benefit

**Best for**: Research/experimentation only

---

## Memory Consumption Analysis

### FullRandom Example

**ROM size**: 10 MB
```
Generation:
- Output buffer: 10,485,760 bytes
- Argon2H' working memory: ~10 MB (depends on impl)
- Total: ~20 MB peak

Storage:
- ROM data: 10,485,760 bytes
- ROM digest: 64 bytes
- Total: ~10 MB
```

### TwoStep Example

**ROM size**: 10 MB, **pre_size**: 16 KB, **mixing_numbers**: 4
```
Generation:
- mixing_buffer: 16,384 bytes
- offsets_bytes: 163,840 bytes
- offsets_diff: 256 bytes
- Output buffer: 10,485,760 bytes
- Total: ~10.66 MB peak

Storage:
- ROM data: 10,485,760 bytes
- ROM digest: 64 bytes
- Total: ~10 MB
```

**GPU memory (TwoStep):**
```
Option 1 (Full ROM):
- Texture memory: 10 MB
- Constant memory: 64 bytes (digest)
- Total: 10 MB

Option 2 (Hybrid):
- Texture memory: 16 KB (mixing_buffer) + 164 KB (offsets)
- Constant memory: 64 bytes (digest)
- Working memory: 10 MB per thread-block (generated on-the-fly)
- Total: ~180 KB + working memory
```

---

## Summary for CUDA Implementation

### ROM Generation
- **Recommended**: TwoStep with pre_size=16KB, mixing_numbers=4
- **Where**: CPU (fast, optimized, simple)
- **Upload**: Once per ROM change
- **Storage**: GPU texture memory

### ROM Access
- **Method**: Texture memory with caching
- **Pattern**: Random 64-byte chunks
- **Frequency**: ~30-40% of instructions trigger ROM access
- **Optimization**: Texture cache handles random access well

### Key Insights
1. **Sequential bottleneck**: Argon2H' for pre-memory (~5ms for 16KB)
2. **Parallel potential**: ROM expansion (negligible time for 10MB)
3. **Upload cost**: ~50ms for 10MB ROM (one-time per mining session)
4. **Memory efficiency**: TwoStep reduces generation time by 100-200x

### Decision Matrix

| Use Case | ROM Size | Generation | Upload | Strategy |
|----------|----------|------------|--------|----------|
| **PoW Mining** | 10 MB - 1 GB | Once | Once | CPU gen + GPU texture ✅ |
| **Frequent ROM change** | 10 MB - 100 MB | Multiple | Multiple | Hybrid (CPU pre, GPU expand) |
| **Research** | Small | Variable | N/A | Any |

### Next: Cryptographic Primitives Analysis
Now that ROM generation is understood, next analyze Blake2b and Argon2H' usage patterns throughout the hash computation for optimal CUDA crypto library selection.
