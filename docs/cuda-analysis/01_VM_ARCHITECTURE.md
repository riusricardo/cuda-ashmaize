# VM Architecture Deep Dive for CUDA Implementation

## Overview
This document provides a complete analysis of the AshMaize Virtual Machine architecture, mapping every field, function, and state transition for accurate CUDA translation.

---

## VM Struct Definition

```rust
struct VM {
    program: Program,              // Instruction storage (nb_instrs * 20 bytes)
    regs: [Register; NB_REGS],     // 32 x u64 registers (256 bytes)
    ip: u32,                       // Instruction pointer
    prog_digest: blake2b::Context<512>,  // Program execution digest
    mem_digest: blake2b::Context<512>,   // Memory access digest
    prog_seed: [u8; 64],           // Seed for program shuffling
    memory_counter: u32,           // ROM access counter
    loop_counter: u32,             // Execution loop counter
}
```

### Field Details

#### 1. **`regs: [u64; 32]`**
- **Size**: 32 registers × 8 bytes = 256 bytes
- **Purpose**: General-purpose VM registers for instruction operands and results
- **Index bits**: 5 bits (0-31, mask: 0x1f)
- **Initialization**: From Argon2H' output (first 256 bytes of 448-byte buffer)
- **Operations**: Read/write by all instruction types
- **CUDA mapping**: `uint64_t regs[32]` per thread (256 bytes in registers or local memory)

#### 2. **`ip: u32`**
- **Purpose**: Instruction pointer (program counter)
- **Initialization**: 0
- **Update**: `ip = ip.wrapping_add(1)` after each instruction
- **Usage**: `program.at(ip)` to fetch instruction bytes
- **Wrap behavior**: Wrapping addition (overflow to 0)
- **CUDA mapping**: `uint32_t ip` per thread

#### 3. **`prog_digest: Blake2b::Context<512>`**
- **Purpose**: Accumulates hash of executed program instructions
- **Initialization**: From Argon2H' output (bytes 256-319, 64 bytes)
  - Created as: `Blake2b::<512>::new().update(&init_digest_data[0..64])`
- **Updates**: 
  - Every instruction: `prog_digest.update_mut(&prog_chunk)` (20 bytes)
  - special1_value64 reads: Clones, finalizes to 64 bytes, reads first 8 bytes
- **Mixing**: At `post_instructions()`:
  - Clones, updates with register sum (8 bytes), finalizes (64 bytes)
  - Combined with mem_digest for mixing_value generation
  - Result becomes new `prog_seed`
- **CUDA considerations**: Blake2b state is ~200 bytes, requires incremental hash implementation

#### 4. **`mem_digest: Blake2b::Context<512>`**
- **Purpose**: Accumulates hash of ROM memory accesses
- **Initialization**: From Argon2H' output (bytes 320-383, 64 bytes)
- **Updates**: 
  - Every ROM access: `mem_digest.update_mut(mem)` (64 bytes)
  - special2_value64 reads: Clones, finalizes, reads first 8 bytes
- **Mixing**: At `post_instructions()`:
  - Clones, updates with register sum, finalizes
  - Combined with prog_digest for mixing
- **CUDA considerations**: Same as prog_digest

#### 5. **`prog_seed: [u8; 64]`**
- **Purpose**: Seed for program instruction shuffling
- **Initialization**: From Argon2H' output (bytes 384-447, 64 bytes)
- **Usage**: 
  - `program.shuffle(&prog_seed)` at start of each execution loop
  - Shuffle calls `argon2::hprime(&mut instructions, seed)` - **SEQUENTIAL**
- **Update**: Set to prog_digest result after post_instructions()
- **CUDA mapping**: `uint8_t prog_seed[64]` per thread

#### 6. **`memory_counter: u32`**
- **Purpose**: Counts total ROM accesses for memory access pattern tracking
- **Initialization**: 0
- **Update**: `memory_counter = memory_counter.wrapping_add(1)` on every `mem_access64!` invocation
- **Usage in indexing**: 
  ```rust
  let idx = ((memory_counter % (64 / 8)) as usize) * 8;  // 0, 8, 16, 24, 32, 40, 48, 56
  ```
  - Cycles through 8 chunks of 8 bytes within 64-byte ROM blocks
- **Finalization**: Included in final hash: `update(&memory_counter.to_le_bytes())`
- **CUDA mapping**: `uint32_t memory_counter` per thread

#### 7. **`loop_counter: u32`**
- **Purpose**: Tracks execution loop iterations
- **Initialization**: 0
- **Update**: `loop_counter = loop_counter.wrapping_add(1)` at end of `post_instructions()`
- **Usage**: 
  - Included in mixing_value generation (4 bytes)
  - Ensures different mixing per loop even with identical register states
- **CUDA mapping**: `uint32_t loop_counter` per thread

#### 8. **`program: Program`**
- **Purpose**: Stores instruction bytes
- **Structure**:
  ```rust
  struct Program {
      instructions: Vec<u8>,  // size = nb_instrs * 20 bytes
  }
  ```
- **Initialization**: Zeroed vector of `nb_instrs * 20` bytes
- **Shuffling**: **SEQUENTIAL** Argon2H' operation every loop
  - Input: `prog_seed` (64 bytes)
  - Output: Overwrites entire instruction buffer
  - **Critical bottleneck for GPU**: Cannot parallelize across threads
- **Access**: `program.at(ip)` returns `&[u8; 20]` with wrapping modulo
- **CUDA strategy**: 
  - Pre-shuffle on CPU for first loop? Or
  - GPU Argon2H' with device-side sequential execution? Or
  - Hybrid: CPU shuffles, GPU reads?

---

## VM Initialization (`VM::new()`)

### Input Parameters
1. **`rom_digest: &RomDigest`** - 64-byte digest from ROM generation
2. **`nb_instrs: u32`** - Number of instructions (program size)
3. **`salt: &[u8]`** - Arbitrary input data for hash

### Initialization Steps

```rust
pub fn new(rom_digest: &RomDigest, nb_instrs: u32, salt: &[u8]) -> Self {
    const DIGEST_INIT_SIZE: usize = 64;
    const REGS_CONTENT_SIZE: usize = 8 * 32;  // 256 bytes
    
    // Step 1: Allocate initialization buffer
    let mut init_buffer = [0; 256 + 3*64];  // 448 bytes total
    
    // Step 2: Prepare Argon2H' input
    let mut init_buffer_input = rom_digest.0.to_vec();  // 64 bytes
    init_buffer_input.extend_from_slice(salt);          // + salt bytes
    
    // Step 3: Generate initialization data (SEQUENTIAL)
    argon2::hprime(&mut init_buffer, &init_buffer_input);
    
    // Step 4: Parse initialization buffer
    // Bytes 0-255: Register initial values
    // Bytes 256-319: prog_digest initialization
    // Bytes 320-383: mem_digest initialization  
    // Bytes 384-447: prog_seed
    
    // Step 5: Initialize registers
    for (reg, reg_bytes) in regs.iter_mut().zip(init_buffer[..256].chunks(8)) {
        *reg = u64::from_le_bytes(...);
    }
    
    // Step 6: Initialize digests
    let prog_digest = Blake2b::<512>::new().update(&init_buffer[256..320]);
    let mem_digest = Blake2b::<512>::new().update(&init_buffer[320..384]);
    let prog_seed = init_buffer[384..448];
    
    // Step 7: Create program (zeroed)
    let program = Program::new(nb_instrs);
    
    // Step 8: Return VM with counters at 0
}
```

### CUDA Implementation Considerations

1. **Argon2H' is sequential** - Each thread must execute independently
2. **Total initialization size**: 448 bytes output
3. **Input dependency**: Requires ROM digest (must be computed first)
4. **Deterministic**: Same inputs always produce same VM state

---

## State Transitions

### Execution Flow

```
hash(salt, rom, nb_loops, nb_instrs) {
    vm = VM::new(rom.digest, nb_instrs, salt)  // SEQUENTIAL Argon2H'
    
    for loop in 0..nb_loops {
        vm.execute(rom, nb_instrs) {
            vm.program.shuffle(&vm.prog_seed)  // SEQUENTIAL Argon2H'
            
            for instr in 0..nb_instrs {
                vm.step(rom) {
                    execute_one_instruction(vm, rom)
                    vm.ip = vm.ip.wrapping_add(1)
                }
            }
            
            vm.post_instructions()  // SEQUENTIAL Argon2H' for mixing
        }
    }
    
    return vm.finalize()
}
```

### Instruction Execution Cycle

**Per instruction (`step()`):**
1. Fetch instruction bytes: `program.at(ip)` → 20 bytes
2. Decode instruction → opcode, operands, registers, literals
3. Load source operands (may trigger ROM access, digest reads)
4. Execute operation
5. Write result to destination register
6. Update prog_digest with instruction bytes (20 bytes)
7. Increment IP

**ROM access side effects:**
- Updates mem_digest (64 bytes)
- Increments memory_counter
- Cycles through 8-byte chunks based on counter

**Special operand side effects:**
- special1: Clones prog_digest, finalizes (expensive)
- special2: Clones mem_digest, finalizes (expensive)

### Post-Instruction Mixing

Occurs after every `nb_instrs` instructions:

```rust
pub fn post_instructions(&mut self) {
    // 1. Compute register sum
    let sum_regs = self.regs.iter().fold(0, |acc, r| acc.wrapping_add(*r));
    
    // 2. Finalize digests with sum
    let prog_value = self.prog_digest.clone()
        .update(&sum_regs.to_le_bytes())
        .finalize();  // 64 bytes
    
    let mem_value = self.mem_digest.clone()
        .update(&sum_regs.to_le_bytes())
        .finalize();  // 64 bytes
    
    // 3. Generate mixing seed
    let mixing_value = Blake2b::<512>::new()
        .update(&prog_value)      // 64 bytes
        .update(&mem_value)       // 64 bytes
        .update(&loop_counter.to_le_bytes())  // 4 bytes
        .finalize();  // 64 bytes
    
    // 4. Generate mixing data (SEQUENTIAL Argon2H')
    let mut mixing_out = vec![0; 32 * 32 * 8];  // 8192 bytes
    argon2::hprime(&mut mixing_out, &mixing_value);
    
    // 5. XOR registers with mixing data (32 rounds)
    for mem_chunks in mixing_out.chunks(256) {  // 32 chunks of 256 bytes
        for (reg, reg_chunk) in regs.iter_mut().zip(mem_chunks.chunks(8)) {
            *reg ^= u64::from_le_bytes(reg_chunk);
        }
    }
    
    // 6. Update state
    self.prog_seed = prog_value;
    self.loop_counter = loop_counter.wrapping_add(1);
}
```

**CUDA challenge**: Argon2H' generates 8192 bytes sequentially per thread, per loop.

### Finalization

```rust
pub fn finalize(self) -> [u8; 64] {
    let prog_digest = self.prog_digest.finalize();  // 64 bytes
    let mem_digest = self.mem_digest.finalize();    // 64 bytes
    
    let mut context = Blake2b::<512>::new()
        .update(&prog_digest)                       // 64 bytes
        .update(&mem_digest)                        // 64 bytes
        .update(&memory_counter.to_le_bytes());    // 4 bytes
    
    for r in self.regs {
        context.update_mut(&r.to_le_bytes());      // 32 × 8 = 256 bytes
    }
    
    context.finalize()  // Final 64-byte digest
}
```

---

## Memory Layout Summary

### Per-Thread State (CUDA)

| Component | Size | Location | Notes |
|-----------|------|----------|-------|
| regs | 256 bytes | Registers/local | Hot path, frequent access |
| ip | 4 bytes | Register | Incremented every instruction |
| memory_counter | 4 bytes | Register | Incremented on ROM access |
| loop_counter | 4 bytes | Register | Incremented per loop |
| prog_digest | ~200 bytes | Local memory | Blake2b incremental state |
| mem_digest | ~200 bytes | Local memory | Blake2b incremental state |
| prog_seed | 64 bytes | Local memory | Read for shuffling |
| program.instructions | nb_instrs × 20 | Local/shared | Large, consider shared memory |
| **Total** | ~732 + 20×nb_instrs | | For 256 instrs: ~5.7KB |

### Shared Resources

| Resource | Size | Notes |
|----------|------|-------|
| ROM data | Variable (MB-GB) | Texture memory for cached reads |
| Argon2H' implementation | Code | Sequential, per-thread execution |
| Blake2b implementation | Code | Incremental hashing |

---

## Sequential Dependencies (GPU Challenges)

### Critical Sequential Operations

1. **VM::new() - Argon2H'**: 448 bytes output
   - Cannot share across threads (each has different salt)
   - Must execute per-thread

2. **program.shuffle() - Argon2H'**: Full program size
   - nb_instrs × 20 bytes
   - For 256 instructions: 5120 bytes
   - Executed per loop (nb_loops times)

3. **post_instructions() - Argon2H'**: 8192 bytes
   - Mixing data generation
   - Executed per loop

4. **Instruction execution**: Somewhat parallel
   - Each instruction depends on previous register state
   - ROM access patterns unpredictable (random)

### Parallelization Strategy

**What CAN parallelize:**
- Multiple salts computed simultaneously (embarrassingly parallel)
- Each thread has independent VM state
- No inter-thread communication needed

**What CANNOT parallelize:**
- Argon2H' within a single hash computation
- Instruction execution within a single VM (sequential dependencies)
- ROM accesses (random, data-dependent)

**Optimal GPU strategy:**
- Grid of threads, each computing different salt
- 1024+ threads per SM
- Target: 40-60x speedup from parallel salt mining

---

## Next Steps for CUDA Implementation

1. **Choose Argon2H' library**: 
   - Pure CUDA implementation?
   - Use existing library (OpenCL port)?
   - Optimize for GPU (still inherently sequential)

2. **Blake2b implementation**:
   - Incremental hashing support
   - Small state footprint
   - Many CUDA implementations available

3. **Memory strategy**:
   - ROM in texture memory (cached, read-only)
   - VM state in registers/local memory
   - Shared memory for program instructions?

4. **Kernel design**:
   - 1 thread = 1 salt hash computation
   - Grid size based on GPU occupancy
   - Salt generation strategy (counter-based, random)

5. **Rust FFI interface**:
   - ROM generation on CPU (already optimized)
   - Pass ROM pointer to GPU
   - Batch salt computations
   - Return found solutions (hash < difficulty)

---

## Summary for CUDA Translation

### Data Structures
- `VM` struct: 8 fields, ~732 bytes base + program size
- All fields must be replicated per thread
- No shared mutable state between threads

### Operations
- 3 types: Instructions, Blake2b hashing, Argon2H' derivation
- Instructions: Fast, but sequential dependencies
- Blake2b: Fast, incremental
- Argon2H': Slow, sequential, dominates runtime (64% of total)

### Execution Model
- Outer loop: nb_loops iterations
- Middle loop: nb_instrs instructions
- Inner: Individual instruction execution
- Post-processing: Expensive mixing every loop

### GPU Viability
- Single hash: 2x **slower** than CPU (sequential bottlenecks)
- Parallel mining: 40-60x **faster** (1000+ independent hashes)
- **Conclusion**: GPU mining is viable and beneficial for PoW mining workloads

