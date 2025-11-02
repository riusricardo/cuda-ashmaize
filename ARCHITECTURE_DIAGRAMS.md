# AshMaize Visual Architecture Diagrams

This document provides visual representations of AshMaize's architecture and data flows.

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ASHMAIZE SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    INITIALIZATION PHASE                        │    │
│  │                                                                 │    │
│  │  Input: key (bytes)                                            │    │
│  │    │                                                            │    │
│  │    ├─→ ROM Generator ──→ ROM (64 MB - 2 GB)                   │    │
│  │    │                           │                                │    │
│  │    │                           └─→ rom_digest (64 bytes)       │    │
│  │    │                                                            │    │
│  │    └─→ Configuration ──→ Parameters (loops, instrs)            │    │
│  │                                                                 │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                 ↓                                       │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                      HASHING PHASE                              │    │
│  │                                                                 │    │
│  │  Input: salt (bytes)                                            │    │
│  │    │                                                            │    │
│  │    ├─→ VM Initializer ──→ VM State                             │    │
│  │    │                      ├─ 32 registers                       │    │
│  │    │                      ├─ prog_digest context                │    │
│  │    │                      ├─ mem_digest context                 │    │
│  │    │                      └─ prog_seed                          │    │
│  │    │                                                            │    │
│  │    └─→ Execution Loop ──→ nb_loops iterations                  │    │
│  │         (see detailed diagram below)                            │    │
│  │                                 │                                │    │
│  │                                 ↓                                │    │
│  │                         Finalization ──→ digest (64 bytes)      │    │
│  │                                                                 │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. ROM Generation Detailed Flow

### Option A: FullRandom (1-Step)

```
┌──────────────────────────────────────────────────────────────────┐
│                      FullRandom ROM Generation                   │
└──────────────────────────────────────────────────────────────────┘

  Input: key, size
     │
     ├─→ seed = Blake2b-256(LE32(size) || key)
     │
     ├─→ Argon2H' Sequential Expansion
     │    │
     │    ├─ V[0] = Blake2b-512(LE32(size) || seed)
     │    │   ↓
     │    ├─ ROM[0..32] ← V[0][0..32]
     │    │
     │    ├─ V[1] = Blake2b-512(V[0])
     │    │   ↓
     │    ├─ ROM[32..64] ← V[1][0..32]
     │    │
     │    ├─ V[2] = Blake2b-512(V[1])
     │    │   ↓
     │    ├─ ROM[64..96] ← V[2][0..32]
     │    │
     │    └─ ... (continue until ROM filled)
     │
     ├─→ rom_digest = Blake2b-512(entire ROM)
     │
     └─→ Output: ROM (size bytes), rom_digest (64 bytes)

Properties:
✓ Fully sequential (no parallelization)
✓ Maximum ASIC resistance
✓ Each byte depends on all previous bytes
✗ Slower generation (linear in size)
```

### Option B: TwoStep (2-Step)

```
┌──────────────────────────────────────────────────────────────────┐
│                      TwoStep ROM Generation                      │
└──────────────────────────────────────────────────────────────────┘

  Input: key, size, pre_size, mixing_numbers
     │
     ├─→ STEP 1: Generate Pre-ROM
     │    │
     │    ├─ seed = Blake2b-256(LE32(size) || key)
     │    │
     │    ├─ pre_rom = Argon2H'(seed, pre_size)
     │    │   [Fully sequential, same as FullRandom]
     │    │
     │    └─ pre_rom (pre_size bytes)
     │
     ├─→ STEP 2: Generate Offsets
     │    │
     │    ├─ FOR i in 0..4:
     │    │    offset_diff[i] = Blake2b-512(seed || "offset" || i)
     │    │                      └─→ 32 u16 values
     │    │
     │    └─ offset_base = Argon2H'(Blake2b(seed || "base"), rom_chunks)
     │                      └─→ u8 array (one per final chunk)
     │
     ├─→ STEP 3: Assemble Final ROM
     │    │
     │    ├─ FOR each 64-byte chunk[i] in final ROM:
     │    │    │
     │    │    ├─ chunk[i] = pre_rom[i % pre_chunks]
     │    │    │
     │    │    ├─ FOR j in 1..mixing_numbers:
     │    │    │    idx = (offset_base[i] + offset_diff[j]) % pre_chunks
     │    │    │    chunk[i] ^= pre_rom[idx]
     │    │    │
     │    │    └─ rom_digest.update(chunk[i])
     │    │
     │    └─→ ROM (size bytes)
     │
     └─→ Output: ROM, rom_digest

Properties:
✓ Faster generation (pre_size << size)
✓ Still requires full ROM storage
✓ Random access pattern preserved
✗ Slightly weaker (pre_rom can be optimized)
```

---

## 3. VM Execution Loop (Single Iteration)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     VM EXECUTION LOOP (One Iteration)                    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  STEP 1: Program Generation                                    │     │
│  │                                                                 │     │
│  │  Input: prog_seed (64 bytes)                                   │     │
│  │     │                                                           │     │
│  │     ├─→ program_bytes = Argon2H'(prog_seed, nb_instrs × 20)   │     │
│  │     │                                                           │     │
│  │     └─→ Program (nb_instrs instructions, 20 bytes each)        │     │
│  │                                                                 │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                              ↓                                           │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  STEP 2: Instruction Execution (nb_instrs times)               │     │
│  │                                                                 │     │
│  │  FOR instr_idx in 0..nb_instrs:                                │     │
│  │                                                                 │     │
│  │     ┌──────────────────────────────────────────────────┐       │     │
│  │     │ 2a. Fetch & Decode                               │       │     │
│  │     │                                                  │       │     │
│  │     │  raw_bytes = program[ip]  (20 bytes)            │       │     │
│  │     │     │                                            │       │     │
│  │     │     ├─→ opcode: Instr                           │       │     │
│  │     │     ├─→ op1_type, op2_type: Operand             │       │     │
│  │     │     ├─→ r1, r2, r3: u8  (register indices)      │       │     │
│  │     │     └─→ lit1, lit2: u64 (literals/addresses)    │       │     │
│  │     │                                                  │       │     │
│  │     └──────────────────────────────────────────────────┘       │     │
│  │                              ↓                                 │     │
│  │     ┌──────────────────────────────────────────────────┐       │     │
│  │     │ 2b. Load Operands                                │       │     │
│  │     │                                                  │       │     │
│  │     │  src1 = CASE op1_type OF:                        │       │     │
│  │     │    Register → regs[r1]                           │       │     │
│  │     │    Memory   → ROM[lit1 % rom_chunks]             │       │     │
│  │     │               ├─→ mem_digest.update(cache_line)  │       │     │
│  │     │               ├─→ memory_counter += 1            │       │     │
│  │     │               └─→ extract 8 bytes based on MC    │       │     │
│  │     │    Literal  → lit1                               │       │     │
│  │     │    Special1 → prog_digest.finalize_copy()[0..8]  │       │     │
│  │     │    Special2 → mem_digest.finalize_copy()[0..8]   │       │     │
│  │     │                                                  │       │     │
│  │     │  src2 = CASE op2_type OF: [same logic]           │       │     │
│  │     │                                                  │       │     │
│  │     └──────────────────────────────────────────────────┘       │     │
│  │                              ↓                                 │     │
│  │     ┌──────────────────────────────────────────────────┐       │     │
│  │     │ 2c. Execute Operation                            │       │     │
│  │     │                                                  │       │     │
│  │     │  result = CASE opcode OF:                        │       │     │
│  │     │    Add     → src1 + src2                         │       │     │
│  │     │    Mul     → src1 * src2                         │       │     │
│  │     │    MulH    → (src1 * src2) >> 64                 │       │     │
│  │     │    Div     → src1 / src2  (or special1 if div0)  │       │     │
│  │     │    Xor     → src1 ^ src2                         │       │     │
│  │     │    Hash[N] → Blake2b(src1||src2)[N*8..(N+1)*8]   │       │     │
│  │     │    ISqrt   → floor(sqrt(src1))                   │       │     │
│  │     │    ...                                           │       │     │
│  │     │                                                  │       │     │
│  │     └──────────────────────────────────────────────────┘       │     │
│  │                              ↓                                 │     │
│  │     ┌──────────────────────────────────────────────────┐       │     │
│  │     │ 2d. Store Result & Update State                  │       │     │
│  │     │                                                  │       │     │
│  │     │  regs[r3] = result                               │       │     │
│  │     │  prog_digest.update(raw_bytes)  # 20 bytes       │       │     │
│  │     │  ip += 1                                         │       │     │
│  │     │                                                  │       │     │
│  │     └──────────────────────────────────────────────────┘       │     │
│  │                                                                 │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                              ↓                                           │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  STEP 3: Post-Instruction Mixing                               │     │
│  │                                                                 │     │
│  │  ┌──────────────────────────────────────────────────────┐      │     │
│  │  │ 3a. Compute Register Sum                             │      │     │
│  │  │                                                      │      │     │
│  │  │  sum_regs = Σ regs[i]  (wrapping add)                │      │     │
│  │  │                                                      │      │     │
│  │  └──────────────────────────────────────────────────────┘      │     │
│  │                              ↓                                 │     │
│  │  ┌──────────────────────────────────────────────────────┐      │     │
│  │  │ 3b. Finalize Digests (Temporary)                     │      │     │
│  │  │                                                      │      │     │
│  │  │  prog_value = prog_digest.clone()                    │      │     │
│  │  │                .update(sum_regs.to_le())             │      │     │
│  │  │                .finalize()  # 64 bytes               │      │     │
│  │  │                                                      │      │     │
│  │  │  mem_value = mem_digest.clone()                      │      │     │
│  │  │               .update(sum_regs.to_le())              │      │     │
│  │  │               .finalize()  # 64 bytes                │      │     │
│  │  │                                                      │      │     │
│  │  └──────────────────────────────────────────────────────┘      │     │
│  │                              ↓                                 │     │
│  │  ┌──────────────────────────────────────────────────────┐      │     │
│  │  │ 3c. Generate Mixing Material                         │      │     │
│  │  │                                                      │      │     │
│  │  │  mixing_seed = Blake2b(prog_value || mem_value       │      │     │
│  │  │                        || loop_counter.to_le())      │      │     │
│  │  │                                                      │      │     │
│  │  │  mixing_data = Argon2H'(mixing_seed, 8192 bytes)     │      │     │
│  │  │                # 32 rounds × 32 regs × 8 bytes       │      │     │
│  │  │                                                      │      │     │
│  │  └──────────────────────────────────────────────────────┘      │     │
│  │                              ↓                                 │     │
│  │  ┌──────────────────────────────────────────────────────┐      │     │
│  │  │ 3d. XOR Mixing Into Registers (32 rounds)            │      │     │
│  │  │                                                      │      │     │
│  │  │  offset = 0                                          │      │     │
│  │  │  FOR round in 0..32:                                 │      │     │
│  │  │    FOR reg_idx in 0..32:                             │      │     │
│  │  │      regs[reg_idx] ^= u64_le(mixing_data[offset])    │      │     │
│  │  │      offset += 8                                     │      │     │
│  │  │                                                      │      │     │
│  │  └──────────────────────────────────────────────────────┘      │     │
│  │                              ↓                                 │     │
│  │  ┌──────────────────────────────────────────────────────┐      │     │
│  │  │ 3e. Update State for Next Loop                       │      │     │
│  │  │                                                      │      │     │
│  │  │  prog_seed = prog_value  # Becomes next program seed │      │     │
│  │  │  loop_counter += 1                                   │      │     │
│  │  │                                                      │      │     │
│  │  └──────────────────────────────────────────────────────┘      │     │
│  │                                                                 │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  [Repeat entire loop nb_loops times]                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Finalization Phase

```
┌────────────────────────────────────────────────────────────────┐
│                      FINALIZATION                              │
└────────────────────────────────────────────────────────────────┘

  After nb_loops iterations complete:
     │
     ├─→ prog_final = prog_digest.finalize()     # 64 bytes
     │
     ├─→ mem_final = mem_digest.finalize()       # 64 bytes
     │
     ├─→ Assemble final input:
     │    │
     │    ├─ prog_final                # 64 bytes
     │    ├─ mem_final                 # 64 bytes
     │    ├─ memory_counter.to_le()    # 4 bytes
     │    ├─ regs[0].to_le()           # 8 bytes
     │    ├─ regs[1].to_le()           # 8 bytes
     │    ├─ ...
     │    └─ regs[31].to_le()          # 8 bytes
     │
     │    Total: 64 + 64 + 4 + (32 × 8) = 388 bytes
     │
     ├─→ final_digest = Blake2b-512(assembled_input)
     │
     └─→ Output: 64-byte digest

This digest is the PoW hash output.
```

---

## 5. Data Flow Summary

```
┌──────────┐
│   Key    │  (Application-specific, rotates periodically)
└────┬─────┘
     │
     ├──────────────┐
     │              ↓
     │      ┌───────────────┐
     │      │ ROM Generator │ (One-time per key)
     │      └───────┬───────┘
     │              ↓
     │         ┌────────┐
     │         │  ROM   │  (Large, 64MB-2GB, stored in memory)
     │         └────┬───┘
     │              │
┌────┴────┐         │
│  Salt   │  ←──────┴────┐ (Variable input, changes each hash attempt)
└────┬────┘              │
     │                   │
     ├───────────────┐   │
     │               ↓   ↓
     │         ┌──────────────┐
     │         │ VM Initialize│
     │         └──────┬───────┘
     │                ↓
     │         ┌─────────────┐
     │         │   VM State  │
     │         │  32 regs    │
     │         │  2 digests  │
     │         │  prog_seed  │
     │         └──────┬──────┘
     │                │
     │                ├──→ Loop 0 ──┐
     │                │              ↓
     │                ├──→ Loop 1 ──┤
     │                │              ├─→ Execute instructions
     │                ├──→ Loop 2 ──┤    Access ROM
     │                │              ├─→ Update digests
     │                ├──→ ...   ───┤    Mix state
     │                │              ↓
     │                └──→ Loop N ───┘
     │                       │
     │                       ↓
     │                 ┌────────────┐
     │                 │ Finalize   │
     │                 └─────┬──────┘
     │                       ↓
     └──────────────→  ┌──────────┐
                       │  Digest  │  (64-byte hash output)
                       └──────────┘
```

---

## 6. Memory Layout

### Native Rust
```
Stack (per thread, ~1 MB):
├─ Function call frames
├─ Small local variables
└─ Return addresses

Heap (dynamic):
├─ ROM                        [size bytes, e.g., 256 MB]
│  └─ Cache-aligned allocation
├─ VM Program                 [nb_instrs × 20 bytes, e.g., 5 KB]
├─ VM State                   [~512 bytes]
│  ├─ 32 registers            [32 × 8 = 256 bytes]
│  ├─ Digest contexts         [2 × 128 = 256 bytes]
│  └─ Counters                [~64 bytes]
├─ Mixing buffers             [8192 bytes per loop]
└─ Temporary allocations      [minimal]

Total: ~ROM_size + 100 KB
```

### WebAssembly
```
WASM Linear Memory (single contiguous block):
┌─────────────────────────────────────────────┐
│ 0x00000000 - 0x000FFFFF : Stack (1 MB)      │
├─────────────────────────────────────────────┤
│ 0x00100000 - 0x001FFFFF : Heap Metadata     │
├─────────────────────────────────────────────┤
│ 0x00200000 - 0x0FFFFFFF : ROM (variable)    │
│   Example: 256 MB = 0x10000000 bytes        │
├─────────────────────────────────────────────┤
│ 0x10000000 - 0x10100000 : Working Memory    │
│   ├─ VM State                               │
│   ├─ Program buffer                         │
│   └─ Mixing buffers                         │
└─────────────────────────────────────────────┘

Growth: Pages allocated in 64 KB chunks
Maximum: 4 GB (WASM32 limit)
```

---

## 7. Instruction Encoding Visualization

```
Instruction Format (20 bytes):
┌────────┬─────┬─────┬───────┬────────┬────────┬─────────────────┬─────────────────┐
│ Byte 0 │Byte1│Byte2│ Byte3 │Bytes 4-11       │Bytes 12-19      │
├────────┼─────┴─────┴───────┼─────────────────┼─────────────────┤
│ OpCode │ Operands & Regs   │      Lit1       │      Lit2       │
└────────┴───────────────────┴─────────────────┴─────────────────┘

Byte 0 (OpCode):
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│  Bit7 │  Bit6 │  Bit5 │  Bit4 │  Bit3 │  Bit2 │  Bit1 │  Bit0 │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
  └──────────────────────────────┬──────────────────────────────┘
                    Instruction Type (0-255)

Byte 1 (Operand Types):
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│  Bit7 │  Bit6 │  Bit5 │  Bit4 │  Bit3 │  Bit2 │  Bit1 │  Bit0 │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
  └──────────┬──────────┘ └──────────┬──────────┘
      Op1 Type (4 bits)         Op2 Type (4 bits)

Bytes 2-3 (Register Indices):
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│15 │14 │13 │12 │11 │10 │ 9 │ 8 │ 7 │ 6 │ 5 │ 4 │ 3 │ 2 │ 1 │ 0 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
  └────┬────┘ └────┬────┘ └────┬────┘ └─┘
       R1          R2          R3      Unused
    (5 bits)    (5 bits)    (5 bits)  (1 bit)

Bytes 4-11 (Literal 1):
┌─────────────────────────────────────────────────────────────────┐
│                  64-bit value (little-endian)                   │
└─────────────────────────────────────────────────────────────────┘
  Used as: Immediate value OR Memory address

Bytes 12-19 (Literal 2):
┌─────────────────────────────────────────────────────────────────┐
│                  64-bit value (little-endian)                   │
└─────────────────────────────────────────────────────────────────┘
  Used as: Immediate value OR Memory address

Example Instruction:
  OpCode=42 (Mul), Op1=0 (Reg), Op2=5 (Mem), R1=3, R2=7, R3=15
  Lit1=0x1234567890ABCDEF, Lit2=0xFEDCBA0987654321

Execution:
  src1 = regs[3]                      # Register access
  src2 = ROM[0xFEDCBA0987654321]      # Memory access (+ digest update)
  result = src1 * src2
  regs[15] = result
```

---

## 8. Performance Breakdown Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│          Hash Time Distribution (typical, WASM)                 │
│                                                                 │
│  Total: ~100 ms per hash (256 MB ROM, 8 loops, 256 instrs)     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  VM Init (2%)        ██                                         │
│  ────────────────────────────────────────────────────────────── │
│                                                                 │
│  Program Gen (40%)   ████████████████████████████████████████   │
│                      Argon2H' × 8 (5 ms each)                  │
│  ────────────────────────────────────────────────────────────── │
│                                                                 │
│  Instruction         ██████████████████████████████             │
│  Execution (30%)     Decode + Compute + Store                  │
│  ────────────────────────────────────────────────────────────── │
│                                                                 │
│  Post-Mix (25%)      █████████████████████████                 │
│                      Argon2H' × 8 (3 ms each)                  │
│  ────────────────────────────────────────────────────────────── │
│                                                                 │
│  Finalization (3%)   ███                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Key Bottleneck: Argon2H' calls (64% of total time)
  - Program shuffle: 40%
  - Post-mixing: 24%

Optimization Target: SIMD Blake2b in Argon2H' → ~30-50% speedup potential
```

---

## 9. WASM vs Native Performance Gap

```
┌───────────────────────────────────────────────────────────────┐
│               Operation Performance Comparison                 │
│                (Native = 1.0x baseline)                        │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  64-bit Arithmetic (Add/Mul/Xor):                             │
│    Native: ████████████████████ 1.0x                          │
│    WASM:   ████████████████████ 1.0x (i64 native support)    │
│                                                               │
│  128-bit MulH:                                                │
│    Native: ████████████████████ 1.0x                          │
│    WASM:   ██████████████████████████████████████ 2.0x       │
│            (software emulation)                               │
│                                                               │
│  Blake2b-512:                                                 │
│    Native: ████████████████████ 1.0x                          │
│    WASM:   ██████████████████████████████ 1.5x               │
│            (no SIMD yet)                                      │
│                                                               │
│  Argon2H':                                                    │
│    Native: ████████████████████ 1.0x                          │
│    WASM:   ██████████████████████████████ 1.5x               │
│            (Blake2b overhead)                                 │
│                                                               │
│  ROM Access:                                                  │
│    Native: ████████████████████ 1.0x                          │
│    WASM:   ████████████████████████ 1.2x                     │
│            (bounds checking)                                  │
│                                                               │
│  ─────────────────────────────────────────────────────────    │
│  Overall:                                                     │
│    Native: ████████████████████ 1.0x (~1000 H/s)             │
│    WASM:   ██████████████████████████████ 1.5x (~600 H/s)    │
│                                                               │
└───────────────────────────────────────────────────────────────┘

Future with WASM SIMD:
  Blake2b: 1.5x → 1.1x (30% improvement)
  Overall: 1.5x → 1.2x (25% improvement → ~800 H/s)
```

---

## 10. Dependency Graph

```
┌──────────────────────────────────────────────────────────────┐
│                   DEPENDENCY HIERARCHY                       │
└──────────────────────────────────────────────────────────────┘

  ashmaize (root crate)
    │
    ├── cryptoxide (external)
    │   ├── hashing::blake2b
    │   │   └── Blake2b<512>
    │   └── kdf::argon2
    │       └── hprime()
    │
    ├── src/lib.rs
    │   ├── VM struct
    │   ├── Program struct
    │   ├── Instruction enum
    │   ├── execute_one_instruction()
    │   ├── hash() [main entry point]
    │   └── tests
    │
    └── src/rom.rs
        ├── Rom struct
        ├── RomDigest struct
        ├── RomGenerationType enum
        ├── random_gen()
        └── tests

  ashmaize-web (WASM bindings)
    │
    ├── ashmaize (internal dependency)
    │
    ├── wasm-bindgen (external)
    │   └── FFI generation
    │
    ├── console_error_panic_hook (optional)
    │
    ├── wee_alloc (optional)
    │
    ├── src/lib.rs
    │   ├── Rom wrapper
    │   ├── RomBuilder
    │   └── RomBuilderError
    │
    ├── src/utils.rs
    │   └── set_panic_hook()
    │
    └── tests/wasm.rs
        └── Browser-based tests

  ashmaize-webdemo (Demo application)
    │
    ├── ashmaize (internal dependency)
    │
    ├── leptos (external)
    │   └── CSR framework
    │
    └── src/
        ├── lib.rs (App component)
        ├── main.rs (entry point)
        └── components/
            └── ashmaize.rs (mining UI)
```

---

**Document Version**: 1.0  
**Created**: October 27, 2025  
**Purpose**: Visual reference for AshMaize architecture
