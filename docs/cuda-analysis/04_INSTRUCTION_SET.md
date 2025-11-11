# Instruction Set Deep Dive for CUDA Implementation

## Overview
AshMaize VM uses a custom 13-operation instruction set with 5 operand types. Instructions are 20 bytes each, encoding operation, operand types, register indices, and 128-bit literals. This document provides complete implementation details for CUDA translation.

---

## Instruction Format (20 bytes)

```
Byte Layout:
[0]       : Opcode (u8, 0-255) → determines operation
[1]       : Operand types (u8)
              - High nibble (bits 4-7): op1 type
              - Low nibble (bits 0-3): op2 type
[2-3]     : Register indices (u16)
              - Bits 15-10: r1 (6 bits → masked to 5 bits)
              - Bits 9-5:   r2 (5 bits)
              - Bits 4-0:   r3 (5 bits)
[4-11]    : lit1 (u64, little-endian)
[12-19]   : lit2 (u64, little-endian)

Total: 20 bytes per instruction
```

### Decoding Example

```rust
fn decode_instruction(instruction: &[u8; 20]) -> Instruction {
    let opcode = Instr::from(instruction[0]);
    let op1 = Operand::from(instruction[1] >> 4);
    let op2 = Operand::from(instruction[1] & 0x0f);
    
    let rs = ((instruction[2] as u16) << 8) | (instruction[3] as u16);
    let r1 = ((rs >> 10) as u8) & 0x1f;  // Bits 15-10, mask to 5 bits
    let r2 = ((rs >> 5) as u8) & 0x1f;   // Bits 9-5
    let r3 = (rs as u8) & 0x1f;          // Bits 4-0
    
    let lit1 = u64::from_le_bytes(instruction[4..12]);
    let lit2 = u64::from_le_bytes(instruction[12..20]);
    
    Instruction { opcode, op1, op2, r1, r2, r3, lit1, lit2 }
}
```

**CUDA implementation:**
```cuda
struct Instruction {
    uint8_t opcode;
    uint8_t op1_type;
    uint8_t op2_type;
    uint8_t r1, r2, r3;
    uint64_t lit1, lit2;
};

__device__ Instruction decode_instruction(const uint8_t* instr) {
    Instruction i;
    i.opcode = instr[0];
    i.op1_type = instr[1] >> 4;
    i.op2_type = instr[1] & 0x0f;
    
    uint16_t rs = (instr[2] << 8) | instr[3];
    i.r1 = ((rs >> 10) & 0x1f);
    i.r2 = ((rs >> 5) & 0x1f);
    i.r3 = (rs & 0x1f);
    
    // Little-endian u64 reads
    i.lit1 = *((uint64_t*)&instr[4]);
    i.lit2 = *((uint64_t*)&instr[12]);
    
    return i;
}
```

---

## Opcode Mapping

### Distribution

| Opcode Range | Operation | Count | Percentage | Type |
|--------------|-----------|-------|------------|------|
| 0-39 | Add | 40 | 15.6% | Op3 |
| 40-79 | Mul | 40 | 15.6% | Op3 |
| 80-95 | MulH | 16 | 6.3% | Op3 |
| 96-111 | Div | 16 | 6.3% | Op3 |
| 112-127 | Mod | 16 | 6.3% | Op3 |
| 128-137 | ISqrt | 10 | 3.9% | Op2 |
| 138-147 | BitRev | 10 | 3.9% | Op2 |
| 148-187 | Xor | 40 | 15.6% | Op3 |
| 188-203 | RotL | 16 | 6.3% | Op2 |
| 204-219 | RotR | 16 | 6.3% | Op2 |
| 220-239 | Neg | 20 | 7.8% | Op2 |
| 240-247 | And | 8 | 3.1% | Op3 |
| 248-255 | Hash(v) | 8 | 3.1% | Op3 |

**Probability distribution:**
- **High frequency**: Add, Mul, Xor (15.6% each)
- **Medium frequency**: MulH, Div, Mod, RotL, RotR, Neg (6-8% each)
- **Low frequency**: ISqrt, BitRev, And, Hash (3-4% each)

**Decoding:**
```rust
impl From<u8> for Instr {
    fn from(value: u8) -> Self {
        match value {
            0..40 => Instr::Op3(Op3::Add),
            40..80 => Instr::Op3(Op3::Mul),
            80..96 => Instr::Op3(Op3::MulH),
            96..112 => Instr::Op3(Op3::Div),
            112..128 => Instr::Op3(Op3::Mod),
            128..138 => Instr::Op2(Op2::ISqrt),
            138..148 => Instr::Op2(Op2::BitRev),
            148..188 => Instr::Op3(Op3::Xor),
            188..204 => Instr::Op2(Op2::RotL),
            204..220 => Instr::Op2(Op2::RotR),
            220..240 => Instr::Op2(Op2::Neg),
            240..248 => Instr::Op3(Op3::And),
            248..=255 => Instr::Op3(Op3::Hash(value - 248)),
        }
    }
}
```

**CUDA implementation:**
```cuda
enum OpType { OP3, OP2 };

__device__ OpType get_op_type(uint8_t opcode) {
    if (opcode >= 128 && opcode < 138) return OP2;  // ISqrt
    if (opcode >= 138 && opcode < 148) return OP2;  // BitRev
    if (opcode >= 188 && opcode < 204) return OP2;  // RotL
    if (opcode >= 204 && opcode < 220) return OP2;  // RotR
    if (opcode >= 220 && opcode < 240) return OP2;  // Neg
    return OP3;
}

__device__ int get_operation(uint8_t opcode) {
    if (opcode < 40) return OP_ADD;
    if (opcode < 80) return OP_MUL;
    if (opcode < 96) return OP_MULH;
    if (opcode < 112) return OP_DIV;
    if (opcode < 128) return OP_MOD;
    if (opcode < 138) return OP_ISQRT;
    if (opcode < 148) return OP_BITREV;
    if (opcode < 188) return OP_XOR;
    if (opcode < 204) return OP_ROTL;
    if (opcode < 220) return OP_ROTR;
    if (opcode < 240) return OP_NEG;
    if (opcode < 248) return OP_AND;
    return OP_HASH;  // 248-255
}
```

---

## Operand Types (5 variants)

### Encoding

```rust
impl From<u8> for Operand {
    fn from(value: u8) -> Self {
        assert!(value <= 0x0f);  // 4-bit encoding
        match value {
            0..5 => Self::Reg,
            5..9 => Self::Memory,
            9..13 => Self::Literal,
            13..14 => Self::Special1,
            14.. => Self::Special2,
        }
    }
}
```

**Probability distribution:**
- **Reg**: 5/16 = 31.25%
- **Memory**: 4/16 = 25%
- **Literal**: 4/16 = 25%
- **Special1**: 1/16 = 6.25%
- **Special2**: 2/16 = 12.5%

### 1. Register (Reg)

**Encoding**: 0-4 (5 values)
**Usage**: Read from VM register array

```rust
Operand::Reg => vm.regs[r1 as usize]  // For op1
Operand::Reg => vm.regs[r2 as usize]  // For op2
```

**Characteristics:**
- Fastest operand type (register file access)
- Register index determined by r1/r2 field (5-bit, 0-31)
- No side effects

**CUDA:**
```cuda
case OPERAND_REG:
    src1 = vm->regs[instr.r1];
    break;
```

### 2. Memory

**Encoding**: 5-8 (4 values)
**Usage**: ROM access with side effects

```rust
Operand::Memory => {
    let mem = rom.at(lit1 as u32);  // Get 64-byte chunk
    vm.mem_digest.update_mut(mem);  // Update digest
    vm.memory_counter = vm.memory_counter.wrapping_add(1);
    
    // Extract 8 bytes based on counter
    let idx = ((vm.memory_counter % 8) as usize) * 8;
    u64::from_le_bytes(mem[idx..idx + 8])
}
```

**Characteristics:**
- **Expensive**: ROM access + digest update
- Uses lit1 (op1) or lit2 (op2) as address
- Cycles through 8-byte chunks within 64-byte blocks
- Side effects: Increments counter, updates mem_digest

**CUDA:**
```cuda
case OPERAND_MEMORY: {
    uint32_t addr = (uint32_t)instr.lit1;
    uint64_t chunk[8];  // 64 bytes
    
    // Fetch from texture memory
    fetch_rom_chunk(rom_texture, addr, chunk);
    
    // Update digest
    blake2b_update(&vm->mem_digest, (uint8_t*)chunk, 64);
    vm->memory_counter++;
    
    // Extract 8 bytes
    int idx = (vm->memory_counter % 8);
    src1 = chunk[idx];
    break;
}
```

**ROM access details** (see Section 5 for full analysis):
- Address: `addr % (rom_size / 64)` → chunk index
- Alignment: Always 64-byte aligned
- Access pattern: Pseudo-random based on lit1/lit2

### 3. Literal

**Encoding**: 9-12 (4 values)
**Usage**: Use literal value from instruction

```rust
Operand::Literal => lit1  // For op1
Operand::Literal => lit2  // For op2
```

**Characteristics:**
- Direct value from instruction bytes
- No memory access, no side effects
- Fast (immediate value)

**CUDA:**
```cuda
case OPERAND_LITERAL:
    src1 = instr.lit1;
    break;
```

### 4. Special1

**Encoding**: 13 (1 value)
**Usage**: Current prog_digest value

```rust
Operand::Special1 => {
    let r = vm.prog_digest.clone().finalize();
    u64::from_le_bytes(r[0..8])
}
```

**Characteristics:**
- **Very expensive**: Clone context + finalize Blake2b
- Reads first 8 bytes of 64-byte digest
- Non-deterministic based on execution history
- No side effects (clones, doesn't modify original)

**CUDA:**
```cuda
case OPERAND_SPECIAL1: {
    Blake2bState temp = blake2b_clone(&vm->prog_digest);
    uint8_t digest[64];
    blake2b_final(&temp, digest);
    src1 = *((uint64_t*)digest);  // First 8 bytes
    break;
}
```

**Cost analysis:**
- Clone: ~50 cycles
- Finalize: ~500-1000 cycles
- Total: ~1000+ cycles vs ~10 for register

### 5. Special2

**Encoding**: 14-15 (2 values)
**Usage**: Current mem_digest value

```rust
Operand::Special2 => {
    let r = vm.mem_digest.clone().finalize();
    u64::from_le_bytes(r[0..8])
}
```

**Characteristics:**
- Same as Special1, but uses mem_digest
- Reflects memory access history

**CUDA:**
```cuda
case OPERAND_SPECIAL2: {
    Blake2bState temp = blake2b_clone(&vm->mem_digest);
    uint8_t digest[64];
    blake2b_final(&temp, digest);
    src1 = *((uint64_t*)digest);
    break;
}
```

---

## Operations

### Three-Operand Operations (Op3)

Format: `dst = src1 OP src2`
- Load src1 based on op1 type
- Load src2 based on op2 type
- Compute result
- Store to `regs[r3]`

---

#### 1. Add (Opcodes 0-39)

```rust
Op3::Add => src1.wrapping_add(src2)
```

**Behavior**: Wrapping addition (overflow wraps to 0)
**Example**: `0xFFFFFFFFFFFFFFFF + 1 = 0`

**CUDA:**
```cuda
case OP_ADD:
    result = src1 + src2;  // Native wrapping on GPU
    break;
```

**Properties:**
- Fast (1 cycle)
- No branches
- No special cases

---

#### 2. Mul (Opcodes 40-79)

```rust
Op3::Mul => src1.wrapping_mul(src2)
```

**Behavior**: Wrapping multiplication (64-bit × 64-bit → low 64 bits)
**Example**: `0x100000000 * 0x100000000 = 0` (overflow discarded)

**CUDA:**
```cuda
case OP_MUL:
    result = src1 * src2;
    break;
```

**Properties:**
- Moderate cost (4-8 cycles on GPU)
- No branches
- Discards high 64 bits

---

#### 3. MulH (Opcodes 80-95)

```rust
Op3::MulH => ((src1 as u128 * src2 as u128) >> 64) as u64
```

**Behavior**: High 64 bits of 128-bit multiplication
**Example**: `0xFFFFFFFFFFFFFFFF * 0xFFFFFFFFFFFFFFFF >> 64 = 0xFFFFFFFFFFFFFFFE`

**CUDA:**
```cuda
case OP_MULH: {
    // PTX mul.hi.u64 instruction
    uint64_t hi, lo;
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(src1), "l"(src2));
    result = hi;
    break;
}
```

**Alternative (portable):**
```cuda
__device__ uint64_t mulhi64(uint64_t a, uint64_t b) {
    uint32_t a_lo = a, a_hi = a >> 32;
    uint32_t b_lo = b, b_hi = b >> 32;
    
    uint64_t a_lo_b_lo = (uint64_t)a_lo * b_lo;
    uint64_t a_hi_b_lo = (uint64_t)a_hi * b_lo;
    uint64_t a_lo_b_hi = (uint64_t)a_lo * b_hi;
    uint64_t a_hi_b_hi = (uint64_t)a_hi * b_hi;
    
    uint64_t carry = ((a_lo_b_lo >> 32) + (a_hi_b_lo & 0xFFFFFFFF) + (a_lo_b_hi & 0xFFFFFFFF)) >> 32;
    return a_hi_b_hi + (a_hi_b_lo >> 32) + (a_lo_b_hi >> 32) + carry;
}
```

**Properties:**
- Moderate cost (8-12 cycles)
- No branches
- Useful for large integer operations

---

#### 4. Div (Opcodes 96-111)

```rust
Op3::Div => {
    if src2 == 0 {
        special1_value64!(vm)
    } else {
        src1 / src2
    }
}
```

**Behavior**: Integer division with div-by-zero fallback
**Special case**: Division by zero returns Special1 value

**CUDA:**
```cuda
case OP_DIV:
    if (src2 == 0) {
        Blake2bState temp = blake2b_clone(&vm->prog_digest);
        uint8_t digest[64];
        blake2b_final(&temp, digest);
        result = *((uint64_t*)digest);
    } else {
        result = src1 / src2;
    }
    break;
```

**Properties:**
- **Branch divergence risk**: Different threads may take different paths
- Division: ~20-30 cycles on GPU
- Special1: ~1000+ cycles
- Probability of div-by-zero: ~1/2^64 (extremely rare in practice)

**Optimization**: Profile if div-by-zero path is ever taken. If not, can simplify.

---

#### 5. Mod (Opcodes 112-127)

```rust
Op3::Mod => {
    if src2 == 0 {
        special1_value64!(vm)
    } else {
        src1 / src2  // BUG: Should be src1 % src2
    }
}
```

**⚠️ KNOWN BUG**: Implementation uses division instead of modulo!

**Expected behavior**: `src1 % src2` (remainder)
**Actual behavior**: `src1 / src2` (quotient)

**CUDA implementation (bug-compatible):**
```cuda
case OP_MOD:
    if (src2 == 0) {
        // Special1 fallback
        Blake2bState temp = blake2b_clone(&vm->prog_digest);
        uint8_t digest[64];
        blake2b_final(&temp, digest);
        result = *((uint64_t*)digest);
    } else {
        result = src1 / src2;  // BUG: Should be src1 % src2
    }
    break;
```

**For corrected version:**
```cuda
result = src1 % src2;
```

**Properties:**
- Same as Div (including branch divergence)
- Modulo operation: ~20-30 cycles
- **CRITICAL**: Maintain bug compatibility for consensus

---

#### 6. Xor (Opcodes 148-187)

```rust
Op3::Xor => src1 ^ src2
```

**Behavior**: Bitwise XOR

**CUDA:**
```cuda
case OP_XOR:
    result = src1 ^ src2;
    break;
```

**Properties:**
- Fast (1 cycle)
- No branches
- Common operation (15.6% frequency)

---

#### 7. And (Opcodes 240-247)

```rust
Op3::And => src1 & src2
```

**Behavior**: Bitwise AND

**CUDA:**
```cuda
case OP_AND:
    result = src1 & src2;
    break;
```

**Properties:**
- Fast (1 cycle)
- No branches
- Low frequency (3.1%)

---

#### 8. Hash(v) (Opcodes 248-255)

```rust
Op3::Hash(v) => {
    assert!(v < 8);  // v = opcode - 248 (0-7)
    let out = Blake2b::<512>::new()
        .update(&src1.to_le_bytes())
        .update(&src2.to_le_bytes())
        .finalize();
    u64::from_le_bytes(out.chunks(8).nth(v).unwrap())
}
```

**Behavior**: Hash two operands, extract 8-byte chunk

**Variants:**
- Hash(0): Extract bytes 0-7
- Hash(1): Extract bytes 8-15
- Hash(2): Extract bytes 16-23
- Hash(3): Extract bytes 24-31
- Hash(4): Extract bytes 32-39
- Hash(5): Extract bytes 40-47
- Hash(6): Extract bytes 48-55
- Hash(7): Extract bytes 56-63

**CUDA:**
```cuda
case OP_HASH: {
    uint8_t hash_variant = opcode - 248;  // 0-7
    uint8_t digest[64];
    
    // One-shot Blake2b
    Blake2bState state;
    blake2b_init(&state, NULL, 0);
    blake2b_update(&state, (uint8_t*)&src1, 8);
    blake2b_update(&state, (uint8_t*)&src2, 8);
    blake2b_final(&state, digest);
    
    // Extract 8-byte chunk
    result = *((uint64_t*)&digest[hash_variant * 8]);
    break;
}
```

**Properties:**
- **Expensive**: Full Blake2b hash (~1000-2000 cycles)
- No branches (variant determined by opcode)
- Low frequency (~3%)
- Deterministic (same inputs → same output)

---

### Two-Operand Operations (Op2)

Format: `dst = OP src1`
- Load src1 based on op1 type
- Compute result (src2/op2 ignored for most)
- Store to `regs[r3]`

**Note**: Some Op2 operations use r1 as a parameter, not just for register indexing.

---

#### 9. Neg (Opcodes 220-239)

```rust
Op2::Neg => !src1
```

**Behavior**: Bitwise NOT (one's complement)
**Example**: `!0x00FF = 0xFFFFFFFFFFFFFF00`

**CUDA:**
```cuda
case OP_NEG:
    result = ~src1;
    break;
```

**Properties:**
- Fast (1 cycle)
- No branches
- Moderate frequency (7.8%)

---

#### 10. RotL (Opcodes 188-203)

```rust
Op2::RotL => src1.rotate_left(r1 as u32)
```

**Behavior**: Rotate left by r1 bits
**Example**: `0x0000000000000001.rotate_left(63) = 0x8000000000000000`

**CUDA:**
```cuda
case OP_ROTL: {
    uint32_t shift = instr.r1 & 0x3f;  // Mask to 0-63
    result = (src1 << shift) | (src1 >> (64 - shift));
    break;
}
```

**Properties:**
- Fast (2-3 cycles)
- No branches
- r1 as shift amount (0-31, but should wrap to 0-63)

---

#### 11. RotR (Opcodes 204-219)

```rust
Op2::RotR => src1.rotate_right(r1 as u32)
```

**Behavior**: Rotate right by r1 bits

**CUDA:**
```cuda
case OP_ROTR: {
    uint32_t shift = instr.r1 & 0x3f;
    result = (src1 >> shift) | (src1 << (64 - shift));
    break;
}
```

**Properties:**
- Fast (2-3 cycles)
- No branches
- r1 as shift amount

---

#### 12. ISqrt (Opcodes 128-137)

```rust
Op2::ISqrt => src1.isqrt()
```

**Behavior**: Integer square root (floor(√src1))
**Example**: `isqrt(15) = 3`, `isqrt(16) = 4`

**CUDA (software implementation):**
```cuda
__device__ uint64_t isqrt64(uint64_t n) {
    if (n == 0) return 0;
    if (n <= 3) return 1;
    
    // Newton's method
    uint64_t x = n;
    uint64_t y = (x + 1) / 2;
    
    while (y < x) {
        x = y;
        y = (x + n / x) / 2;
    }
    
    return x;
}

case OP_ISQRT:
    result = isqrt64(src1);
    break;
```

**Alternative (faster, approximate):**
```cuda
__device__ uint64_t isqrt64_fast(uint64_t n) {
    // Use floating point sqrt with correction
    double d = sqrt((double)n);
    uint64_t r = (uint64_t)d;
    
    // Correct for rounding errors
    if (r * r > n) r--;
    if ((r+1) * (r+1) <= n) r++;
    
    return r;
}
```

**Properties:**
- **Expensive**: 10-50+ cycles depending on implementation
- No branches in inner loop (Newton's method)
- Low frequency (3.9%)

---

#### 13. BitRev (Opcodes 138-147)

```rust
Op2::BitRev => src1.reverse_bits()
```

**Behavior**: Reverse all 64 bits
**Example**: `0x0000000000000001 → 0x8000000000000000`

**CUDA:**
```cuda
__device__ uint64_t reverse_bits64(uint64_t n) {
    n = ((n & 0xFFFFFFFF00000000) >> 32) | ((n & 0x00000000FFFFFFFF) << 32);
    n = ((n & 0xFFFF0000FFFF0000) >> 16) | ((n & 0x0000FFFF0000FFFF) << 16);
    n = ((n & 0xFF00FF00FF00FF00) >> 8)  | ((n & 0x00FF00FF00FF00FF) << 8);
    n = ((n & 0xF0F0F0F0F0F0F0F0) >> 4)  | ((n & 0x0F0F0F0F0F0F0F0F) << 4);
    n = ((n & 0xCCCCCCCCCCCCCCCC) >> 2)  | ((n & 0x3333333333333333) << 2);
    n = ((n & 0xAAAAAAAAAAAAAAAA) >> 1)  | ((n & 0x5555555555555555) << 1);
    return n;
}

case OP_BITREV:
    result = reverse_bits64(src1);
    break;
```

**Optimization (PTX intrinsic):**
```cuda
__device__ uint64_t reverse_bits64(uint64_t n) {
    uint32_t lo = n, hi = n >> 32;
    uint32_t lo_rev, hi_rev;
    
    asm("brev.b32 %0, %1;" : "=r"(lo_rev) : "r"(lo));
    asm("brev.b32 %0, %1;" : "=r"(hi_rev) : "r"(hi));
    
    return ((uint64_t)lo_rev << 32) | hi_rev;
}
```

**Properties:**
- Moderate cost (6-8 cycles for software, 2-4 for PTX)
- No branches
- Low frequency (3.9%)

---

## Execution Side Effects

### Always

1. **Instruction digest update**: Every instruction
   ```rust
   vm.prog_digest.update_mut(&prog_chunk);  // 20 bytes
   ```

2. **IP increment**: Every instruction
   ```rust
   vm.ip = vm.ip.wrapping_add(1);
   ```

3. **Destination register write**: Every instruction
   ```rust
   vm.regs[r3 as usize] = result;
   ```

### Conditional (Based on operand types)

1. **Memory operand**: ROM access + mem_digest update + counter increment
2. **Special1 operand**: prog_digest clone + finalize
3. **Special2 operand**: mem_digest clone + finalize
4. **Hash operation**: Blake2b hash (16 bytes → 64 bytes)

---

## Performance Characteristics

### Instruction Costs (GPU cycles, approximate)

| Operation | Best Case | Worst Case | Average | Frequency |
|-----------|-----------|------------|---------|-----------|
| Add | 1 | 1000+ | 50 | 15.6% |
| Mul | 4 | 1000+ | 50 | 15.6% |
| MulH | 8 | 1000+ | 100 | 6.3% |
| Div | 20 | 1000+ | 100 | 6.3% |
| Mod | 20 | 1000+ | 100 | 6.3% |
| Xor | 1 | 1000+ | 50 | 15.6% |
| And | 1 | 1000+ | 50 | 3.1% |
| Hash | 2000 | 3000 | 2500 | 3.1% |
| Neg | 1 | 1000+ | 50 | 7.8% |
| RotL | 2 | 1000+ | 50 | 6.3% |
| RotR | 2 | 1000+ | 50 | 6.3% |
| ISqrt | 20 | 1000+ | 100 | 3.9% |
| BitRev | 4 | 1000+ | 50 | 3.9% |

**Worst case**: Special1/Special2 operands add ~1000 cycles
**Average**: Includes typical operand mix (70% Reg/Literal, 25% Memory, 5% Special)

### Aggregate Costs per Hash

**Assumptions:**
- 8 loops × 256 instructions = 2048 instructions
- 70% fast operands (Reg/Literal)
- 25% Memory operands
- 5% Special operands

**Instruction execution only** (excluding crypto):
- Fast instructions: 2048 × 0.7 × 10 cycles = ~14,000 cycles
- Memory instructions: 2048 × 0.25 × 50 cycles = ~26,000 cycles
- Special instructions: 2048 × 0.05 × 1000 cycles = ~102,000 cycles
- **Total**: ~142,000 cycles (~70μs @ 2GHz GPU)

**With crypto overhead:**
- Blake2b updates: ~3000 × 100 cycles = 300,000 cycles
- Argon2H': ~17 invocations × 10M cycles = 170M cycles
- **Total**: ~170M cycles (~85ms @ 2GHz GPU)

**Bottleneck**: Argon2H' dominates (>99% of time)

---

## Branch Divergence Analysis

### Operations with Branches

1. **Div**: `if src2 == 0`
   - Divergence probability: ~1/2^64 (negligible)
   
2. **Mod**: `if src2 == 0`
   - Same as Div

3. **Operand loading**: `match op_type`
   - Divergence: ~50% of threads take different paths
   - **High impact**: Every instruction

### Mitigation Strategies

1. **Predicated execution**: Use CUDA `__all()`, `__any()` for warp-level decisions
2. **Operand type sorting**: Not practical (destroys determinism)
3. **Accept divergence**: Cost is acceptable given low frequency

---

## CUDA Instruction Execution Template

```cuda
__device__ void execute_instruction(
    VM* vm,
    cudaTextureObject_t rom_texture,
    const uint8_t* instruction_bytes
) {
    // Decode
    Instruction instr = decode_instruction(instruction_bytes);
    
    // Load operands
    uint64_t src1 = load_operand(vm, rom_texture, &instr, 1);  // op1
    uint64_t src2 = load_operand(vm, rom_texture, &instr, 2);  // op2
    
    // Execute
    uint64_t result;
    switch (get_operation(instr.opcode)) {
        case OP_ADD: result = src1 + src2; break;
        case OP_MUL: result = src1 * src2; break;
        // ... (all operations)
    }
    
    // Write result
    vm->regs[instr.r3] = result;
    
    // Update prog_digest
    blake2b_update(&vm->prog_digest, instruction_bytes, 20);
    
    // Increment IP
    vm->ip++;
}
```

---

## Summary for CUDA Implementation

### Instruction Format
- **Size**: 20 bytes per instruction
- **Decoding**: Simple bit operations
- **CUDA**: Struct with decoded fields

### Operations
- **Count**: 13 unique operations
- **Types**: 8 Op3 (two operands), 5 Op2 (one operand)
- **Complexity**: Range from 1-cycle (Add) to 2000-cycle (Hash)

### Operands
- **Types**: 5 variants (Reg, Memory, Literal, Special1, Special2)
- **Fast**: Reg, Literal
- **Slow**: Memory, Special1, Special2
- **Branch divergence**: Moderate (operand type selection)

### Performance
- **Bottleneck**: Argon2H' (~99% of time)
- **Instruction execution**: <1% of time
- **Crypto overhead**: Blake2b updates (~1% of time)

### Critical Details
- **Mod bug**: Uses division instead of modulo (maintain for compatibility)
- **Special operands**: Expensive Blake2b finalizations
- **Hash operation**: Full Blake2b per invocation

### Next Steps
- Analyze memory access patterns (ROM usage)
- Study digest state management
- Design CUDA kernel architecture
- Implement crypto primitives

