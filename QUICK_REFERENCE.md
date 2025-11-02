# AshMaize Quick Reference Guide

## 🚀 Quick Start

### Native Rust
```rust
use ashmaize::{Rom, RomGenerationType, hash};

let rom = Rom::new(
    b"key",
    RomGenerationType::TwoStep { pre_size: 16*1024, mixing_numbers: 4 },
    256 * 1024 * 1024  // 256 MB
);

let digest = hash(b"salt", &rom, 8, 256);
```

### WebAssembly
```javascript
import init, { Rom } from './pkg/ashmaize_web.js';

await init();
const builder = Rom.builder();
builder.key(keyBytes);
builder.size(256 * 1024 * 1024);
builder.gen_two_steps(16 * 1024, 4);
const rom = builder.build();

const digest = rom.hash(saltBytes, 8, 256);
```

---

## 📊 Algorithm Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    ASHMAIZE HASH ALGORITHM                   │
└─────────────────────────────────────────────────────────────┘

PHASE 1: ROM GENERATION (One-time, Expensive)
┌──────────────────────────────────────────────────────────────┐
│  Input: key (bytes)                                          │
│         ↓                                                    │
│  seed = Blake2b-256(size || key)                             │
│         ↓                                                    │
│  ┌──────────────────────┐  ┌─────────────────────────────┐  │
│  │   FullRandom         │  │      TwoStep                │  │
│  │  Argon2H'(seed, size)│  │  1. pre_rom = Argon2H'(...)  │  │
│  │         ↓            │  │  2. Generate offsets        │  │
│  │  ROM (size bytes)    │  │  3. XOR mix pre_rom chunks  │  │
│  └──────────────────────┘  └─────────────────────────────┘  │
│         ↓                            ↓                       │
│  rom_digest = Blake2b-512(ROM)                               │
└──────────────────────────────────────────────────────────────┘

PHASE 2: VM INITIALIZATION (Per-hash, Medium Cost)
┌──────────────────────────────────────────────────────────────┐
│  Input: rom_digest (64 bytes), salt (variable)               │
│         ↓                                                    │
│  init_buffer = Argon2H'(rom_digest || salt, 448 bytes)       │
│         ↓                                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  regs[0..31]  (32 × 8 bytes)  ← init_buffer[0..256]  │    │
│  │  prog_digest  (64 bytes)      ← init_buffer[256..320]│    │
│  │  mem_digest   (64 bytes)      ← init_buffer[320..384]│    │
│  │  prog_seed    (64 bytes)      ← init_buffer[384..448]│    │
│  └─────────────────────────────────────────────────────┘    │
│  ip ← 0,  memory_counter ← 0,  loop_counter ← 0             │
└──────────────────────────────────────────────────────────────┘

PHASE 3: EXECUTION LOOP (nb_loops iterations)
┌──────────────────────────────────────────────────────────────┐
│  FOR loop = 0 TO nb_loops-1:                                 │
│    │                                                          │
│    ├─ STEP 1: Generate Random Program                        │
│    │    program_bytes = Argon2H'(prog_seed, nb_instrs × 20)  │
│    │                                                          │
│    ├─ STEP 2: Execute Instructions                           │
│    │    FOR i = 0 TO nb_instrs-1:                            │
│    │      ├─ instr = decode(program[ip])                     │
│    │      ├─ Execute instruction:                            │
│    │      │    ┌──────────────────────────────────┐          │
│    │      │    │ Load operands (Reg/Mem/Lit/Spec) │          │
│    │      │    │         ↓                        │          │
│    │      │    │ Compute (Add/Mul/Xor/Hash/...)   │          │
│    │      │    │         ↓                        │          │
│    │      │    │ Store to destination register    │          │
│    │      │    └──────────────────────────────────┘          │
│    │      ├─ prog_digest.update(raw_instr_bytes)             │
│    │      └─ ip += 1                                         │
│    │                                                          │
│    └─ STEP 3: Post-Instruction Mixing                        │
│         sum = Σ regs[i]                                      │
│         prog_val = prog_digest.finalize_copy(sum)            │
│         mem_val = mem_digest.finalize_copy(sum)              │
│         mixing = Argon2H'(Blake2b(prog_val || mem_val || LC))│
│         FOR round = 0 TO 31:                                 │
│           FOR reg = 0 TO 31:                                 │
│             regs[reg] ^= mixing[...]                         │
│         prog_seed ← prog_val                                 │
│         loop_counter += 1                                    │
└──────────────────────────────────────────────────────────────┘

PHASE 4: FINALIZATION
┌──────────────────────────────────────────────────────────────┐
│  digest = Blake2b-512(                                       │
│    prog_digest.finalize() ||      # 64 bytes                 │
│    mem_digest.finalize()  ||      # 64 bytes                 │
│    memory_counter         ||      # 4 bytes                  │
│    regs[0] || ... || regs[31]     # 256 bytes                │
│  )                                                           │
│         ↓                                                    │
│  Output: 64-byte digest                                      │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔧 Instruction Set Reference

### Instruction Format (20 bytes)
```
┌────────┬────────┬────────┬─────────────────┬─────────────────┐
│ OpCode │Op1|Op2 │ R1|R2|R3│      Lit1       │      Lit2       │
│ 1 byte │ 1 byte │ 2 bytes│    8 bytes      │    8 bytes      │
└────────┴────────┴────────┴─────────────────┴─────────────────┘
```

### Opcode Map
| Range    | Instruction | Operation |
|----------|-------------|-----------|
| 0-39     | Add         | `dst = src1 + src2` |
| 40-79    | Mul         | `dst = src1 * src2` |
| 80-95    | MulH        | `dst = (src1 * src2) >> 64` |
| 96-111   | Div         | `dst = src1 / src2` |
| 112-127  | Mod         | `dst = src1 % src2` ⚠️ BUG: currently does division |
| 128-137  | ISqrt       | `dst = floor(sqrt(src1))` |
| 138-147  | BitRev      | `dst = reverse_bits(src1)` |
| 148-187  | Xor         | `dst = src1 ^ src2` |
| 188-203  | RotL        | `dst = rotate_left(src1, r1)` |
| 204-219  | RotR        | `dst = rotate_right(src1, r1)` |
| 220-239  | Neg         | `dst = ~src1` |
| 240-247  | And         | `dst = src1 & src2` |
| 248-255  | Hash[N]     | `dst = Blake2b(src1 ‖ src2)[N*8..(N+1)*8]` |

### Operand Types
| Value | Type     | Source |
|-------|----------|--------|
| 0-4   | Register | VM registers |
| 5-8   | Memory   | ROM[lit % chunks] |
| 9-12  | Literal  | Immediate value |
| 13-14 | Special1 | prog_digest hash |
| 14-15 | Special2 | mem_digest hash |

---

## 🎯 Parameter Tuning Guide

### ROM Size Selection
```
Development/Testing:   1 MB - 16 MB
Light PoW (CAPTCHA):   64 MB - 256 MB
Standard Mining:       256 MB - 1 GB
High Security:         1 GB - 2 GB
```

### Generation Type Trade-offs
```
FullRandom:
  ✅ Maximum ASIC resistance
  ❌ Slower initialization
  📋 Use when: ROM lifetime > 1 hour

TwoStep:
  ✅ Faster initialization (10-100x)
  ❌ Slightly reduced ASIC resistance
  📋 Use when: Frequent re-generation needed
  
  Recommended: pre_size = ROM_size / 64 to 128
               mixing_numbers = 4
```

### Execution Parameters
```
nb_loops:
  Testing:     4-8
  Production:  8-16
  Max:         Limited by time budget
  
nb_instrs:
  Minimum:     256 (enforced)
  Standard:    256-512
  Heavy:       512-1024
  
Total work = nb_loops × nb_instrs instructions
```

### Difficulty Calibration
```
Leading zero bits:  Expected attempts
─────────────────────────────────────
8 bits              2^8    = 256
12 bits             2^12   = 4,096
16 bits             2^16   = 65,536
20 bits             2^20   = 1,048,576
24 bits             2^24   = 16,777,216

Target solve time = expected_attempts / hash_rate
```

---

## 🐛 Known Issues

### 1. Modulo Operation Bug
**Location**: `src/lib.rs:393`
```rust
// CURRENT (WRONG):
Op3::Mod => {
    if src2 == 0 {
        special1_value64!(vm)
    } else {
        src1 / src2  // ❌ Should be modulo
    }
}

// CORRECT:
Op3::Mod => {
    if src2 == 0 {
        special1_value64!(vm)
    } else {
        src1 % src2  // ✅ Fixed
    }
}
```
**Impact**: Modulo instruction behaves identically to division  
**Severity**: Medium (affects instruction distribution, not security)  
**Fix**: Change `/` to `%` operator

---

## 📈 Performance Expectations

### Initialization Time (Approximate)
```
ROM Size    | FullRandom  | TwoStep (16MB pre)
─────────────────────────────────────────────
16 MB       | ~200ms      | ~50ms
256 MB      | ~3s         | ~200ms
1 GB        | ~12s        | ~500ms
2 GB        | ~25s        | ~1s
```

### Hash Rate (Single Core, Modern CPU)
```
ROM: 256 MB, Loops: 8, Instructions: 256
Native:     ~500-2000 H/s
WASM:       ~100-500 H/s (20-50% of native)
```

### Memory Footprint
```
ROM:            configured size (256MB - 2GB)
VM State:       ~512 bytes (registers + counters)
Program:        20 × nb_instrs bytes
Working Memory: ~10KB (digest contexts, buffers)

Total ≈ ROM_size + 100KB
```

---

## 🔐 Security Best Practices

### ✅ DO
- Rotate ROM keys periodically (hourly/daily)
- Use TwoStep for low-latency applications
- Validate input sizes (prevent DOS via huge ROM)
- Use secure random for key generation
- Store ROM in memory (don't regenerate per hash)

### ❌ DON'T
- Use same ROM key forever (enables pre-computation)
- Set ROM size < 64 MB (weak ASIC resistance)
- Use nb_loops < 2 (enforced, but avoid boundary)
- Use nb_instrs < 256 (enforced, but avoid boundary)
- Share ROM between security contexts

---

## 🌐 WASM Build Commands

### Development Build
```bash
cd crates/ashmaize-web
wasm-pack build --dev --target web
```

### Production Build
```bash
cd crates/ashmaize-web
wasm-pack build --release --target web
# Output: pkg/ashmaize_web.js + .wasm
```

### Test in Browser
```bash
wasm-pack test --headless --chrome
```

### Optimized Build
```bash
# Add to Cargo.toml:
[profile.release]
opt-level = 'z'     # Optimize for size
lto = true          # Link-time optimization
codegen-units = 1   # Single codegen unit

wasm-pack build --release
wasm-opt -Oz -o optimized.wasm pkg/ashmaize_web_bg.wasm
```

---

## 📚 Key Files Reference

| File | Purpose | Lines |
|------|---------|-------|
| `src/lib.rs` | Core VM, execution engine | ~537 |
| `src/rom.rs` | ROM generation algorithms | ~369 |
| `crates/ashmaize-web/src/lib.rs` | WASM bindings | ~135 |
| `SPECS.md` | Algorithm specification | ~300 |
| `examples/hash.rs` | Multi-threaded mining example | ~200 |
| `benches/bench.rs` | Performance benchmarks | ~200 |

---

## 🧪 Test Commands

```bash
# Run all tests
cargo test

# Run only native tests
cargo test --lib

# Run WASM tests
cd crates/ashmaize-web
wasm-pack test --headless --chrome

# Run benchmarks
cargo bench

# Run example
cargo run --release --example hash
```

---

## 🔗 Cryptographic Primitives

### Blake2b-512
- **Library**: `cryptoxide::hashing::blake2b`
- **Output**: 64 bytes (512 bits)
- **Speed**: ~3 GB/s (native), ~1 GB/s (WASM)
- **Used for**: Digests, seeds, finalization

### Argon2H'
- **Library**: `cryptoxide::kdf::argon2::hprime`
- **Type**: Custom variant (not standard Argon2)
- **Method**: Hash-based expansion (sequential)
- **Used for**: ROM generation, mixing, program generation

**Important**: Argon2H' ≠ Argon2. It's a simplified sequential hash chain.

---

## 💡 Common Pitfalls

### 1. ROM Not Shared Across Hashes
```rust
// ❌ BAD: Regenerates ROM every hash
for salt in salts {
    let rom = Rom::new(...);  // SLOW!
    hash(&salt, &rom, 8, 256);
}

// ✅ GOOD: Reuse ROM
let rom = Rom::new(...);  // Once
for salt in salts {
    hash(&salt, &rom, 8, 256);  // Fast
}
```

### 2. Integer Overflow in Parameters
```rust
// ❌ BAD: Overflow in bytes calculation
let rom_size = usize::MAX;  // Will panic

// ✅ GOOD: Validate sizes
const MAX_ROM: usize = 4 * 1024 * 1024 * 1024;  // 4 GB
assert!(rom_size <= MAX_ROM);
```

### 3. WASM Memory Limits
```javascript
// ❌ BAD: Browser may OOM on 2GB ROM
const rom = builder.size(2 * 1024 * 1024 * 1024).build();

// ✅ GOOD: Use reasonable sizes for WASM
const rom = builder.size(256 * 1024 * 1024).build();  // 256 MB
```

---

## 📖 Further Reading

- **SPECS.md**: Formal algorithm specification with pseudocode
- **RandomX**: https://github.com/tevador/RandomX (comparison)
- **Argon2**: https://github.com/P-H-C/phc-winner-argon2 (inspiration)
- **Blake2**: https://www.blake2.net/ (hash function)

---

**Last Updated**: October 27, 2025  
**Version**: 1.0  
**For**: ce-ashmaize development reference
