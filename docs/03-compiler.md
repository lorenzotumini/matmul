# 03 — Compilers and generated code

C source describes a computation; the compiler determines the machine
instructions used to perform it. Similar source can produce different code,
and different source can compile to similar instructions.

## Translation and optimization

Preprocessing expands includes and macros. Compilation translates the program
into assembly; assembling produces object files. Linking combines objects
and libraries into an executable.

Optimization can reuse loaded values, remove redundant work, unroll loops and
use SIMD. A named temporary does not by itself prove fewer memory loads:
the compiler may already reuse the value. Debug and Release are build labels;
the actual compiler options determine optimization and debug information.

## SIMD and dependencies

SIMD performs the same operation on several data lanes. Independent loop
iterations are often easier to vectorize than a reduction into one accumulator.
Floating-point addition depends on evaluation order, which constrains how
reductions can be transformed without relaxing numerical semantics.

Potential pointer overlap also constrains optimization. `const` prevents writes
through a pointer but does not guarantee non-overlap. A compiler may generate
runtime overlap checks and select between scalar and vectorized paths.

## Reading a small assembly example

This x86 example uses Intel syntax. Here, `rdx` points to B, `rax` points to C,
`r8` is a byte offset, and `xmm1` contains four copies of one scalar value alpha:

```asm
movups xmm0, [rdx+r8]    ; load four consecutive B floats
movups xmm3, [rax+r8]    ; load four consecutive C floats
mulps  xmm0, xmm1        ; multiply all four B lanes by alpha
addps  xmm0, xmm3        ; add to the four C lanes
movups [rax+r8], xmm0    ; store four updated C floats
add    r8, 16            ; advance by four 4-byte floats
```

The arithmetic updates four independent output elements. A report describing
"16 byte vectors" corresponds to four 32-bit floats, not sixteen elements or
sixteen threads. `movups` allows memory accesses without 16-byte alignment.

A dot product can also use `mulps` for four products, then shuffle the lanes
and use scalar `addss` instructions to accumulate them into one sum in order.
That still counts as vectorization, but has different dependencies and costs
from packed `mulps` plus `addps` on independent outputs. Strided inputs may
also require separate loads and packing instead of one contiguous load.

**"Both loops vectorized" does not mean they perform the same work per
instruction or have the same performance.**

## What the evidence means

- Vectorization reports describe compiler decisions for particular source
  locations. An outer loop may fail to vectorize while an inner loop succeeds.
- Assembly reveals the generated instructions. On x86, `mulss` and `addss` are
  scalar float operations; `mulps` and `addps` operate on packed float lanes.
  An `xmm` register name alone does not prove SIMD arithmetic.
- A vectorized path can coexist with scalar fallback and remainder loops.
  Its presence does not establish that every matrix shape uses it.

Reports and assembly explain generated code. Timings measure performance;
hardware counters provide evidence about cache behavior and execution stalls.
No single kind of evidence explains every bottleneck.
