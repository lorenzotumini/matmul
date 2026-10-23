# 05 — Explicit SIMD and vector width

SIMD applies an operation to several values, called lanes, in one instruction.
For matrix multiplication, independent output columns can be updated together:
one A value is broadcast across lanes, multiplied by adjacent B values, and
added to the corresponding C values.

## Intrinsics and widths

Intrinsics are compiler-recognized functions that express vector operations
in C. The compiler still allocates registers and schedules instructions; they
are not handwritten assembly.

| x86 instruction set | Vector type | Width | 32-bit float lanes |
| --- | --- | --- | --- |
| SSE | `__m128` | 128 bits | 4 |
| AVX | `__m256` | 256 bits | 8 |

Basic eight-float loads, multiplication, addition and stores need AVX, not AVX2.
Fused multiply-add is another feature: it combines multiplication and addition
with one final rounding, so it can produce different numerical results from
separate operations.

Unaligned load/store intrinsics accept addresses without the vector's full
alignment requirement. They still access every lane: incomplete groups need
a remainder path to avoid accessing beyond the array. Aligned allocations do
not guarantee aligned row starts when row lengths are arbitrary.

## Performance takeaways

Explicit SIMD may produce essentially the same instructions as compiler
auto-vectorization. Expressing those operations manually does not guarantee
a speedup. Wider vectors can improve throughput, but doubling the lane count
does not guarantee twice the performance: memory traffic, dependencies, loop
overhead and hardware execution resources also matter.

SIMD and cache blocking address different concerns: SIMD performs several
operations together; blocking improves data reuse. Their benefits can combine,
but one does not replace the other. SIMD lanes also differ from CPU threads:
multiple lanes do not imply multiple cores.

Optional instruction sets require both CPU and operating-system support.
Keeping specialized code separate allows baseline code and feature detection
to run on machines without those extensions.
