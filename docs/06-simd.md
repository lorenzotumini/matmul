# 06 — SIMD, vector width and FMA

SIMD applies an operation to several values, called lanes, in one instruction.
Independent output columns can be updated together: one A value is broadcast
across lanes, multiplied by adjacent B values, and added to corresponding C
values. Lanes are distinct from CPU threads and do not imply multiple cores.

## Intrinsics and widths

Intrinsics express vector operations in C. The compiler still allocates
registers and schedules instructions; they are not handwritten assembly.

| x86 instruction set | Vector type | Width | 32-bit float lanes |
| --- | --- | --- | --- |
| SSE | `__m128` | 128 bits | 4 |
| AVX | `__m256` | 256 bits | 8 |
| AVX-512 Foundation | `__m512` | 512 bits | 16 |

Unaligned load/store intrinsics do not require full vector alignment, but they
still access every lane. Incomplete groups need scalar tails or masked accesses
to stay within bounds. An aligned allocation does not guarantee aligned row
starts when row lengths are arbitrary.

## Fused multiply-add

FMA computes a product plus an addend with one final rounding:

```text
fma(a, b, c) = round(a*b + c)
separate    = round(round(a*b) + c)
```

Vector FMA operates on matching lanes; it does not sum lanes together. It can
reduce arithmetic instruction count and local rounding error, but the final
matrix result may differ from separate operations. Numerical validation still
matters. Explicit FMA intrinsics and `fmaf` request fused semantics; compiler
contraction settings govern fusion of ordinary multiply-plus-add expressions.

AVX and FMA have separate feature requirements. AVX-512 is also a family of
extensions: supported subsets and OS register-state support must match the
compiler target before specialized code executes.

## Performance takeaways

Intrinsics can produce the same instructions as compiler auto-vectorization.
Wider vectors and FMA provide more arithmetic capacity, but gains depend on
memory traffic, dependencies, instruction throughput and operating frequency.
Twice as many lanes does not guarantee twice the speed. Cache blocking and
packing improve reuse; SIMD changes how operations are executed. Their effects
interact rather than yielding speedups that simply multiply.
