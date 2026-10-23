# 02 — Loop order and memory access

Reordering loops can change performance without changing the mathematical
result or the number of arithmetic operations. For row-major matrix
multiplication, compare `row -> col -> p` with `row -> p -> col`, where p
walks over the shared dimension K.

## What changes in the inner loop?

| Access | Advance p: row → col → p | Advance col: row → p → col |
| --- | --- | --- |
| A | Advance one float | Reuse one value |
| B | Advance N floats, down a column | Advance one float, along a row |
| C | Accumulate one dot product, then store | Update consecutive output elements |

A cache line contains several adjacent values. Consecutive accesses can use
more of each fetched line and help hardware prefetching. Reusing one A value
across an output row also avoids requesting a different A element for every
inner-loop iteration.

Independent output columns give the compiler different SIMD opportunities
from a dependent dot-product accumulation. Loop interchange can therefore
improve locality and generated instructions at the same time.

## Accumulation requires initialization

An update using `+=` reads the previous output value. When accumulating into
C, each output starts at zero before receiving any products, including when
K is nonzero. Resetting it before every product would discard earlier terms.
A NaN initial value remains NaN after addition; a finite initial value would
silently be added to the answer.

## Interpreting a speedup

```text
speedup = baseline elapsed time / alternative elapsed time
```

A speedup applies to the measured workload, compiler and build settings.
It does not establish that cache misses caused the entire difference, that
SIMD was used, or that memory bandwidth is the remaining bottleneck.
The arithmetic work remains approximately `2*M*N*K` for either loop order.
