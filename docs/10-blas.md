# 10 — Comparing with BLAS

BLAS specifies linear-algebra operations; OpenBLAS, BLIS and oneMKL are
implementations. Performance depends on the implementation, its build,
selected hardware kernels and threading configuration.

## Matching the operation

Single-precision general matrix multiplication is called SGEMM:

```text
C = alpha * op(A) * op(B) + beta * C
```

With no transposes, `alpha=1` and `beta=0`, this becomes `C = A*B` and must
overwrite the previous output. Row-major leading dimensions are the distances
between row starts: K for A, N for B and N for C in contiguous storage.
Library dimension arguments may be 32-bit or 64-bit integers, which can differ
from a caller's `size_t` interface.

## Matching the measurement

Useful comparisons hold precision, input data, dimensions, validation and
thread limits constant. Configured thread counts are limits, not proof that
every small call uses that many workers. Thread-control variables depend on
the library's threading backend.

An ordinary library call can pack inputs and manage internal workspace; those
per-call costs belong in timings. Warmed repeated calls and first-call startup
latency are different measurements. Different reduction orders and FMA can
change rounding, so numerical validation remains necessary.

Tuned libraries combine architecture-specific microkernels, data packing,
cache blocking and scheduling. A library comparison establishes a practical
reference point. A gap suggests opportunities to investigate; its size alone
does not identify which optimization explains it. A custom kernel can also
be competitive for particular shapes without matching a library across its
full range of workloads.
