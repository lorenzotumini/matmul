# 16 — Comparing GPU multiplication with BLAS

cuBLAS provides tuned matrix operations over device-resident arrays. Its
SGEMM operation computes `C = alpha*A*B + beta*C` with float32 arrays.
Choosing alpha=1 and beta=0 gives an overwrite multiplication; old output
values must not influence the result.

The traditional cuBLAS interface uses column-major layout. A row-major
multiplication can use the same buffers through the identity
`C^T = B^T*A^T`: swap operand order and the output dimensions when interpreting
the buffers as column-major. This adaptation needs no physical transpose.
Rectangular tests reveal mistakes that square-only tests can hide.

Library handles, workspace initialization and warmup are distinct from
steady-state operation costs. Reusing a handle avoids charging repeated library
startup to every multiplication. Per-call allocations and transfers still
belong in a host-array latency measurement. CUDA events over prepared buffers
measure a different scenario: repeated device-resident computation.

A library operation can launch several kernels or perform internal preparation.
Event measurements should cover the complete operation, not just a selected
internal kernel. Validate its output against the same numerical reference.

Array storage type alone does not fully specify arithmetic precision. Explicit
math modes prevent accidental comparisons between full-precision float32 and
reduced-precision paths. Tensor Core support also depends on the GPU generation.
Once precision and measurement boundaries match, the library is a useful
reference for how effectively an implementation uses the hardware.

See the [cuBLAS reference](https://docs.nvidia.com/cuda/archive/12.4.0/cublas/index.html)
for SGEMM, data layout, handles and precision modes.
