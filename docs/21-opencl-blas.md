# 21 — Comparing OpenCL multiplication with BLAS

An OpenCL BLAS library supplies tuned matrix operations over device buffers.
Float32 SGEMM computes `C = alpha*A*B + beta*C`; alpha=1 and beta=0 express
overwrite multiplication. Layout, precision and scalar choices must match
the custom kernels, and rectangular tests expose leading-dimension errors.

GEMM can use a direct kernel or several kernels for padding, transposition,
multiplication and output conversion. Time the complete library operation,
including its internal preparation. The event returned by an API can describe
only the final kernel, so it may not bound the full operation.

On an in-order profiling queue, enclosing markers can capture the operation's
device-timeline window. Such a window includes inter-launch gaps and can
differ from timestamps taken directly from one custom kernel. Full host-array
latency also includes allocation, explicit copies, synchronization and cleanup.
Initial compilation and library caches are separate from steady-state costs.

Default device parameters and explicitly tuned parameters are different
experiments. Record library version, tuning state and extra compiler options.
A library baseline is a measured reference, not proof of a hardware limit.

See [CLBlast's GEMM design](https://github.com/CNugteren/CLBlast/blob/1.7.0/doc/details_gemm.md)
and [tuning guide](https://github.com/CNugteren/CLBlast/blob/1.7.0/doc/tuning.md).
