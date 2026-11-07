# 15 — Combining GPU optimizations

Block tiling, register tiling and memory layout address different reuse
levels. Larger output tiles amortize input loads across more arithmetic.
Warp-aligned column access permits contiguous B reads and output stores;
threads within a warp can broadcast shared A values while keeping several
independent accumulation chains in registers.

Software prefetching issues loads for a future reduction tile before computing
with the current shared tile. Loaded values remain in registers until all
threads finish reading shared memory. A barrier then permits replacement,
followed by another barrier before consumption. This pipeline requires neither
Tensor Cores nor hardware asynchronous-copy instructions. Its benefit depends
on compiler scheduling and the additional register pressure.

Block dimensions, output tile dimensions and per-thread output count are
separate choices. Larger tiles improve reuse but increase register/shared
storage and can leave fewer active blocks. Partial tiles also waste arithmetic.
Small, narrow or short-reduction workloads can favor a simpler kernel.

A small set of measured dimension-based rules can choose between strategies
without performing a runtime search. Sweep a few configurations, preserve
correctness tests around boundaries, and confirm choices with representative
data. A local tuning result is not a guarantee for another GPU or power state.

Compare both device-resident compute time and full host-array latency with a
matching-precision library reference. Neither individual optimization speedups
nor theoretical reuse ratios predict the final combined speedup by themselves.

NVIDIA's [efficient GEMM guide](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
illustrates hierarchical tiling and software pipelines in more advanced kernels.
