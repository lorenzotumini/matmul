# Experiment roadmap

At each stage: predict, implement one change, test, measure, explain. Keep earlier
kernels as baselines. Only stage 1 is implemented; choose one next experiment
rather than preparing infrastructure for the whole roadmap.

| Stage | Build or investigate | What it teaches | Evidence to collect |
| --- | --- | --- | --- |
| 1 | Direct definition in C | Shapes, indexing, pointers, ownership, debugging, tests | Paper example, all shared tests pass, first Release timings |
| 2 | Compiler experiments on the same source | Preprocessing/compilation/linking, optimization, alias analysis, assembly | Debug vs Release; GCC vs Clang; vectorization reports |
| 3 | Loop interchange | Row-major layout, strides, cache lines, dependence and loop order | Square and rectangular timings; identify contiguous accesses |
| 4 | Transpose B before dot products | Trading preparation and memory for locality | Include transpose cost; separately study repeated reuse |
| 5 | Cache blocking, then packing | Working sets, cache hierarchy, tiles, tail handling | Sweep tile sizes and awkward dimensions |
| 6 | SIMD, auto first and intrinsics second | Vector lanes, FMA, registers, alignment, remainder loops | Assembly/reports, numerical differences, vector-width boundary tests |
| 7 | CPU threads (OpenMP first) | Work decomposition, races, scheduling, false sharing and overhead | Fixed thread counts from 1 upward, scaling across matrix sizes |
| 8 | A tuned BLAS comparison | APIs/ABIs, mature microkernels, separating algorithm and implementation | Compare against SGEMM with controlled BLAS thread count |
| 9 | First CUDA kernel, one output per thread | Grid/block indexing, device memory, transfer, launch and synchronization | Shared correctness checks, end-to-end timing, memory footprint |
| 10 | Tiled CUDA kernel | Coalesced loads, shared memory, barriers, occupancy and reuse | Nonmultiple tile sizes, CUDA-event timing, profiling |
| 11 | cuBLAS comparison | Library integration, row/column-major conventions, device reuse | Compare both kernel-only and transfer-inclusive cost |
| 12 | Alternative algorithms and precision | Strassen, crossover sizes, temporary storage, stability | Total time/memory and a justified accuracy policy |

Learn SIMD and threads separately before combining them. Loop interchange and
tiling change memory access and reuse while keeping the usual cubic arithmetic
count.

Optional language comparisons can come after the C baseline: keep the same
algorithm, precision, fixtures, and timing contract. A library-backed operation
also compares its backend; crossing languages adds FFI costs.

## Local GPU reminder

The initial inspection on 2026-10-22 found an NVIDIA GeForce MX330 with compute
capability 6.1 and 2 GiB VRAM, and CUDA Toolkit 12.4. Before the CUDA lesson,
recheck architecture support and the required Windows host compiler using
[NVIDIA's architecture guidance](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers)
and the [CUDA 12.4 Windows guide](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-installation-guide-microsoft-windows/).
Keep initial workloads within available VRAM.

Use [the conventions](conventions.md) to add each implementation to the shared
tests and benchmark.
