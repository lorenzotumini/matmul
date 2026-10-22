# A route through the project

This is a sequence of experiments, not a requirement to implement everything
at once. At each stage: predict, implement one change, test, measure, explain.
Keep the previous versions available. Commit meaningful stages if you put the
project under Git; write notes about failed hypotheses too.

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

SIMD and threads are distinct forms of parallel work. Learn each separately
before combining them. Loop interchange and tiling keep the usual cubic
arithmetic count, but can dramatically change memory traffic and reuse.

Matrix multiplication does not have the same simple FFT-to-pointwise-product
step as ordinary convolution. Strassen is a later algorithmic experiment:
it changes the asymptotic arithmetic count but also introduces additions,
temporary buffers, numerical concerns, and a practical crossover size.

## Explore languages without losing the experiment

After you understand the C baseline, make small side branches if useful:

- A tiny Python version is good for expressing the math and seeing interpreter
  overhead. NumPy matrix multiplication typically delegates to compiled code;
  it is a library/backend experiment as well as a language experiment.
- Reimplement the same contract in Rust, C++ or Fortran to compare ownership,
  array layout, bounds checking and compiler output. Start with the same
  algorithm, precision and data. Include FFI overhead when crossing languages.
- Study the C ABI when calling a C library from another language. Keep reference
  fixtures shared rather than inventing a different test definition per language.

These are optional detours, not prerequisites for CPU or GPU optimization.

## Local GPU path

The initial machine inspection on 2026-10-22 found:

- Intel Core i7-1065G7, GCC 16.1, Clang and CMake/Ninja available.
- NVIDIA GeForce MX330, compute capability **6.1**, 2 GiB VRAM (queried through
  `nvidia-smi`), and CUDA Toolkit **12.4** (`nvcc --version`).
- `cl.exe` was not on this shell's PATH. That does not establish whether Visual
  Studio Build Tools is installed; check the Developer terminal at the CUDA stage.

This hardware is a useful target for ordinary CUDA and shared-memory tiling.
It is not a Tensor Core target. Keep initial experiments small enough for the
available VRAM and account for other applications using the GPU.

Use a CUDA 12.x toolchain for this GPU. NVIDIA removed offline compilation for
architectures below 7.5 in CUDA 13. An automatic upgrade to the newest toolkit
would therefore prevent building kernels for this machine. See NVIDIA's
[architecture support guidance](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers).
The CUDA version printed by `nvidia-smi` describes driver capability; use
`nvcc --version` to identify the installed compiler toolkit.

Native Windows CUDA also requires a supported host compiler. Before that stage,
check the [CUDA 12.4 Windows installation guide](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-installation-guide-microsoft-windows/)
and select a compatible MSVC toolchain. The GCC-based CPU build does not prove
the CUDA host build is ready. No installation or CUDA build is needed now.

## Primary references to consult when needed

- [CMake tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
  and [CTest](https://cmake.org/cmake/help/latest/module/CTest.html).
- [GCC optimization options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)
  and [optimization reports](https://gcc.gnu.org/onlinedocs/gcc/Developer-Options.html).
- [OpenMP specifications](https://www.openmp.org/specifications/).
- [BLAS SGEMM contract](https://www.netlib.org/lapack/explore-html/dd/d09/group__gemm_ga8cad871c590600454d22564eff4fed6b.html).
- [CUDA C++ Programming Guide, 12.4](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html)
  and [Best Practices Guide, 12.4](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-best-practices-guide/index.html).
- [cuBLAS documentation, 12.4](https://docs.nvidia.com/cuda/archive/12.4.0/cublas/index.html).

Use the guides as references for the current experiment. The first session only
needs the definition, the flat-array layout and your own three loops.
