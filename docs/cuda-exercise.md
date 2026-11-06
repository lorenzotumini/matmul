# CUDA: guided naive matrix multiplication

Your goal is to write the body of `cuda_naive_kernel` in
[`src/cuda_naive.cu`](../src/cuda_naive.cu). One GPU thread owns one element
of C. Begin with the [GPU concept summary](11-gpu-execution.md).

The build, memory handling, launch and correctness checks are supplied. This
is an intentionally unfinished exercise: it compiles, but the CUDA test fails
until the kernel writes the correct result. Hardware availability does not
mean the exercise is complete.

## Run the exercise

From the repository root, in PowerShell:

```powershell
.\nob.exe test --cuda --filter cuda
```

This builds first, then runs only the `cuda` CTest check. Its first fixture
multiplies a 2-by-3 array of ones by a 3-by-2 array of ones. Every output
should be 3. An unfinished kernel leaves NaNs; that failure is expected.
After completing the four TODOs, the same command checks all 61 fixtures,
each twice. These include rectangular shapes, partial blocks, cancellation,
zero dimensions, unchanged inputs and pre-existing output values.

## 1. Decide which output belongs to this thread

The supplied launch uses 16-by-16 blocks. The x direction follows columns;
the y direction follows rows. All blocks run the same kernel function.

The built-in variables are:

| Variable | Meaning |
| --- | --- |
| `blockIdx.x`, `.y` | Which block contains this thread |
| `blockDim.x`, `.y` | Number of threads along each direction in a block |
| `threadIdx.x`, `.y` | This thread's position inside its block |

Write `row` and `col` using the block's starting position plus the thread's
local offset. Use `size_t` for indices, casting before multiplication.

Before coding, predict the column for `blockIdx.x = 2`, `blockDim.x = 16`,
`threadIdx.x = 5`. Then do the analogous calculation for a row. This is the
same translation from a local coordinate to a global coordinate in both axes.

## 2. Protect the boundaries

The host rounds the block count up. For M=17 and N=31, the launch has four
blocks and 1024 threads, but only 527 outputs. Some threads must return
without touching memory. Add the guard before any A, B or C access.

Reason through `row == m` and `col == n`, as well as larger values.
Why would a square 16-by-16 test alone be insufficient to check this?

## 3. Compute one dot product

Use a local float accumulator initialized to zero, and one loop over K.
Consult your CPU naive kernel for the row-major offsets into A and B.
There are no row or column loops here: the grid supplies that parallelism.

The pointers inside the kernel refer to GPU memory. `size_t`, local floats
and the loop syntax still look like ordinary C, although `.cu` files are
compiled as CUDA C++.

## 4. Write the output

Overwrite the one C element owned by this thread. It must not depend on its
previous value. No atomics or thread barriers are needed because threads
own disjoint output elements and only read the inputs.

Remove the unused-parameter markers when you use the parameters. Run the
test command; read the reported M, N, K and output coordinate when it fails.
A NaN usually means a missing write or invalid arithmetic. A wrong finite
answer suggests indexing or accumulation. An illegal-memory-access error
suggests a boundary or offset problem.

## After correctness passes

Run the complete suite, then measure the Release build:

```powershell
.\nob.exe test --cuda
.\nob.exe test --cuda --mode release --filter cuda
.\nob.exe bench --cuda -- --impl cuda_naive --m 256 --n 256 --k 256
.\nob.exe bench --cuda -- --impl cuda_naive --m 17 --n 31 --k 33
```

The benchmark validates output before reporting performance. Its timer covers
the complete host call: device allocations, copies to the GPU, poisoning the
output, kernel launch, synchronization, copying C back, and freeing buffers.
It excludes driver/context startup warmed by validation. It is **not a
kernel-only timing**. Tiny problems are likely to expose overhead strongly.
No CPU or GPU speed comparison is implied until measured with suitable sizes.

`nob profile` is the existing CPU/Samply workflow. CUDA device profiling and
event-based kernel timings are separate later steps.

## Supplied files and toolchain

- `src/cuda_naive.cu`: your exercise and its launch syntax.
- `src/cuda_support.cu`: device allocation, copies, error checks and waiting.
- `src/cuda_internal.cuh`: the private launch interface.
- `cmake/CUDABackend.cmake`: optional sources, compiler and architecture.

CUDA is opt-in; ordinary `nob test` keeps the CPU workflow. On Windows,
`--cuda` defaults the C host build to clang-cl, compatible with nvcc's MSVC
objects. The C sources remain C; only CUDA sources use CUDA C++17. Build
directories are separate: `build/clang-cl/cuda-debug` and `cuda-release`.
The root `.clangd` adds CUDA parsing and the installed toolkit's headers for
`.cu`/`.cuh` files, leaving the C editor flags intact. Adjust its toolkit path
if using a different installation.

This laptop has CUDA Toolkit 12.4, a GeForce MX330 (compute capability 6.1),
and VS 2022 Build Tools for nvcc. CMake prefers that VS installation over a
newer one and targets architecture 61. An explicit `CUDAHOSTCXX` or
`CMAKE_CUDA_HOST_COMPILER` overrides discovery. Other GPUs can configure
`CMAKE_CUDA_ARCHITECTURES` explicitly.
[CUDA 13 dropped offline compilation for Pascal](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/);
keep a suitable 12.x toolkit for this GPU.

For setup failures, `--verbose` shows commands; ordinary failures replay the
stage logs automatically. Do not edit toolkit headers to bypass compatibility
checks. On a machine without an available CUDA device, CTest skips the
optional CUDA check; explicit benchmark requests fail rather than falling back
to a CPU implementation.

References: [CUDA 12.4 Windows installation guide](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-installation-guide-microsoft-windows/)
and [CUDA C++ programming guide](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html).
