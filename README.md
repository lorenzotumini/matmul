Learn matrix multiplication by implementing one technique at a time, testing it,
and measuring it against earlier versions.

## Workflow

1. Read [the kernel contract](include/matmul.h).
2. Use the concept summaries below as a reference.
3. Run the commands below to check correctness and take a measurement.
4. Follow [the conventions](docs/conventions.md) when adding an implementation.

## Concept summaries

1. [Matrix multiplication basics](docs/01-definition.md): shapes, indexing and correctness.
2. [Loop order and memory access](docs/02-loop-interchange.md): locality and reused values.
3. [Compilers and generated code](docs/03-compiler.md): optimization and assembly.
4. [Cache blocking](docs/04-blocking.md): working sets and cache reuse.
5. [Packing matrix panels](docs/05-packing.md): temporary layouts and copy costs.
6. [SIMD and FMA](docs/06-simd.md): vector widths, fused arithmetic and rounding.
7. [Register tiling](docs/07-register-tiling.md): keeping partial sums in registers.
8. [CPU parallelism](docs/08-parallelism.md): ownership, scheduling and scaling.
9. [Combining CPU optimizations](docs/09-cpu-optimization.md): layered reuse and practical tradeoffs.
10. [Comparing with BLAS](docs/10-blas.md): library interfaces and fair measurements.
11. [GPU execution and memory](docs/11-gpu-execution.md): grids, blocks, ownership and transfers.
12. [GPU shared-memory tiling](docs/12-gpu-tiling.md): cooperative loading, reuse and barriers.
13. [GPU register tiling](docs/13-gpu-register-tiling.md): multiple outputs per thread and register reuse.
14. [GPU software prefetching](docs/14-gpu-prefetch.md): staging future inputs in registers, latency hiding and barriers.
15. [Combining GPU optimizations](docs/15-gpu-combination.md): hierarchical reuse, software prefetch and dimension-based selection.
16. [Comparing with GPU BLAS](docs/16-gpu-blas.md): cuBLAS, precision modes and measurement boundaries.
17. [OpenCL and integrated GPUs](docs/17-opencl.md): work-items, runtime compilation, shared RAM and device timing.
18. [OpenCL local-memory tiling](docs/18-opencl-tiling.md): work-group reuse, padded ranges and barriers.
19. [OpenCL register tiling](docs/19-opencl-register-tiling.md): multiple outputs per work-item and private reuse.
20. [OpenCL BLAS](docs/20-opencl-blas.md): CLBlast, tuning state and whole-operation measurements.

The benchmark follows this learning order, from CPU implementations and BLAS
to optional GPU backends. This is an order of complexity; matrix shape and hardware still
influence which implementation is faster.

## Build, test, measure

Install GCC, CMake 3.21 or newer, and Ninja, with each on PATH. The examples run
from the repository root. Compile the command runner
once:

```sh
gcc nob.c -o nob
```

Use Nob for the everyday workflow:

```sh
./nob test
./nob bench
./nob profile
```

- `test` builds in Debug and runs the shared correctness tests.
- `bench` builds in Release, validates output, and reports timings and GFLOP/s.
- `profile` records CPU hotspots with Samply. It needs additional tooling;
  follow [the profiling guide](docs/profiling.md) when ready.

Every action builds first. Nob rebuilds itself when its source changes.
Arguments before `--` configure Nob; arguments after it configure the test or
benchmark runner. Use `./nob --help` for build options and
`./nob bench -- --help` for workload options.

## CUDA implementations

The GPU exercises use the same correctness tests and host-side memory handling:

- [Naive CUDA](src/cuda_naive.cu): one output per thread.
- [Shared-memory tiling](docs/12-gpu-tiling.md): cooperative input reuse in
  [`src/cuda_tiled.cu`](src/cuda_tiled.cu).
- [Register tiling](docs/13-gpu-register-tiling.md): four outputs per thread in
  [`src/cuda_register_tiled.cu`](src/cuda_register_tiled.cu).
- [Software prefetching](docs/14-gpu-prefetch.md): next-tile inputs staged in
  registers in [`src/cuda_prefetch.cu`](src/cuda_prefetch.cu), keeping the
  register-tiled baseline's geometry, ownership and arithmetic.
- [Combined CUDA](src/cuda_combined.cu): readable warp-aligned tiles, register
  reuse, software prefetch and a small dimension-based dispatcher.

```powershell
.\nob.exe test --cuda --filter cuda_naive
.\nob.exe test --cuda --filter cuda_tiled
.\nob.exe test --cuda --filter '^cuda_register_tiled$'
.\nob.exe test --cuda --filter '^cuda_prefetch$'
```

A filter of `cuda` runs custom GPU checks and the timing check; `cublas`
selects the library reference check. Use the implementation name to focus
on one, or omit the filter to run the complete suite.
`--cuda` opts into a separate CUDA build and selects clang-cl on Windows;
ordinary CPU commands keep their defaults. After correctness passes, use
`nob bench --cuda -- --impl cuda_naive` (or `cuda_tiled`) to see complete
host-call latency and separate GPU kernel diagnostics together.

To isolate prefetching, compare `cuda_register_tiled` and `cuda_prefetch` with
the same dimensions and sampling options:

```powershell
.\nob.exe bench --cuda -- --impl cuda_register_tiled --m 512 --n 512 --k 512
.\nob.exe bench --cuda -- --impl cuda_prefetch --m 512 --n 512 --k 512
```

Both use 16x16 threads, 32x32 output tiles, depth-16 shared inputs and four
adjacent outputs per thread. The prefetched version primes the first tile,
fetches the next one into registers before computing, then publishes it after
all shared-memory readers finish. Register pressure can offset any benefit.
The combined backend uses this scheduling idea with its own tile/layout rules.

### CUDA timing

The default `--timing both` reports full-call milliseconds for every backend,
then adds a separate kernel-time row for each GPU implementation:

```powershell
.\nob.exe bench --cuda -- --impl cuda_naive --m 512 --n 512 --k 512
.\nob.exe bench --cuda -- --impl cuda_tiled --m 512 --n 512 --k 512
```

Compare the **full-call milliseconds** across CPU and GPU: CPU calls include
packing and thread coordination; GPU calls include device allocations,
input/output copies, output poisoning and cleanup. CPU rows retain clearly
labeled effective GFLOP/s. GPU full-call rows show no GFLOP/s; CUDA-event rows
show kernel GFLOP/s as a diagnostic of device-resident computation.

Use `--timing host` for full-call results only, or `--timing kernel` for only
the device-event measurement. The kernel-only option requires one named GPU
implementation; it rejects `--impl all`, CPU implementations and internal
profile/validation modes. Default `both` works with `--impl all`. Inputs
are copied to device buffers once and reused for validation, warmups and samples.
Allocation, transfers, output poisoning and event creation/destruction stay
outside the event interval. Results are copied back and checked before reporting
times and after every sample. The ordinary host-array kernels retain their
original behavior.

`--batch B` launches B kernels between the start/stop events and reports time
per launch. For tiny kernels, a larger batch (for example 100) can reduce the
relative effect of event resolution. GPU scheduling and gaps between host
launches can still affect this interval; do not interpret it as an exact
instruction-level kernel duration. These are warm, reused-buffer measurements.

Human rows label `full-call` or `kernel`. CSV has a `timing` column after
`checksum`, containing `host` or `kernel`; default GPU runs produce two rows.
The `gflops` field is empty for GPU host rows, numeric for CPU effective
throughput and GPU kernel throughput. Metadata stays on stderr. Keep those
throughput scopes separate. The two GPU rows are separate experiments, so
subtracting their times does not isolate an exact transfer cost.

### cuBLAS reference

CUDA builds also discover the toolkit's cuBLAS library, when available. The
`cublas` entry uses float32 SGEMM with `alpha=1`, `beta=0` and explicit
`CUBLAS_PEDANTIC_MATH`. It adapts row-major arrays by swapping operands and
dimensions, without a transpose copy. Your learning kernels are unchanged.

```powershell
.\nob.exe test --cuda --filter cublas
.\nob.exe bench --cuda -- --impl cuda_register_tiled --m 512 --n 512 --k 512
.\nob.exe bench --cuda -- --impl cublas --m 512 --n 512 --k 512
```

Both full-call and CUDA-event rows use the existing validator and sampling
settings. cuBLAS reuses a handle per host thread on device 0; handle creation
and initial library warmup are outside sampling. Full calls include the same
allocation, transfers, output poisoning and cleanup as the learning wrappers.
Event timing covers the whole cuBLAS operation, including any library GPU
preparation or multiple kernels, over resident device buffers. It excludes
handle creation and host transfers, just like the prepared custom-kernel path.

The metadata reports the cuBLAS version and precision mode. On Windows,
the CUDA toolkit's `bin` directory must remain on PATH for its cuBLAS DLLs
(normally it is already there for `nvcc`). CMake's `MATMUL_ENABLE_CUBLAS`
option can disable this optional reference without disabling CUDA exercises.

### Combined CUDA reference

`cuda_combined` keeps the ordinary host-array contract and both timing paths.
Its dispatcher and kernel are in `src/cuda_combined.cu`. All arithmetic uses
float32 CUDA-core FMA; this backend does not call cuBLAS or use Tensor Cores.

| Ordered selection rule | Path |
| --- | --- |
| M < 32, N < 32, or K < 16 | Existing 16x16 shared-memory kernel |
| M*N*K < 128 cubed | Existing 32x32 register-tiled kernel |
| M < 128 | 32x64 prefetched tile if N >= 512; otherwise existing register kernel |
| M,N >= 512 and K >= 128 | 128x128 prefetched tile |
| Remaining shapes | 64x64 prefetched tile |

The new kernels use 256 threads arranged as 32 columns by 8 warps. Each
thread owns multiple rows and groups of 32-spaced columns. Shared inputs are
cooperatively loaded; scalars for the next depth-16 reduction tile are prefetched
into registers before current-tile arithmetic. All boundary loads and stores
are guarded. No allocation, data transfers or tuning search occurs inside the
device kernels. Allocation and transfers remain inside full-call measurements.

These heuristics came from a small local MX330 sweep, not a general GPU
autotuner. Larger tiles and prefetch can trade reuse for occupancy. Small and
awkward shapes can still favor another kernel or library algorithm.

```powershell
.\nob.exe test --cuda --filter 'cuda_combined|cuda_timing'
.\nob.exe bench --cuda -- --impl cuda_combined --m 512 --n 512 --k 512
.\nob.exe bench --cuda -- --impl cublas --m 512 --n 512 --k 512
```

## Intel GPU OpenCL exercise

Completed baselines are `src/opencl_naive.cl` and `src/opencl_tiled.cl`.
The next [register-tiling exercise](docs/opencl-register-exercise.md) is in
[`src/opencl_register_tiled.cl`](src/opencl_register_tiled.cl), with TODOs for
the dot products and cooperative loading. Host plumbing is supplied. Registry
order is naive, tiled, register tiled, then CLBlast. The runtime selects
the first available Intel GPU with a compiler, never a CPU OpenCL device or
the NVIDIA GPU. A missing Intel GPU is skipped; incorrect output is a failure.

```powershell
.\nob.exe bench --opencl -- --list
.\nob.exe test --opencl --filter opencl
.\nob.exe test --opencl --filter '^opencl_register_tiled$'
# After completing the exercise and passing correctness:
.\nob.exe bench --opencl -- --impl opencl_tiled --m 256 --n 256 --k 256
```

`--opencl` enables a separate build, normally with GCC. It requires OpenCL
1.2 or newer development headers/import library plus the Intel graphics
driver's OpenCL runtime. CMake can discover the vendor-neutral OpenCL SDK
files in an installed CUDA toolkit; using those headers does not select the
NVIDIA device. Without that SDK, install an OpenCL SDK and configure CMake's
`OpenCL_INCLUDE_DIR` and `OpenCL_LIBRARY` if discovery needs help. No SYCL or CUDA compiler is needed. CLBlast additionally needs a matching
C++ compiler; its pinned source is downloaded on first configuration and built
statically. Ordinary CPU/CUDA-only builds do not fetch or build it.

Each `.cl` source is read and compiled independently at first use. The naive
path uses exact global dimensions and driver-chosen groups; tiling uses rounded
global dimensions and fixed 16x16 groups in both timing paths.
Compilation/context initialization are excluded from steady-state
samples. Full calls include fresh buffers, explicit input copies, NaN poisoning,
execution, completion, output copy and release. Physical shared RAM does not
grant free zero-copy buffers in this first version.

Default `both` timing gives full-call ms and a separate kernel row. OpenCL
uses the first kernel event's START through the last kernel event's END on
an in-order profiling queue, divided by batch. Copies/fill/readback are
outside that window; gaps between kernels remain inside. This differs from
CUDA's enclosing event markers, especially for tiny launches. Metadata records
the timing scope, device, driver and reported unified-memory property. CSV
keeps the existing schema; OpenCL full-call GFLOP/s is blank like CUDA's.

`nob test --cuda --opencl` includes both backends in a separate combined
build. Until register TODOs are complete, `opencl_register_tiled` and
`opencl_register_timing` should fail; earlier kernels/library checks remain independent.
`opencl_timing` checks the naive prepared path. `nob profile` is for CPU captures
and rejects either GPU opt-in flag.

### CLBlast reference

OpenCL builds include pinned CLBlast 1.7.0 by default. It uses the same Intel
GPU, context, in-order queue, float32 buffers and validator as the exercises.
SGEMM is row-major, without transposition, alpha=1 and beta=0. Its default
parameter database is used; no runtime tuning search or reduced precision is
requested. Metadata reports the version, default tuning and any
effective `CLBLAST_BUILD_OPTIONS` value. Leave user flags unset for the baseline;
changing them defines a different experiment. On Intel driver 30.0.100.9864,
the adapter adds `-D__has_builtin(x)=0` in the current process to avoid a proven
compiler crash on `llvm.assume`. This disables optional optimization hints,
preserves float32 arithmetic/default tuning, and is reported in metadata.
Other drivers keep their normal builtin detection. A trailing option separator
also works around CLBlast 1.7.0's build-option concatenation.

```powershell
.\nob.exe test --opencl --filter '^clblast'
.\nob.exe bench --opencl -- --impl opencl_tiled --m 512 --n 512 --k 512
.\nob.exe bench --opencl -- --impl clblast --m 512 --n 512 --k 512
```

Full calls pay for the same buffers, explicit copies, output poisoning,
completion and readback. CLBlast can perform several internal kernels, so
its device row uses enclosing OpenCL marker END timestamps across the whole
SGEMM batch. Internal preparation and gaps are included; initial JIT/cache
setup occurs during validation/warmup. Custom kernels retain their first-START
to last-END measurement. Compare full-call ms for host-array workloads and
keep these device timestamp differences in mind for tiny matrices.

CMake's `MATMUL_ENABLE_CLBLAST=OFF` disables the reference without disabling
OpenCL. For example, configure the existing build directory with that cache
option before the next Nob action. Source is pinned by SHA256; builds require
network access on first configuration. Dimension/element counts must fit
32-bit library kernel indexing. The optional library uses C++ internally;
your kernels remain OpenCL C.

## OpenMP thread settings

The OpenMP backend is built when CMake finds a compatible C compiler and
runtime. `nob bench -- --list` lists the available implementations. OpenMP
compile options apply only to parallel backends; the other kernels retain
their existing options.

In PowerShell, configure thread settings before testing and measuring:

```powershell
$env:OMP_NUM_THREADS = "4"
$env:OMP_DYNAMIC = "FALSE"
.\nob.exe test
.\nob.exe bench -- --impl openmp --m 512 --n 512 --k 512
```

These settings persist in that shell; change `OMP_NUM_THREADS` to compare
1, 2 and 4 threads. Record both settings with the command and timings: they
are not included in the benchmark CSV. Correct output alone does not prove
that a loop ran in parallel; CPU profiles can show worker activity.

## Combined CPU reference

`cpu_combined` is one adaptive implementation. Start reading at
[`src/cpu_combined.c`](src/cpu_combined.c), which selects between two private paths:

| Path | When used | Techniques |
| --- | --- | --- |
| Direct (`cpu_combined_direct.c`) | Small or narrow problems; also CPUs without AVX-512 | 4-by-16 AVX/FMA register tiles; cache blocking and OpenMP for enough work |
| Packed (`cpu_combined_packed.c`) | AVX-512 supported, M >= 128, N >= 32, M*N*K >= 8*1024*1024 | Packed A/B, 8-by-32 AVX-512/FMA register tiles, cache blocking and OpenMP |

Small calls skip packing and thread-team setup. Large direct calls partition
independent output tiles, which also allows column parallelism for short/wide
matrices. The packed path shares one B panel across workers, each with private
A panels and disjoint output rows. Workspace is bounded: 256 KiB for shared B
plus 32 KiB per worker for A. Packing remains inside every measured call.

CPU/OS support is checked and cached safely on first use. Without AVX/FMA,
`cpu_combined` uses the portable blocked implementation. Without OpenMP, its
optimized paths run serially. Build flags stay local to the specialized files;
the dispatcher itself uses the baseline instruction set. Tile sizes and dispatch
thresholds are heuristics, not a claim of optimal performance for every shape.

```powershell
$env:OMP_NUM_THREADS = "4"
$env:OMP_DYNAMIC = "FALSE"
.\nob.exe bench -- --impl cpu_combined --m 1024 --n 1024 --k 1024
```

The individual learning implementations remain separately benchmarkable. The
former `cpu_combined_v2` entry is now part of `cpu_combined`; use that single name
for tests, benchmarks and profiles.

## CPU technique progression

| Implementations, in benchmark order | Main idea |
| --- | --- |
| `naive` | Direct definition |
| `loop_interchange` | Contiguous inner-loop accesses |
| `blocked` | Cache tiles |
| `packed` | B-panel packing on the blocked traversal |
| `simd_sse`, `simd_avx`, `simd_avx512` | Four, eight and sixteen lanes; AVX versions include FMA |
| `register_avx` | 4-by-16 register tiles using eight-lane FMA |
| `openmp` | Independent output rows across threads |
| `cpu_combined` | Adaptively combine the techniques |
| `openblas` | External tuned library reference |

All use `nob test` and `nob bench -- --impl NAME`. Portable C can also be
vectorized by the compiler. Optional ISA flags apply only to specialized
targets, and CPU/OS checks guard their use. Packing and SIMD do not guarantee
improvements for every size; the sequence describes increasing complexity.

## OpenBLAS comparison

OpenBLAS is optional. For the Windows MSYS2 UCRT64/GCC toolchain, download its
packages into the ignored build directory once:

```powershell
.\tools\setup_openblas.ps1
.\nob.exe test
```

The script uses the existing MSYS2 package database and verifies downloads with
pacman. It extracts packages under `build/deps/openblas/` without installing or
upgrading the global toolchain. CMake discovers this local prefix for MinGW and
copies the required DLLs beside the executables. Other installations can set
the CMake cache variable `OPENBLAS_ROOT` (or the environment variable of that
name) to a compatible OpenBLAS prefix. Builds without a compatible library
continue to provide the learning kernels.

Compare separate runs with matching thread limits:

```powershell
$env:OMP_NUM_THREADS = "4"
$env:OMP_DYNAMIC = "FALSE"
$env:OPENBLAS_NUM_THREADS = "4"
$env:OPENBLAS_DEFAULT_NUM_THREADS = "4"
.\nob.exe bench -- --impl cpu_combined --m 1024 --n 1024 --k 1024
.\nob.exe bench -- --impl openblas --m 1024 --n 1024 --k 1024
```

Repeat with all thread limits set to 1, and include rectangular shapes. These
are configured limits: a library can use fewer workers for small calls. The
OpenBLAS metadata line reports its version/build settings, selected CPU core,
configured thread count and threading backend. In CSV mode that line goes to
stderr, preserving clean result stdout. Record it alongside the commands and
environment settings.

The adapter calls ordinary row-major SGEMM with `alpha=1`, `beta=0`, without
prepacking inputs or adding an outer OpenMP region. Per-call packing and
workspace costs stay inside the library call. Like other implementations,
validation and warmup occur before measured samples; this measures warmed
calls rather than first-call initialization latency. The adapter handles empty
dimensions and zero K itself, and preserves the size_t interface using the
portable baseline if dimensions exceed the library's integer ABI.

## Save measurements

```sh
mkdir -p results
./nob bench -- --csv --m 256 --n 256 --k 256 > results/baseline.csv
```

Results go to stdout and diagnostics to stderr, so redirected CSV stays clean.
In ordinary benchmark output, the workload header shares stdout with the result
rows to preserve their order. In CSV mode, that header goes to stderr.
Keep the command, source revision, and machine details with your measurements;

## Repo structure

- `src/`: kernel implementations and the registry shared by tests and benchmarks  
- `include/matmul.h`: kernel signature and mathematical/memory contract  
- `tests/test_matmul.c`: infrastructure checks and correctness fixtures for every registered kernel  
- `tools/`: benchmark runner, double reference, data generation, allocation, and timer  
- `CMakeLists.txt`: source lists, compiler flags, targets, and test registration  
- `nob.c`: build/test/benchmark/profile commands over CMake, Ninja, and CTest  

Support code in `tools/` can stay unread while you learn the kernel loops.
Adding a kernel changes the CMake source list and registry; the command runner
needs no separate build recipe. Portable kernels have no external library
dependencies; optional parallel backends use an OpenMP runtime. Nob's helper
header is vendored in `deps/nob.h`.
