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

The benchmark follows this learning order, with the combined implementation and
BLAS last. This is an order of complexity; matrix shape and hardware still
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
