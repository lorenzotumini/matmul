Learn matrix multiplication by implementing one technique at a time, testing it,
and measuring it against earlier versions.

## Workflow

1. Read [the kernel contract](include/matmul.h).
2. Use the concept summaries below as a reference.
3. Run the commands below to check correctness and take a measurement.
4. Follow [the conventions](docs/conventions.md) when adding an implementation.

## Concept summaries

| Summary | Focus |
| --- | --- |
| [01 — Matrix multiplication basics](docs/01-definition.md) | Shapes, indexing, correctness and arithmetic cost |
| [02 — Loop order and memory access](docs/02-loop-interchange.md) | Locality, reuse, initialization and performance interpretation |
| [03 — Compilers and generated code](docs/03-compiler.md) | Optimization, dependencies, SIMD and evidence |
| [04 — Cache blocking and implementation selection](docs/04-blocking.md) | Working sets, tiling overhead, sustained throughput and adaptive selection |

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
needs no separate build recipe. The kernels have no external library dependencies;
Nob's helper header is vendored in `deps/nob.h`.
