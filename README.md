# Matrix multiplication lab

A hands-on computing project: start with the definition of matrix multiplication,
then investigate how algorithms, memory, compilers, CPU cores, SIMD and GPUs
affect the same computation.

The first naive kernel is implemented. Further techniques are hands-on
experiments, using shared correctness tests and a benchmark.
The kernels have no third-party library dependencies in this first stage.

## Start here

1. Read [the first lesson](docs/01-definition.md).
2. Read the short contract in [include/matmul.h](include/matmul.h).
3. Compare [src/naive.c](src/naive.c) with the definition.
4. Run the tests, then measure your implementation.

The reference implementation in `tools/lab.c` is support code. You can leave it
unread until you have written your own loops if you want to avoid spoilers.

## Everyday commands

From the project directory in PowerShell, bootstrap Nob once:

```powershell
gcc nob.c -o nob.exe
```

Then use:

```powershell
.\nob.exe build
.\nob.exe test
.\nob.exe test --filter kernels -- -V
.\nob.exe test --mode release
.\nob.exe bench -- --impl naive --m 256 --n 256 --k 256
.\nob.exe profile -- --impl naive
.\nob.exe bench -- --list
.\nob.exe --help
```

Every command builds first. `test` stops if the build or any selected test
fails, so it cannot silently test an old executable. All registered tests should pass
for the current baseline. `harness` checks the infrastructure independently
of your kernel; checks remain enabled in optimized builds.

| Command | Default mode | Purpose |
| --- | --- | --- |
| `build` (also the default with no arguments) | `debug` | Compile |
| `test` | `debug` | Compile and run CTest |
| `bench` (`benchmark` also accepted) | `release` | Compile and run validated timings |
| `profile` | `profile` | Compile, test, validate the workload, and record with Samply |

Use `--cc gcc`, `--cc clang`, `--cc msvc` or `--cc clang-cl`, and
`--mode debug`, `--mode release` or `--mode profile`. `profile` maps to CMake's
optimized `RelWithDebInfo` configuration. `--jobs N` / `-j N` controls build
parallelism. Like Toy, outputs live under `build/<compiler>/<mode>/`.
The default compiler is GCC, except `profile` on Windows defaults to clang-cl
for PDB symbols. Windows captures accept clang-cl or MSVC. If toolchain
discovery fails, use an x64 Visual Studio Developer terminal.

Arguments after `--` go directly to CTest or the benchmark. Nob options belong
before it. `--filter` matches CTest names using a regular expression; a filter
matching no tests fails. Get benchmark options with `nob bench -- --help`.

Successful compilation is quiet by default, with a short build status followed
by a separate results section. Full configure/build output is saved under
`build/<compiler>/<mode>/logs/` (the latest run of each stage). Failures print
their diagnostics automatically. Use `--verbose` before `--` for commands and
full logs, for example `nob bench --verbose -- --impl naive`.

On Linux/macOS, bootstrap with `cc nob.c -o nob`, then use `./nob` with the same
arguments (select an installed compiler with `--cc` if needed). The Windows
GCC and clang-cl paths are locally verified.

To save measurements:

```powershell
New-Item -ItemType Directory -Force results | Out-Null
.\nob.exe bench -- --csv --m 256 --n 256 --k 256 > results\baseline.csv
```

The benchmark validates output before timing and after each sample. Incorrect
output produces a failing exit status and no performance row for that kernel.
It reports minimum, median, maximum, and conventional `2*M*N*K / time` GFLOP/s.
Use `--help` for seeds, warmups and batching tiny problems.

## How the build works

Nob is a small C command layer over **CMake + Ninja + CTest**, which must be
installed. It vendors [nob.h](https://github.com/tsoding/nob.h) and rebuilds
itself when `nob.c` or that header changes. It does not download dependencies.
Status messages and diagnostic logs go to stderr, preserving benchmark CSV on
stdout.

CMake remains the single place for source lists, compiler flags and future
OpenMP/CUDA dependencies. Adding a kernel does not require a second build
recipe in Nob. This keeps the commands short without duplicating build rules.
You can still call the backend directly:

```powershell
cmake -S . -B build/gcc/debug -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Debug
cmake --build build/gcc/debug
ctest --test-dir build/gcc/debug --output-on-failure
```

Previous `build/debug`, `build/release` and `build/profile-clang` directories
are separate from the new layout; Nob does not modify or remove them.

## Neovim / language server

The root `compile_flags.txt` supplies C17 and the `include/` and `tools/` paths
for clangd, without requiring a build first. It also supplies editor-only
values for the benchmark's build metadata macros. Paths are relative to the
repository root. Restart your LSP if it was already running when this file
was added.

For exact flags for a particular compiler/mode, use the generated
`build/<compiler>/<mode>/compile_commands.json` instead. See
[clangd's project setup](https://clangd.llvm.org/installation.html#project-setup).

## Profiling

Record a CPU profile with:

```powershell
.\nob.exe profile -- --impl naive
```

Nob builds optimized code with symbols, runs tests and validates the requested
workload, then records only the workload process with Samply. It defaults to
200 measured calls and saves a unique capture under `results/` with a symbol
sidecar and command note. Existing captures are protected from overwrite.

Use `--open` to open the interactive viewer after capture, or `--output FILE`
to name the capture. These are Nob options and belong before `--`:

```powershell
.\nob.exe profile --open -- --impl naive --repeats 500
.\nob.exe profile --output results\naive-baseline.json.gz -- --impl naive
```

Samply must be on PATH. See [the profiling guide](docs/profiling.md) for symbols,
viewing saved captures and interpreting results. The benchmark measures speed,
Samply locates hotspots, and VTune is the later step for hardware counters.

## Project layout

| Location | Purpose |
| --- | --- |
| `nob.c` | Short build/test/benchmark/profile commands |
| `CMakeLists.txt` | Build targets, flags, dependencies and CTest registration |
| `src/naive.c` | Your first exercise: literal definition |
| `src/registry.c` | List of implementations used by both runners |
| `include/matmul.h` | Shared mathematical and memory contract |
| `tests/test_matmul.c` | Known patterns, random inputs, edge cases, repeated calls |
| `tools/lab.c` | Allocation checks, deterministic data, double reference, timer |
| `tools/bench.c` | Command line, validation and repeated measurements |
| `docs/01-definition.md` | First lesson, with questions and a worked example |
| `docs/roadmap.md` | Suggested experiments, from C to CUDA |
| `docs/conventions.md` | Adding implementations, numerical checks and timing rules |
| `docs/profiling.md` | Minimal profiling workflow and saved results |

Only `naive` is registered today. Later stages are exercises in the roadmap,
not implemented features.
