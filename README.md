# Matrix multiplication lab

Learn matrix multiplication by implementing one technique at a time, testing it,
and measuring it against earlier versions. The direct C implementation,
[`src/naive.c`](src/naive.c), is complete and stays as the baseline.

## Start here

1. Read [the first lesson](docs/01-definition.md) for the math and array layout.
2. Read [the kernel contract](include/matmul.h), then compare the baseline with
   the definition.
3. Run the commands below to check correctness and take a first measurement.
4. Choose the next experiment from [the roadmap](docs/roadmap.md). Follow
   [the conventions](docs/conventions.md) when adding an implementation.

## Build, test, measure

Install GCC, CMake 3.21 or newer, and Ninja, with each on PATH. From the repository
root in PowerShell, compile the command runner once:

```powershell
gcc nob.c -o nob.exe
```

Use Nob for the everyday workflow:

```powershell
.\nob.exe test
.\nob.exe bench -- --impl naive --m 256 --n 256 --k 256
.\nob.exe profile -- --impl naive
```

- `test` builds in Debug and runs the shared correctness tests.
- `bench` builds in Release, validates output, and reports timings and GFLOP/s.
- `profile` records CPU hotspots with Samply. It needs additional tooling;
  follow [the profiling guide](docs/profiling.md) when ready.

Every action builds first. Nob rebuilds itself when its source changes.
Arguments before `--` configure Nob; arguments after it configure the test or
benchmark runner. Use `.\nob.exe --help` for build options and
`.\nob.exe bench -- --help` for workload options.

Builds live in `build/<compiler>/<mode>/`. Successful setup output goes to
`logs/` inside that directory; failures show diagnostics automatically.
Add `--verbose` before `--` to see full commands and logs. Ordinary actions
default to GCC; compiler and build-mode overrides are available in Nob's help.

On Linux/macOS, bootstrap with `cc nob.c -o nob` and use `./nob` for these commands.

## Save measurements

```powershell
New-Item -ItemType Directory -Force results | Out-Null
.\nob.exe bench -- --csv --m 256 --n 256 --k 256 > results\baseline.csv
```

Results go to stdout and diagnostics to stderr, so redirected CSV stays clean.
Keep the command, source revision, and machine details with your measurements;
see [the measurement rules](docs/conventions.md#what-a-timing-means).
`build/` and `results/` are local, ignored artifacts.

## Where things belong

| Location | Responsibility |
| --- | --- |
| `src/` | Kernel implementations and the registry shared by tests and benchmarks |
| `include/matmul.h` | Kernel signature and mathematical/memory contract |
| `tests/test_matmul.c` | Infrastructure checks and correctness fixtures for every registered kernel |
| `tools/` | Benchmark runner, double reference, data generation, allocation, and timer |
| `CMakeLists.txt` | Source lists, compiler flags, targets, and test registration |
| `nob.c` | Build/test/benchmark/profile commands over CMake, Ninja, and CTest |

Support code in `tools/` can stay unread while you learn the kernel loops.
Adding a kernel changes the CMake source list and registry; the command runner
needs no separate build recipe. The kernels have no external library dependencies;
Nob's helper header is vendored in `deps/nob.h`.

For editing, `compile_flags.txt` gives clangd the C17 setting, include paths, and
editor-only build metadata without requiring a build first.
