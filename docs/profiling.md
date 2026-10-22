# Minimal profiling workflow

The first useful loop is: implement, test, benchmark, profile, change one thing,
then test and benchmark again. Start with the first kernel. No profiler is
needed to complete that exercise.

## 1. Measure speed with the benchmark

Use the Release build and commands in the README. Keep a CSV result and the
exact command. Profilers add overhead; their execution times should not replace
ordinary benchmark measurements.

## 2. Build optimized code with symbols

CMake's `RelWithDebInfo` configuration keeps optimization and debug information.
The GNU-style build also keeps frame pointers in this configuration to help
stack unwinding. Use a separate build directory:

```powershell
.\nob.exe build --mode profile
.\nob.exe test --mode profile
```

This is useful for debugging and assembly inspection. For Windows profilers,
use PDB symbols from MSVC or clang-cl. The convolution project already used
clang-cl with Samply. In an **x64 Visual Studio Developer terminal**, build:

```powershell
.\nob.exe build --cc clang-cl --mode profile
.\nob.exe test --cc clang-cl --mode profile
```

Keep the executable, PDB files, source, and build configuration together with
the capture. If the profile displays only addresses, fix symbol resolution
before drawing conclusions. `RelWithDebInfo` may use different optimization
flags from Release; compare performance using the same build configuration.

## 3. Capture a focused workload with Samply

Once the tests pass, first try the workload directly:

```powershell
.\nob.exe profile --cc clang-cl -- --impl naive --m 256 --n 256 --k 256 --repeats 200 --batch 10
```

Increase/decrease the repeat count so the multiplication lasts a few seconds.
Then capture it using the already installed [Samply](https://github.com/mstange/samply):

```powershell
New-Item -ItemType Directory -Force results | Out-Null
samply record --save-only -o results\naive.json.gz -- .\build\clang-cl\profile\matmul_bench.exe --profile --impl naive --m 256 --n 256 --k 256 --repeats 200 --batch 10
samply load results\naive.json.gz
```

Capture the built executable directly, rather than wrapping `nob profile` in
Samply: otherwise the capture would include configuring and compiling too.
`nob profile` runs the workload; it does not launch a profiler or collect counters.

Samply on Windows may need an elevated terminal for ETW recording. The saved
profile can be opened again later, and its JSON is available for programmatic
inspection. Keep the matching binaries and symbols so both of us can resolve
sampled addresses. A capture is useful only after an actual kernel exists;
profiling the unfinished placeholder cannot explain multiplication performance.

`--profile` runs only one named implementation. It allocates A/B/C, generates
inputs and warms up, then repeats the kernel without the double oracle,
per-call timers, validation or printing in the loop. The final checksum makes
output observable; it is not a correctness test. Nonfinite output fails, but
finite wrong answers still require the shared tests to catch them.

The profiler still captures whole-process setup and cleanup. Select
`matmul_naive` in the call tree and inspect its callers/source. Check whether
time is in the multiplication or an unexpected helper. CPU samples show where
execution spends time, not whether cache misses or compute throughput caused it.

## 4. Use VTune when the question needs hardware counters

For cache behavior, bandwidth, memory stalls and instruction execution,
[Intel VTune](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2024-0/running-command-line-analysis.html)
is the next tool to evaluate. It was not on this shell's PATH during initial
inspection. Installation and hardware-counter availability must be checked
before assuming a particular analysis works.

Once available, its basic CLI workflow is:

```powershell
vtune -collect hotspots -result-dir results\vtune-naive -- .\build\clang-cl\profile\matmul_bench.exe --profile --impl naive --m 256 --n 256 --k 256 --repeats 200 --batch 10
vtune -report summary -result-dir results\vtune-naive -format csv -report-output results\vtune-naive-summary.csv
```

Use a fresh result directory for each experiment. Later try Memory Access or
Microarchitecture Exploration when supported. Analyze the kernel region,
rather than attributing process startup or random-data generation to it.

For now, write down a specific question: "Did loop interchange reduce cache
misses?" or "Is memory bandwidth saturated?" High CPU usage alone cannot answer
either question. Requested allocation bytes, resident memory, and bytes moved
through caches/RAM are different measurements.

## Keep just enough evidence

For an experiment, save the benchmark CSV, profile, matching binaries/symbols,
and a small text note containing:

- The commands and source revision (or the changed source if no Git revision).
- Compiler/build/flags, matrix dimensions, seed, warmup, repeats and batch.
- The question, observation, and next change to test.

The existing `results/` ignore rule keeps these local. There is no custom
dashboard, allocation tracker, cache simulator, or GPU integration at this
stage. Add one when an implementation raises a question the current workflow
cannot answer. GPU compatibility notes remain in `roadmap.md`.
