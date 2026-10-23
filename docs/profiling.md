# Profiling through Nob

Use three everyday commands:

```powershell
.\nob.exe test
.\nob.exe bench -- --impl all --m 256 --n 256 --k 256
.\nob.exe profile -- --impl naive
```

`test` checks correctness. `bench` measures speed. `profile` records CPU samples
that explain where execution spends time. Every command builds first.

## Capture a profile

`nob profile` now performs the complete capture workflow:

1. Builds optimized code with debug symbols (`RelWithDebInfo`).
2. Runs the shared tests on that exact build.
3. Validates the selected implementation against the double reference at the
   requested dimensions and seed, once, before recording.
4. Runs Samply on the repeated multiplication process.
5. Saves the profile, resolved-symbol sidecar, and exact capture command.

Compilation, tests and the reference calculation are outside the capture.
The captured process still includes allocation, input generation, warmup and
cleanup. Select the kernel (for example `matmul_naive`) when examining samples.

Defaults are the `naive` implementation, 256-by-256 matrices, 200 measured calls,
and the `profile` build mode. Increase repeats for faster implementations so
the workload runs for a few seconds:

```powershell
.\nob.exe profile -- --impl naive --m 256 --n 256 --k 256 --repeats 500
```

The command saves a unique `results/profile-<timestamp>-<id>.json.gz` file by
default. It prints the path when finished. To choose a name:

```powershell
.\nob.exe profile --output results\naive-baseline.json.gz -- --impl naive
```

Existing capture files or sidecars are protected from overwrite. For custom
output directories other than `results/`, create the parent directory first.

The normal console shows short build/check statuses and the saved profile path.
Successful setup output is stored in `build/<compiler>/profile/logs/`; a
save-only capture's raw output is stored in `.stdout.log` and `.stderr.log`
files beside the profile. Failures display their logs automatically. Add
`--verbose` before `--` to see the commands and all logged output. With
`--open`, Samply's viewer output stays live so its server instructions are visible.

## Open the interactive viewer

Add `--open` to capture and then open Samply's interactive call tree, flame
graph, timeline and source view:

```powershell
.\nob.exe profile --open -- --impl naive
```

Samply keeps its local viewer server running until stopped with Ctrl+C. The
capture is saved on disk. Without `--open`, recording finishes and exits,
which is convenient for automated experiments.

To reopen a saved capture later, use the path printed by Nob:

```powershell
samply load results\naive-baseline.json.gz
```

## Requirements and symbols

[Samply](https://github.com/mstange/samply) must be on PATH. On this Windows
machine, an actual capture was verified with Samply 0.13.1 and clang-cl/PDB
symbols. Captures resolved `matmul_naive` and its source file.

On Windows, `nob profile` defaults to **clang-cl**; other actions default to
GCC. A working Visual Studio C toolchain and Windows SDK are needed. If compiler
discovery fails, run from an x64 Visual Studio Developer terminal. `--cc msvc`
is also accepted for Windows captures. MinGW GCC captures were tested but did
not resolve kernel symbols, so the profile command rejects that combination.
On other platforms the default remains GCC.

Windows ETW recording may require an elevated terminal. If capture fails, Nob
returns failure and preserves the profiler's error message; it does not treat
an ordinary workload run as a successful capture.

The resolved-symbol sidecar uses Samply's `--unstable-presymbolicate` option.
Keep it beside the profile, along with the `.command.txt` note. Preserve the
matching executable/PDB and source revision for long-term source-level analysis;
resolved symbols do not snapshot source files. Both the compressed profile and
symbol JSON can be inspected programmatically.

## What profiling tells us

Start with a question: is time in the kernel, allocation, a helper, or a wait?
CPU samples locate hotspots. They do not by themselves measure cache misses,
memory bandwidth, or prove that the kernel is compute-bound.

Use ordinary `nob bench` results for speed comparisons; profiler overhead changes
execution time. Profile builds may use different optimization flags from Release.
Always compare implementations using the same compiler and configuration.

Intel VTune remains the next tool for cache, bandwidth and execution-stall
questions when its hardware-counter support has been checked. No custom
dashboard, simulator or GPU profiling integration is needed at this stage.

The underlying `matmul_bench --profile` and `--validate-only` flags are internal
parts of this workflow. You do not need to invoke the executable directly or
wrap Nob in Samply. Build-only compiler experiments can still use
`nob build --mode profile`; only the **profile command** records samples.
