# Profiling

Use profiling to locate CPU hotspots. Use `nob bench` for speed comparisons;
profiler overhead and build settings can change timings.

## Requirements

[Samply](https://github.com/mstange/samply) must be on PATH.
On Windows, `nob profile` defaults to clang-cl for PDB symbols and requires a
working Visual Studio C toolchain and Windows SDK. If discovery fails, use an
x64 Visual Studio Developer terminal. MSVC is also supported through `--cc msvc`;
MinGW GCC is rejected for captures because kernel symbols did not resolve in
local testing. On other platforms the default compiler is GCC.

Windows ETW recording may require an elevated terminal. Capture failures show
Samply's diagnostics and return a failing status.

## Capture and inspect

From the repository root:

```powershell
.\nob.exe profile -- --impl naive
```

Nob performs the following steps:

1. Builds optimized code with symbols (`RelWithDebInfo`).
2. Runs the shared tests on that build.
3. Validates the requested implementation, dimensions, and seed against the
   double reference, outside the capture.
4. Records the repeated kernel workload with Samply.

The defaults are `naive`, 256-by-256 matrices, two warmup calls, and 200 workload
calls. Increase `--repeats` for faster kernels so the workload runs for a few
seconds. To capture and open the viewer:

```powershell
.\nob.exe profile --open -- --impl naive --repeats 500
```

The viewer server runs until Ctrl+C. Without `--open`, capture saves and exits.
To choose a filename, put `--output` before `--`:

```powershell
.\nob.exe profile --output results\naive-baseline.json.gz -- --impl naive
```

To reopen that capture:

```powershell
samply load results\naive-baseline.json.gz
```

The capture includes allocation, input generation, warmup, and cleanup. Focus on
the selected kernel, such as `matmul_naive`. During the repeated calls, the runner
has no oracle, per-call timers, or validation; a final finite-output check and
checksum keep the result observable. They do not establish correctness.
`--profile` and `--validate-only` in the benchmark executable are internal parts
of this workflow; use `nob profile` to run it.

## Saved files

The default path is `results/profile-<timestamp>-<id>.json.gz`. Nob prints the
path when finished and protects existing captures and sidecars from overwrite.
Create the parent directory first for custom paths outside `results/`.

Keep each useful capture with its companion files:

| File | Purpose |
| --- | --- |
| `*.json.gz` | Recorded profile |
| `*.json.syms.json` | Resolved symbols, produced through Samply's `--unstable-presymbolicate` option |
| `*.json.gz.command.txt` | Working directory and exact capture command |
| `*.json.gz.stdout.log`, `*.json.gz.stderr.log` | Profiler/workload output for captures without `--open` |

Setup logs live in `build/<compiler>/profile/logs/`. Failures display their logs;
use `--verbose` before `--` for full commands and output.
Preserve the matching executable/PDB and source revision for long-term source
analysis: resolved symbols do not snapshot source files. Remove disposable
captures together with their companions when they are no longer useful.

CPU samples show where time is spent. They do not directly measure cache misses,
memory bandwidth, or execution stalls; investigate hardware counters when a
specific experiment needs them.
