# Learning workflow

This is a hands-on learning project. The user wants to implement the early
matrix multiplication kernels manually to understand the basics.

- When helping with an exercise, default to explanations, small hints, debugging
  and code review. Supply a complete kernel only when the user asks for it.
- Infrastructure (builds, tests, benchmarks) can be implemented and maintained.
- The user has completed `src/naive.c`; preserve that baseline. Future exercise
  placeholders should not be silently solved just to make the suite green.
- Preserve existing implementations as baselines. Put new techniques in their
  own files and register them for the same tests and benchmark.
- Read `docs/conventions.md` before changing the contract or measurement rules.
- Use `nob test`, `nob bench` and `nob profile` for the everyday workflow.
  Nob delegates to CMake/CTest; keep source lists and build flags in CMake.
- `nob profile` captures with Samply after testing and validating the requested
  workload. On Windows it defaults to clang-cl/PDB symbols; ordinary actions
  default to GCC. Avoid putting builds or the double oracle inside captures.
- Nob hides successful setup output in stage logs by default. Preserve clean
  result stdout; show diagnostics on failure and support `--verbose` for detail.
