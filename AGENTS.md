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

# Documentation

- Numbered concept documents are concise summaries of the main takeaways.
  Keep them independent of repository progress and specific implementations.
- Do not put exercises, next-implementation instructions, or completion status
  in those summaries. Keep project commands in the operational references.
- Keep lesson numbering, README links and registry/benchmark order aligned:
  definition, loop order, compilers, blocking, packing, SIMD/FMA, register
  tiling, threads, the combined CPU implementation, BLAS, then GPU execution. This is learning
  complexity, not a guarantee of monotonically improving performance.
- FMA belongs within the SIMD lesson and the AVX/AVX-512 implementations;
  do not add a separate FMA lesson or standalone FMA variant.

# GPU exercises

- Preserve the completed `src/opencl_naive.cl` baseline. `src/opencl_tiled.cl`
  is intentionally unfinished; leave its TODOs for the user. Host plumbing
  belongs in `src/opencl_support.c`. Tiling uses fixed 16x16 work-groups and
  rounded global ranges in both timing paths. Compile sources independently
  so tiled edits cannot break naive. Use `nob test --opencl --filter opencl`
  and, after correctness passes, `nob bench --opencl -- --impl opencl_tiled`.
  Missing Intel hardware may skip; incorrect/unwritten output must fail.
  Never substitute CPU/NVIDIA results. Keep concept summaries and exercise
  instructions separate.

- `src/cuda_combined.cu` is a complete combined reference requested by the
  user. Preserve the earlier learning kernels. Keep its grid selection and
  launch strategy consistent in both timing paths; its fast paths use 32x8
  thread blocks, unlike the 16x16 learning kernels.

- Preserve the completed `src/cuda_register_tiled.cu` baseline.
- `src/cuda_prefetch.cu` is a complete standalone reference requested by the
  user. It keeps the register-tiled baseline's 32x32 output tiles, 16x16 threads
  and adjacent 2x2 ownership; only the load schedule changes. Use
  `nob test --cuda --filter '^cuda_prefetch$'`. Keep its geometry identical in
  host and CUDA-event paths. Preserve the retired ownership experiment in
  `experiments/cuda_coalesced.cu`; it is outside the build/registry.

- `cublas` is an optional complete library reference, using FP32 pedantic
  SGEMM and the same full-call/device-event measurement rules. Keep handle
  initialization outside sampling and report its math mode/version.

- Preserve the completed `src/cuda_naive.cu` and `src/cuda_tiled.cu` kernels. Use
  `nob test --cuda --filter cuda_tiled` to focus on it; `--filter cuda_naive`
  checks the earlier baseline and `--filter cuda` checks all custom CUDA
  backends and prepared timing.

- Do not fill exercise TODOs merely to make tests pass. Supplied host plumbing belongs in
  `src/cuda_support.cu`.
- Use `nob test --cuda --filter cuda` for the CUDA exercise and
  `nob bench --cuda -- --impl cuda_naive` after correctness passes. On Windows,
  this uses a separate clang-cl host build and nvcc with VS 2022 Build Tools.
- CUDA tests must report real incorrect results as failures; missing hardware
  may be skipped. Never substitute a CPU result for an unfinished GPU kernel.
