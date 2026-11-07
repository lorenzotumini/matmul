# Archived experiments

`cuda_coalesced.cu` preserves the completed output-ownership remapping
experiment. It uses the same 32x32 output tiles and 16x16 thread blocks as
`src/cuda_register_tiled.cu`, with separated per-thread output coordinates.
It did not establish a consistent speedup on the MX330.

This source is excluded from CMake, the registry, tests and benchmarks.
Its private header is in `src/`. Restoring it as a runnable backend also
requires its host wrapper, grid mapping and prepared-timing registration.
See [GPU memory access layout](../docs/gpu-memory-layout.md) for the concepts.
