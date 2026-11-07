# 14 — GPU software prefetching

A tiled multiplication repeatedly loads inputs into shared memory and then
computes with them. Software prefetching moves global loads for the next
reduction tile ahead of arithmetic on the current tile, giving the compiler
an opportunity to overlap memory latency with useful work.

The first tile primes the pipeline. During each iteration, future input
values are loaded into thread-private registers while shared memory still
holds the current tile. Once all threads finish reading, a barrier permits
replacement of shared data. Another barrier makes the replacement visible
before the next iteration consumes it. A single pair of shared input tiles
is sufficient; the additional staging storage lives in registers.

Input ownership, output ownership, tile sizes and arithmetic can stay fixed.
This makes loading order a separate experiment from larger tiles or a new
thread layout. Guard and zero-pad future loads just like current loads;
the final iteration must not fetch a nonexistent tile. Threads outside the
output boundary still participate in loading and synchronization. Conditional
block barriers require a condition that is uniform across the whole block.

Earlier loads increase the number of live register values. That can reduce
occupancy or cause spills, offsetting latency hiding. Compiler scheduling,
workload size and the existing bottleneck determine whether prefetching helps;
source order alone does not prove overlap or a speedup. Register staging
works without hardware asynchronous-copy instructions.

Compare an otherwise unchanged baseline with prefetching enabled, validate
partial tiles and repeated calls, and measure device-resident time separately
from full host-array latency. Inspect generated code or profiler data when
attributing a speedup to a particular hardware effect.

NVIDIA's [efficient GEMM guide](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
describes software pipelines and their storage tradeoffs.
