# 12 — GPU shared-memory tiling

A block can cooperate on an output tile while each thread keeps its own
output accumulator. Splitting the reduction into chunks lets the block load
small A and B tiles into shared memory and reuse their values across threads.
Shared memory is explicitly managed and belongs to one block.

For square tiles of width T, a straightforward arrangement loads T squared
values from each input for a chunk and performs T cubed multiply-add steps.
Each loaded A value contributes across output columns; each loaded B value
contributes across output rows. These are logical access counts: hardware
caches can already satisfy some repeated requests in a direct kernel.

The cooperative sequence is **load, synchronize, compute, synchronize**.
The first barrier makes all loaded slots available before other threads read
them. The second prevents early finishers from overwriting a tile still in use.
`__syncthreads()` coordinates threads within a block, not across blocks.

Partial tiles require separate input bounds checks and zero padding. A thread
with no valid output may still load data needed by its neighbors. Threads must
therefore participate in block-wide synchronization even when their final
output store is suppressed. Conditional barriers require a condition uniform
across the block; per-output validity is generally not such a condition.

Larger tiles can increase reuse but also consume more threads and shared
memory, potentially reducing the number of blocks that can execute together.
Synchronization and extra work on padded tiles also cost time. Tiling is a
tradeoff to measure across sizes and shapes, not a universal speedup.

See the [CUDA programming guide](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html)
for shared memory and synchronization.
