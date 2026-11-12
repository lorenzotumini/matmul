# 20 — Combining OpenCL optimizations

Work-group tiling and private register tiles supply reuse at different levels.
Mapping neighboring work-items across columns can make local B reads and
output writes contiguous, while a shared A value feeds several independent
accumulators. Output ownership can be strided without changing the product.

Software prefetching loads future inputs into registers before current-tile
arithmetic. All readers must finish before local memory is replaced, followed
by a barrier before consumption. Boundary work-items still participate and
missing inputs contribute zero. A single pair of local arrays is sufficient.

Larger tiles trade reuse against private/local storage, occupancy and boundary
waste. Earlier loads also increase register pressure. Test tile geometry and
prefetch independently; neither guarantees a gain. Compiler scheduling and
GPU architecture influence whether the source exposes useful latency hiding.

A small offline sweep can guide fixed dimension-based selection without a
runtime search. Keep simpler paths for workloads that cannot amortize larger
tiles. Validate each path and its thresholds against one numerical contract.
Compare both device-resident work and full host-array latency with a library,
recording the different timestamp boundaries and local power/thermal conditions.
