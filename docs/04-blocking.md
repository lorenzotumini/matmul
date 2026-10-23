# 04 — Cache blocking and implementation selection

Cache blocking divides a computation into smaller regions called tiles. A tile
is a range of indices; it does not require a separate allocation. Reusing a
region of B across several output rows can reduce repeated data movement
through the memory hierarchy.

Blocking changes data reuse, not the approximate `2*M*N*K` arithmetic count.
Each reduction tile contributes another partial sum to the same output tile;
initialization happens before accumulating those contributions. Boundary tiles
can be smaller than the chosen tile size.

## Locality has a cost

Extra loops, boundary calculations and shorter inner loops add overhead.
Blocking can therefore lose on small workloads while helping larger ones.
Tile size balances reuse against overhead and the working set needed in cache.
Tiles in row-major matrices retain the original row strides; they are not
necessarily compact, contiguous blocks of memory.

For square matrices of 32-bit floats, B occupies `4*N*N` bytes and A, B and C
together occupy `12*N*N` bytes. These estimates help interpret cache capacity,
but capacity alone does not predict performance. Access order, cache placement,
sharing and other active data also matter.

## Throughput and selection

Stable GFLOP/s as dimensions grow means throughput is being sustained, not
that elapsed time stays constant. Doubling all dimensions creates eight times
the arithmetic work. A throughput drop near a working-set boundary is evidence
consistent with cache effects; it does not identify the exact bottleneck.

Different implementations can win for different workloads. A dispatcher can
choose an implementation or tile size using matrix shape and measured crossover
behavior. Such thresholds depend on hardware and compiler settings; there is
no universal input-size rule. Adaptive selection is useful without being
necessary to understand the individual techniques.
