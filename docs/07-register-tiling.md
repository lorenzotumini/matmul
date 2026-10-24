# 07 — Register tiling

A 4-by-16 output tile contains 64 floats. Eight 256-bit SIMD registers can
hold all its partial sums. Each reduction step loads 16 B values, reuses them
across four rows, and broadcasts four A values across the output columns.
C is stored after many reduction steps instead of after every product.
Independent accumulators also provide work for the processor while other
instructions are waiting on dependencies.

Register capacity limits the tile size: excessive accumulators can spill to
memory. Larger cache tiles can surround these small register tiles.
Boundary handling is needed at both levels.

With 512-bit vectors, an 8-by-32 output tile fits in sixteen vector accumulators.
Register tiling can be studied independently of cache tiling: keeping partial
sums in registers reduces repeated C loads/stores even without packing or
threads. Cache tiling instead organizes reuse between registers and memory.


Register tiling is separate from cache blocking: registers hold active partial
sums, while cache tiles organize which matrix regions are reused around them.
The two can be combined with packed inputs and thread ownership. Register
capacity and instruction dependencies determine useful microkernel shapes;
tile dimensions are hardware-dependent choices, not universal constants.
