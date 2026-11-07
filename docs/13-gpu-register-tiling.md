# 13 — GPU register tiling

A thread can own several outputs and maintain a separate accumulator for
each. Shared memory supplies block-wide input reuse; scalar values and
accumulators in registers add reuse within each thread.

For a 2-by-2 output group, one reduction step combines two A values with two
B values to update four sums. Each loaded value participates in two products.
This reduces logical shared-memory loads compared with four independent
one-output threads. Actual traffic and throughput also depend on broadcasts,
bank conflicts, compiler-generated instructions and the surrounding layout.

Thread-block dimensions describe threads, not output dimensions. If each
thread computes multiple outputs, the output region can grow without adding
threads. Cooperative loading must cover the entire shared working set, even
when threads need several loads to populate it. Loading ownership need not
match output ownership.

Larger output regions offer more block-level reuse but also change grid size,
shared-memory use and the amount of work per thread. More live accumulators
consume registers and can reduce the number of active warps or blocks.
Excessive register pressure can cause spills into device-backed local memory.
Scalar source variables do not guarantee a particular register allocation.

Partial outputs need independent bounds checks for each stored element.
Input padding and block synchronization still apply to all participating
threads. Throughput improvements must be measured across shapes and sizes;
more per-thread work is not automatically faster.

See the [CUDA programming guide](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html)
for the register, shared-memory and thread hierarchy models.
