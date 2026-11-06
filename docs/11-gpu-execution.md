# 11 — GPU execution and memory

A GPU kernel is a function executed by many threads. Threads are grouped
into blocks, and blocks form a grid. Each thread derives its work from its
block's position and its position inside that block.

Assigning one matrix output to each thread turns the row and column loops
into parallel work. Each thread still performs its own reduction over K.
Rounded-up grids require bounds checks for incomplete boundary blocks.

Threads that own separate output elements need no atomics or barriers for
a direct multiplication: they read shared inputs and write disjoint outputs.
CUDA executes threads in groups called warps; neighboring addresses accessed
by neighboring lanes can improve memory access efficiency. SIMT describes
the GPU thread execution model, while CPU SIMD exposes vector operations.

Host and device pointers describe different memory locations in a basic
discrete-GPU program. Allocating device buffers, copying inputs, executing a
kernel and copying results back are distinct operations. A kernel launch is
asynchronous: returning from the launch does not mean the result is ready.
Check launch errors and completion errors separately.

End-to-end latency includes preparation, transfers and synchronization.
Device events can measure kernel time separately; they do not substitute for
a host timer around the complete operation. Workloads reusing device-resident
data answer a different performance question from one-shot host-array calls.

More threads alone do not guarantee high throughput. Repeated input loads,
memory access patterns, limited parallel work and launch overhead can dominate.
Shared-memory tiling introduces cooperation within blocks to reuse inputs.

See the [CUDA programming guide](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html)
for the execution and memory models.
