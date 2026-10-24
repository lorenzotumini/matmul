# Computing topics through matrix multiplication

The main topics follow increasing implementation complexity. This ordering
organizes the concepts; it is not a ranking of performance for every workload.

| Order | Topic | Main ideas |
| --- | --- | --- |
| 01 | [Definition and representation](01-definition.md) | Shapes, indexing, ownership and correctness |
| 02 | [Loop order](02-loop-interchange.md) | Strides, locality and dependencies |
| 03 | [Compilers](03-compiler.md) | Translation, optimization and assembly |
| 04 | [Cache blocking](04-blocking.md) | Working sets and cache hierarchy |
| 05 | [Packing](05-packing.md) | Layout, temporary storage and amortized copy costs |
| 06 | [SIMD and FMA](06-simd.md) | Lanes, vector widths, fused rounding and boundaries |
| 07 | [Register tiling](07-register-tiling.md) | Microkernels, register reuse and spills |
| 08 | [CPU parallelism](08-parallelism.md) | Ownership, scheduling, races and scaling |
| 09 | [Combined CPU optimization](09-cpu-optimization.md) | Layered reuse and practical tradeoffs |
| 10 | [BLAS](10-blas.md) | Library interfaces and mature performance references |

Further topics include GPU execution and memory transfers, GPU shared-memory
tiling, GPU library comparisons, alternative algorithms and precision, and
language interoperability. Transposition is another way to explore locality.

Performance depends on algorithm, memory access, compiler and hardware.
Meaningful comparisons use the same mathematical problem, compatible accuracy
requirements and clearly defined timing boundaries.
