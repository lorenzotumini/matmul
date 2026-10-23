# Computing topics through matrix multiplication

Matrix multiplication connects several layers of computing. This map groups
the concepts; it does not prescribe an implementation sequence.

| Topic | Main ideas |
| --- | --- |
| [Definition and representation](01-definition.md) | Shapes, indexing, pointers, ownership, correctness and arithmetic cost |
| [Loop order](02-loop-interchange.md) | Strides, locality, cache lines, reuse and dependencies |
| [Compilers](03-compiler.md) | Translation, optimization, alias analysis, vectorization and assembly |
| Transposition | Preparation cost, extra storage and contiguous dot products |
| Cache blocking and packing | Working sets, cache hierarchy, data layout and boundary handling |
| SIMD | Lanes, registers, alignment, fused operations and remainder handling |
| CPU parallelism | Work decomposition, races, scheduling, false sharing and scaling |
| BLAS | Library interfaces, mature implementations and comparison baselines |
| GPU computing | Thread grids, device memory, transfers, synchronization and launch overhead |
| GPU tiling | Coalescing, shared memory, barriers, occupancy and reuse |
| GPU libraries | Layout conventions, device-data reuse and timing boundaries |
| Algorithms and precision | Arithmetic complexity, crossover sizes, temporary memory and numerical stability |
| Language comparisons | Representation, compiler behavior, interoperability and backend costs |

Performance depends on the interaction between algorithm, memory access,
compiler and hardware. Meaningful comparisons require the same mathematical
problem, compatible accuracy requirements and clearly defined timing boundaries.
