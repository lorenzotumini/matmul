# 06 — Combining CPU optimizations

CPU optimizations address different kinds of reuse and parallelism. Their
benefits can combine, but their individual speedups do not simply multiply.

| Technique | Main purpose |
| --- | --- |
| Loop order and reused intermediate values | Contiguous access and reuse of loaded inputs |
| Cache blocking | Reuse matrix regions before they leave useful cache levels |
| Register blocking | Keep several output accumulators in registers across many products |
| SIMD | Update several independent values with each vector instruction |
| Threads | Assign independent output regions to different CPU cores |

## Register tiles inside cache tiles

A 4-by-16 output tile contains 64 floats. Eight 256-bit SIMD registers can
hold all its partial sums. Each reduction step loads 16 B values, reuses them
across four rows, and broadcasts four A values across the output columns.
C is stored after many reduction steps instead of after every product.
Independent accumulators also provide work for the processor while other
instructions are waiting on dependencies.

Register capacity limits the tile size: excessive accumulators can spill to
memory. Larger cache tiles surround these small register tiles. Assigning each
output tile to one thread avoids competing updates to the same result.
Boundary handling is needed at both levels.

Small workloads may benefit from skipping thread coordination and cache
partitioning. Matrix shape, hardware and compiler behavior affect crossover
points; simple measured heuristics are useful without being universal rules.

## Other meaningful techniques

- **Packing:** copy input panels into a layout tailored to the microkernel,
  improving access regularity and addressing overhead. Copying, workspace and
  preparation costs belong in one-shot timings.
- **Fused multiply-add:** combine multiplication and addition, with one final
  rounding. This needs separate feature support and numerical validation.
- **Hardware tuning:** tile dimensions, thread placement, memory placement on
  NUMA systems, and instruction scheduling can affect throughput.
- **Tuned BLAS libraries:** provide a useful comparison against mature matrix
  multiplication implementations, with matching precision and thread counts.

The layered cache-and-microkernel design is described in
[Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~flame/books/TOMS.pdf)
and the [BLIS microkernel documentation](https://github.com/flame/blis/blob/master/docs/KernelsHowTo.md).
