# 09 — Combining CPU optimizations

CPU optimizations address different kinds of reuse and parallelism. Their
benefits can combine, but their individual speedups do not simply multiply.

| Technique | Main purpose |
| --- | --- |
| Loop order and reused intermediate values | Contiguous access and reuse of loaded inputs |
| Cache blocking | Reuse matrix regions before they leave useful cache levels |
| Register blocking | Keep several output accumulators in registers across many products |
| Packing | Arrange inputs for regular microkernel access |
| SIMD/FMA | Update several values with fused multiply-add instructions |
| Threads | Assign independent output regions to different CPU cores |

## Layers of reuse

A small register tile keeps active partial sums near the arithmetic units.
Larger cache tiles organize reuse around those microkernels. Packed panels
supply a convenient layout, and each worker owns a disjoint output region.
Synchronization protects any packed panels shared across workers.

## Packed micro-panels

A microkernel-oriented layout puts the values needed for one reduction step
next to one another:

```text
packed B: [column micro-panel][reduction index][column lane]
packed A: [row micro-panel]   [reduction index][row lane]
```

Padding incomplete panels with zeros keeps the arithmetic regular. Masked
stores or boundary loops still protect output bounds. Sharing packed B across
threads avoids redundant copies; workers can keep private A panels. Barriers
are needed before consuming shared data and before overwriting a panel that
other workers may still read.

## Choosing how much machinery to use

A combined algorithm can choose a strategy for each call. Small workloads may
benefit from register tiles directly over the original arrays, avoiding copying,
large workspaces and thread coordination. Larger workloads can amortize packing
and share cached panels across many register tiles and workers.

Shape matters as well as operation count. A short, wide output may offer plenty
of column tiles but too few row tiles to occupy a team that splits only rows.
Tiny reduction dimensions provide little arithmetic per output store. No single
size threshold captures every such case.

CPU/OS support determines which instructions are safe; workload size determines
whether their surrounding strategy is worthwhile. Simple measured rules keep
that decision understandable. Compare small, large, rectangular and awkward-tail
shapes, including preparation costs and matching thread limits. Thresholds and
tile sizes remain hardware-dependent heuristics.

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
