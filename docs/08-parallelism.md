# 08 — CPU parallelism

Threads let multiple CPU cores compute independent parts of a problem. Giving
each worker complete output rows or tiles avoids concurrent updates to the same
result. A and B can be shared read-only; local accumulators belong to workers.
Splitting a single output reduction across threads instead requires combining
partial sums and can change floating-point evaluation order.

OpenMP expresses work distribution through directives and manages thread teams,
scheduling and synchronization. Manual threading makes worker creation, work
assignment and completion explicit. Both require correct data ownership;
manual control alone does not guarantee better performance.

Static scheduling suits similarly sized row computations. Small workloads can
lose to coordination overhead. Larger ones eventually encounter limits from
shared caches, memory bandwidth or other hardware resources. Extra logical
threads do not provide the same resources as extra physical cores.

```text
speedup(T)    = time(1 thread) / time(T threads)
efficiency(T) = speedup(T) / T
```

Configured thread counts are limits rather than proof of actual worker usage.
Meaningful scaling comparisons hold the algorithm and workload constant.
Completion synchronization ensures all outputs are ready before returning to
the caller.
