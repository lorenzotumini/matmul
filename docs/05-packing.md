# 05 — Packing matrix panels

Packing copies a matrix region into a layout chosen for its consumer. Cache
blocking chooses which indices to visit; packing also changes where those
values physically sit in temporary storage. It preserves the logical product.

A simple B-panel copy makes rows of a cache tile adjacent, removing the gaps
caused by the original matrix's row stride. Predictable access can improve
locality and simplify addressing. More elaborate packed layouts are designed
around the computation that consumes them.

## Reuse must repay the copy

Packing adds reads, writes and workspace. It can lose for small workloads or
when each packed region is consumed too few times. Its preparation costs belong
inside one-call timings. Reusing already-packed inputs is a separate workload
with different ownership and timing requirements.

Workspace can be bounded to a panel rather than an entire matrix. Boundary
panels still need correct dimensions; compact storage must not change the
logical indexing of the original multiplication.
