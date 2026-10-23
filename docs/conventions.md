# Shared conventions

## Keep experiments comparable

The starting problem is dense, contiguous, row-major, single-precision
`C = A * B`. M is the number of A/C rows, N is the number of B/C columns,
and K is the shared dimension. Document dimensions in that order.

All registered functions obey `include/matmul.h`. They overwrite C, leave A
and B unchanged, handle rectangular and empty shapes, and do not require SIMD
alignment or dimensions divisible by a vector/tile width. Inputs and output
do not overlap. Arguments are prevalidated by the caller, outside hot loops.

The initial tests use finite, moderate inputs; this is not a specification for
NaN/infinity inputs or overflowing arithmetic. Exceptional inputs and mixed
precision can become a separate numerical experiment later.

For a new CPU implementation:

1. Add a separate source file and a declaration in `include/matmul.h`.
2. Add the source to the `matmul` CMake target.
3. Add one entry to `src/registry.c`.
4. Run `nob test` and `nob test --mode release`; both exercise every entry.
5. Compare against the preserved naive baseline using the same command.

Keep a fallback or an explicit availability check for optional instruction
sets and GPU backends when introduced. A portable build should continue to
work without them. Do not globally enable AVX or CUDA in anticipation of a
later lesson. Add runtime CPU feature checks before distributing an intrinsics
binary that requires a particular instruction set.

Avoid fast-math initially. Compiler auto-vectorization is allowed: `naive`
means source code that follows the definition, not a promise of scalar machine
code. Later, inspect assembly and compare controlled compiler options.

## Correctness is separate from performance

The reference uses double products and double accumulation of the original
float inputs. We also test that reference against a hand-computed rectangular
example. The candidate uses float arithmetic and need not be bit-identical.

For each output, let `s = sum(abs(A[i,p] * B[p,j]))`, evaluated in double,
and `gamma = (K * FLT_EPSILON) / (1 - K * FLT_EPSILON)`. The current check is:

```text
abs(candidate - reference) <= 1e-6 + 2 * gamma * s
```

This conservative dot-product bound accounts for reduction length and
cancellation. NaN, infinity, and unwritten output fail. The validator rejects
`K * FLT_EPSILON >= 0.01`, where this bound becomes too loose for this lab.
It is a guardrail for these float implementations, not a general-purpose
accuracy guarantee. Revisit it explicitly for Strassen or mixed precision;
do not quietly loosen it to make a failing implementation pass.

Fixtures include random signed data from fixed seeds, zero and one matrices,
left/right identities, cancellation, empty axes, and rectangular dimensions
around likely vector and tile boundaries. Each kernel is called twice, first
with NaN-filled output and then with finite pre-existing output. Copies check
that inputs remain unchanged. Later SIMD work should add sanitizer runs or
guarded allocations to catch out-of-bounds accesses as well as wrong results.

## What a timing means

The current benchmark measures a synchronous call on preallocated host arrays.
Data generation, the double oracle, allocation by the runner, validation,
sorting and printing happen outside the timer. Any preparation performed
inside the kernel is included. Do not give one version a free transpose or
packing step while charging another for it.

The runner does an initial validation call, warmup calls, then repeated samples.
Each sample makes `--batch` calls and reports elapsed time divided by batch.
For very small matrices, increase batch (for example 1000); call overhead and
timer overhead can otherwise dominate. Both operands are reused: these are
warm-buffer experiments, with no cache flushing. Validation between samples
also touches the output/reference. Do not label these cold-memory timings.

`--profile` is a separate diagnostic workload. It requires one named
implementation, allocates only A/B/C, and skips the reference calculation and
correctness validation. After warmup it runs `repeats * batch` calls without
per-call timers or checks. A final finite-output scan and checksum do not
establish correctness; run the shared tests first. Whole-process captures
still include allocation, data generation, warmup, and cleanup; focus analysis
on the selected kernel. See [profiling.md](profiling.md).

For everyday use, `nob profile` handles the low-level flags: it builds with
symbols, tests the exact build, and validates the requested inputs with the
reference outside the trace, then starts Samply. The captured workload has
no oracle or validation between calls. Captures include resolved-symbol JSON
and an exact command note so both people and tools can inspect them.

Minimum, median and maximum show some timing variation. GFLOP/s uses the
median. The checksum makes output observable, but correctness comes from the
elementwise reference check, not from the checksum.

CSV includes compiler/version, CMake configuration, dimensions, seed, repeats,
warmup and batch. Keep experiment notes with the source revision (or a diff),
CPU/GPU, OS, full configure command/flags and power conditions. Later include
thread count and GPU/backend parameters. The CSV alone is not a complete
reproduction record. Do not commit transient build directories or timings as
universal performance claims.

For CUDA, the first wrapper should preserve the host-array contract and wait
for completion, so host timing includes transfers and synchronization. Add a
separate prepared-device experiment when learning CUDA events and buffer
reuse. Report kernel-only and end-to-end time separately; a host timer around
an asynchronous launch measures submission, not completed multiplication.

For packed CPU kernels, the same distinction applies: report one-shot cost
and reuse cost separately when a prepared representation becomes useful.
