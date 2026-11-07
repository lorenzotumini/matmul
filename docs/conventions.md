# Shared conventions

## Keep experiments comparable

The starting problem is dense, contiguous, row-major, single-precision
`C = A * B`. M is the number of A/C rows, N is the number of B/C columns,
and K is the shared dimension. Document dimensions in that order.

All registered functions obey [the kernel contract](../include/matmul.h).
That header defines buffer ownership, layout, alignment, and empty shapes.
Arguments are prevalidated by the caller, outside hot loops.

The initial tests use finite, moderate inputs; this is not a specification for
NaN/infinity inputs or overflowing arithmetic. Exceptional inputs and mixed
precision can become a separate numerical experiment later.

For a new CPU implementation:

1. Add a separate source file and a declaration in `include/matmul.h`.
2. Add the source to the `matmul` CMake target.
3. Add one entry to `src/registry.c`.
   Use a null availability callback for a portable backend or a CPU/OS check
   for an optional instruction set. Registry order does not determine availability.
4. Run `nob test` and `nob test --mode release`; both exercise every entry.
5. Run `nob bench -- --impl all` with the dimensions you want to compare.
   Keep compiler, configuration, inputs, and sampling options the same.

Add optional instruction sets and backends when their lesson needs them, with
a fallback or explicit availability check. Keep the portable build working;
avoid global AVX/CUDA flags. Check runtime CPU features before distributing
an intrinsics binary that needs a particular instruction set.

The registry lists built implementations, including unavailable ISA variants.
Call `matmul_implementation_available` before invoking a registry entry. Tests
skip unavailable entries; the benchmark marks them in `--list` and rejects
explicit requests to run them. Packing kernels may use bounded automatic
workspace; copying and workspace preparation remain inside each measured call.

External BLAS backends may allocate/cache internal workspace. Their ordinary
synchronous calls use the same host-array contract and correctness checks.
Per-call preparation, including packing, is timed; no prepacked input is
provided for free. Initial validation/warmup also initializes a library before
measurement, so the benchmark does not measure cold library startup.

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

The common comparison is a synchronous call on preallocated host arrays.
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

Profiles locate hotspots and have different measurement conditions from the
benchmark. Use [the profiling guide](profiling.md) for that workflow.

Minimum, median and maximum show some timing variation. GFLOP/s uses the
median. CPU full-call throughput is labeled effective GFLOP/s, including
packing and thread coordination. GPU full-call rows report milliseconds;
their GFLOP/s field is omitted. Separate CUDA-event rows expose GPU kernel
throughput as a diagnostic, rather than a ranking against CPU full-call
throughput. The checksum makes output observable, but correctness comes from the
elementwise reference check, not from the checksum.

CSV includes compiler/version, CMake configuration, dimensions, seed, repeats,
warmup and batch. Keep experiment notes with the source revision (or a diff),
CPU/GPU, OS, full configure command/flags and power conditions. Later include
thread count and GPU/backend parameters. The CSV alone is not a complete
reproduction record. Do not commit transient build directories or timings as
universal performance claims.

When adding packed CPU or GPU kernels, distinguish one-shot cost from reuse
of prepared data. A GPU wrapper must preserve the host-array contract and
wait for completion, including transfers and synchronization in host timings.
Report kernel-only/device timings separately.

The CUDA learning backends use a fresh device allocation per host call. Input
copies, NaN output initialization, launch, synchronization, output copy and
freeing are included. This host-call baseline does not retain device buffers.
Device errors terminate
with diagnostics because the shared void-returning interface cannot carry an
error code. Missing hardware is detected before invoking the backend.

## CUDA event measurements

The default `--timing both` measures full calls for all backends and adds a
separate CUDA-event measurement for each available CUDA backend, including
when `--impl all` is selected. CPU backends have only the full-call row. Use
`--timing host` to request only full-call measurements or `--timing kernel`
to request only CUDA events for one named CUDA implementation.

CUDA-event measurements use reusable device buffers. Allocation, one-time
input copies and event creation happen
before sampling. The prepared path is validated against the same double oracle
in addition to the initial host-call validation.

Every sample poisons device C before its start event, then queues `batch`
launches on the same default stream and records a stop event. Waiting for the
stop event reports execution errors and ensures completion before elapsed time
is read. CUDA-event milliseconds are converted to seconds and divided by batch.
Output is copied back and checked outside the interval after every sample.
Device buffers and events are freed after measurement. Warmup launches use
the same prepared buffers and are not included in the reported sample set.

This measures a window on the GPU timeline, excluding allocation, copies,
poisoning, host validation and cleanup. It includes any scheduling delays or
launch gaps inside that window. No other GPU workloads should be run during a
comparison. Inputs are reused and caches are not flushed. Tiny kernels can
benefit from batching but are still sensitive to event resolution and host
launch throughput. This is not a cold-data or end-to-end latency measurement.

Human output labels rows `full-call` or `kernel`. The final CSV `timing`
column contains `host` or `kernel`; GPU host rows have an empty `gflops` field,
while CPU host rows retain effective throughput. Both measurements use the
same dimensions and sampling settings but run separately. Their difference
does not isolate an exact transfer cost: preparation, cache state and scheduling
also differ. Compare full-call milliseconds for host-array workloads; do not
mix CPU effective GFLOP/s and GPU kernel GFLOP/s as an equivalent ranking.
Kernel timing is unavailable in CPU-only builds and is not an
internal Samply profile/validation mode.

CUDA event semantics and approximate timer resolution are documented in the
[CUDA runtime event API](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-runtime-api/group__CUDART__EVENT.html).
