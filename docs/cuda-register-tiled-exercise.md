# CUDA: guided register tiling

Complete the four TODOs in [`src/cuda_register_tiled.cu`](../src/cuda_register_tiled.cu).
The [concept summary](13-gpu-register-tiling.md) explains the takeaways.

Each thread owns a 2-by-2 group of adjacent outputs and four private sums.
The block still has 16-by-16 threads, but now covers 32-by-32 outputs. Its
reduction chunks still have depth 16. Coordinates, launch geometry and the
barriers you learned previously are supplied; focus on loading and reuse.

## 1. Populate the larger shared tiles

The shared A tile is 32 rows by 16 reduction positions. Shared B is 16
reduction positions by 32 columns. Each has 512 elements, so 256 threads
must supply two elements per input rather than one.

The supplied `half` loop distributes the loads:

| Shared slot | Source matrix coordinates |
| --- | --- |
| `a_tile[ar][tx]` | A at `(row_base + ar, p0 + tx)` |
| `b_tile[ty][bc]` | B at `(p0 + ty, col_base + bc)` |

Replace the zeros with guarded loads, using zero for a missing input. Shared
tile coordinates and global matrix coordinates have different strides. Do not
guard loading by whether this thread's four outputs exist: other threads can
need its loads. Keep every thread participating in both barriers.

Before coding, trace `tx=3, ty=5` in the first block. Which four outputs does
it own? Which A rows and B columns does it load in the two half iterations?
Why are loading ownership and output ownership different?

## 2. Load four reusable scalar values

Inside the reduction loop, replace the scalar placeholders with two shared A
values and two shared B values. A uses this thread's two local output rows;
B uses its two local output columns. The other shared-memory coordinate is
the reduction position `p`.

Use the four scalars for the arithmetic, so it is clear how many values are
loaded before reuse. No additional global-memory loads belong in this loop.

## 3. Update the four accumulators

Use the four combinations of those two A values and two B values to update
`sum00`, `sum01`, `sum10`, `sum11`. All four sums must survive across every K
chunk. Ordinary multiply/add expressions let the CUDA compiler generate FMA.

How many shared-memory value loads would four independent output threads
perform for one reduction position? How many does this thread perform?

## 4. Store each valid output independently

The four output coordinates are `(row,col)`, `(row,col+1)`, `(row+1,col)`
and `(row+1,col+1)`. Protect each store. A top-left element inside C does not
guarantee the other three fit, particularly with odd dimensions.

Trace M=33, N=35, K=17: the output grid has two blocks in each direction,
and K has two reduction chunks. Work through padding and the bottom/right
stores before running the tests. Never return early before the barriers.

## Check and compare

```powershell
.\nob.exe test --cuda --filter '^cuda_register_tiled$'
```

The template compiles but intentionally fails with NaN output until completed.
It is registered for the same 61 fixtures as the earlier kernels, including
odd dimensions and tile tails. `--filter cuda` and `--impl all` include this
exercise, so they will also fail until it is complete. The separate timing
infrastructure test continues to validate the completed baselines.

After correctness passes:

```powershell
.\nob.exe test --cuda --mode release --filter '^cuda_register_tiled$'
.\nob.exe bench --cuda -- --impl cuda_tiled --m 512 --n 512 --k 512
.\nob.exe bench --cuda -- --impl cuda_register_tiled --m 512 --n 512 --k 512
```

Default output shows full-call latency and separate CUDA-event kernel time
and throughput. Compare several sizes and rectangular shapes. This version
also doubles the output tile width, so a speed difference includes the change
in block-level reuse and grid size, as well as per-thread register reuse.
More registers and larger tiles can reduce active parallel work.

For memory/race/barrier checking after building the Debug test:

```powershell
compute-sanitizer --tool memcheck --error-exitcode 1 .\build\clang-cl\cuda-debug\matmul_tests.exe --impl cuda_register_tiled
compute-sanitizer --tool racecheck --error-exitcode 1 .\build\clang-cl\cuda-debug\matmul_tests.exe --impl cuda_register_tiled
compute-sanitizer --tool synccheck --error-exitcode 1 .\build\clang-cl\cuda-debug\matmul_tests.exe --impl cuda_register_tiled
```
