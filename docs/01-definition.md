# 01 — From the definition to C

The first goal is to understand a correct matrix multiplication. This lesson's
implementation is complete in `src/naive.c` and is preserved as the baseline.
Use `nob test` while working. Timing comes after correctness.

## What are we computing?

Let A have M rows and K columns, and B have K rows and N columns.
Their product C has M rows and N columns:

```text
A: M x K        B: K x N        C: M x N

C[i,j] = sum over p = 0,...,K-1 of A[i,p] * B[p,j]
```

Each output is the dot product of one row of A and one column of B.
The two inner dimensions must agree. The result is generally rectangular.
We use zero-based indices throughout the code.

For example:

```text
A = [ 1  2  3 ]    B = [  7   8 ]    C = [  58   64 ]
    [ 4  5  6 ]        [  9  10 ]        [ 139  154 ]
                       [ 11  12 ]

C[0,0] = 1*7 + 2*9 + 3*11 = 58
C[1,1] = 4*8 + 5*10 + 6*12 = 154
```

Work out the other two entries on paper before coding.

## How are matrices stored?

For now, each matrix is one contiguous allocation of `float` values. Rows sit
next to one another in memory. This is called row-major storage:

```text
A = [1, 2, 3, 4, 5, 6]
     row 0   row 1
```

An element in a matrix with `columns` columns is at:

```text
index(row, column) = row * columns + column
```

Apply this separately to A, B and C: they have different column counts.
These are ordinary flat arrays, not `float **` arrays of row pointers.

## Understand the baseline

Before opening `src/naive.c`, sketch the direct definition as three loops on
paper. Use a `float` accumulator and assign every output entry on every call.
Then compare your sketch with the implementation.

Answer:

1. Which two indices identify an output entry?
2. Which index walks along the shared dimension?
3. Where does the accumulator need to start at zero?
4. What is the flat-array index for each of A, B and C?
5. If K is zero, what value should the empty sum produce?

The kernel needs no allocation, threading, intrinsics, or printing. The caller
already owns the buffers; [the header](../include/matmul.h) defines the contract.

Run:

```powershell
.\nob.exe test
```

A mismatch reports the output row and column, actual value, expected value,
and numerical tolerance. A following line identifies the dimensions, pattern,
seed and call number. For detailed kernel-test output, use
`nob test --filter kernels -- -V`.

If you get stuck, inspect one output with a debugger. Write down its expected
three products and compare each array access with your paper calculation.

Once you understand the loops, check the optimized build and take a first timing:

```powershell
.\nob.exe test --mode release
.\nob.exe bench -- --impl naive --m 256 --n 256 --k 256
```

## Investigate your baseline

Try square sizes 64, 128, 256 and 512, plus a rectangular shape. Keep the
compiler, build type, inputs and sample count fixed. Record the results and
your explanation before reading ahead.

- How many multiplications and additions does one output need?
- What happens to the operation count when all three dimensions double?
- Which accesses are adjacent in memory? Which jump between rows?
- Why might two matrices with similar operation counts take different times?
- Why does a Release build differ from Debug?
- Does writing scalar C guarantee scalar machine instructions?

The usual count is approximately `2*M*N*K` floating-point operations. With a
sum initialized to zero there are K multiplies and K additions per output;
other counting conventions omit the first addition. We use the conventional
count consistently for comparisons.

Write a short note with your prediction, command, compiler/build, observation
and interpretation. Keep `src/naive.c` as the baseline when you start the next
implementation in a separate file. Follow [the conventions](conventions.md)
to register it for the same tests and benchmark.
