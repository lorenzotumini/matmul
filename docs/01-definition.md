# 01 — Matrix multiplication basics

For A with shape M-by-K and B with shape K-by-N, the product C has shape
M-by-N. Each output is the dot product of one row of A and one column of B:

```text
C[row,col] = sum over p = 0,...,K-1 of A[row,p] * B[p,col]
```

The shared dimension K must agree. The matrices need not be square.

## Storage and indexing

In contiguous row-major storage, rows sit next to each other. An element's
flat-array index is `row * number_of_columns + column`:

| Matrix | Shape | Index |
| --- | --- | --- |
| A | M-by-K | `row*K + p` |
| B | K-by-N | `p*N + col` |
| C | M-by-N | `row*N + col` |

Each matrix has its own row length. A flat allocation and an array of row
pointers are different representations.

## Correctness and cost

- `C = A*B` overwrites the output; its previous contents must not affect the result.
- A dot-product accumulator starts at zero. If K is zero, the empty sum is zero.
- If M or N is zero, there are no output elements.
- Floating-point rounding means mathematically equivalent calculations may
  differ slightly; numerical comparisons need a justified tolerance.

The conventional operation count is approximately `2*M*N*K`. Doubling all
three dimensions multiplies the arithmetic work by eight. This count describes
work, but memory access and generated machine code also determine elapsed time.
