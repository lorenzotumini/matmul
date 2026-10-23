#ifndef MATMUL_H
#define MATMUL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* C[M,N] = A[M,K] * B[K,N], contiguous row-major float arrays.
 * Inputs are read-only; output must be completely overwritten on every call.
 * A, B, C do not overlap. The caller supplies correctly sized buffers and
 * dimensions whose products fit in size_t. No alignment beyond float required.
 * M or N == 0: no output. K == 0: write an M-by-N matrix of zeros.
 * Even for empty arrays, pass valid pointers (the harness does this).
 * The kernel performs no allocation, I/O, timing, or argument parsing.
 */
typedef void (*MatmulFn)(size_t m, size_t n, size_t k,
                         const float *a, const float *b, float *c);

void matmul_naive(size_t m, size_t n, size_t k,
                  const float *a, const float *b, float *c);
void matmul_loop_interchange(size_t m, size_t n, size_t k,
                  const float *a, const float *b, float *c);
void matmul_blocked(size_t m, size_t n, size_t k,
                    const float *a, const float *b, float *c);

typedef struct {
    const char *name;
    MatmulFn run;
    const char *description;
} MatmulImplementation;

const MatmulImplementation *matmul_implementations(size_t *count);

#ifdef __cplusplus
}
#endif
#endif
