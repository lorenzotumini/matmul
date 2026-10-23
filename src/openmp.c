#include "matmul.h"
#include <stdint.h>

/* A worker owns one complete output row. Its automatic local variables are
 * private to that call; only A and B are read across different workers.
 */
static void compute_row(size_t row, size_t n, size_t k,
                        const float *a, const float *b, float *c) {
    float *c_row = c + row * n;
    for (size_t col = 0; col < n; ++col)
        c_row[col] = 0.0f;

    for (size_t p = 0; p < k; ++p) {
        float a_val = a[row * k + p];
        const float *b_row = b + p * n;
        for (size_t col = 0; col < n; ++col)
            c_row[col] += a_val * b_row[col];
    }
}

void matmul_openmp(size_t m, size_t n, size_t k,
                   const float *a, const float *b, float *c) {
    if (!m || !n) return;

    /* Older MSVC OpenMP requires a signed loop counter. Keep a serial path
     * for dimensions that cannot be represented by that counter.
     */
    if (m > (size_t)PTRDIFF_MAX) {
        for (size_t row = 0; row < m; ++row)
            compute_row(row, n, k, a, b, c);
        return;
    }
    const ptrdiff_t rows = (ptrdiff_t)m;

	#pragma omp parallel for schedule(static) default(none) shared(rows, n, k, a, b, c)
    for (ptrdiff_t row = 0; row < rows; ++row) {
        compute_row((size_t)row, n, k, a, b, c);
    }
}
