#include "matmul.h"

/* SSE is part of the x86-64 baseline. Other architectures omit this exercise
 * from the registry; the portable kernels remain available.
 */
#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>

void matmul_simd_sse(size_t m, size_t n, size_t k,
                     const float *a, const float *b, float *c) {
    if (!m || !n) return;

    for (size_t row = 0; row < m; ++row)
        for (size_t col = 0; col < n; ++col)
            c[row * n + col] = 0.0f;

    /* Only complete groups of four belong in the vector loop.
     * Example: N=7 gives vector_end=4, leaving columns 4, 5, 6 for the tail.
     */
    const size_t vector_end = n - n % 4;

    for (size_t row = 0; row < m; ++row) {
        float *c_row = c + row * n;
        for (size_t p = 0; p < k; ++p) {
            float a_val = a[row * k + p];
            const float *b_row = b + p * n;

            /* __m128 holds four floats. This broadcasts a scalar:
             * a4 = [a_val, a_val, a_val, a_val].
             */
            __m128 a4 = _mm_set1_ps(a_val);

            size_t col = 0;
            for (; col < vector_end; col += 4) {
                /* loadu reads four adjacent floats without requiring 16-byte
                 * alignment. Even if an allocation is aligned, an arbitrary
                 * row length can make a later row unaligned.
                 */
                __m128 b4 = _mm_loadu_ps(b_row + col);
                __m128 c4 = _mm_loadu_ps(c_row + col);

				__m128 products = _mm_mul_ps(a4, b4);
				__m128 updated = _mm_add_ps(c4, products);
                _mm_storeu_ps(c_row + col, updated);
            }

            /* Scalar tail: handles the remaining zero to three columns.
             * This also handles the entire row when N < 4.
             */
            for (; col < n; ++col)
                c_row[col] += a_val * b_row[col];
        }
    }
}
#endif
