#include "matmul.h"
#include <immintrin.h>

/* CMake enables AVX only for this file. The registry checks availability
 * before exposing it; do not call this function directly on an unchecked CPU.
 * Keep SSE and AVX separate so their vector widths can be compared.
 */
void matmul_simd_avx(size_t m, size_t n, size_t k,
                     const float *a, const float *b, float *c) {
    if (!m || !n) return;

    for (size_t row = 0; row < m; ++row)
        for (size_t col = 0; col < n; ++col)
            c[row * n + col] = 0.0f;

    /* Example: N=17 means two groups of eight and one scalar column. */
    const size_t vector_end = n - n % 8;

    for (size_t row = 0; row < m; ++row) {
        float *c_row = c + row * n;
        for (size_t p = 0; p < k; ++p) {
            float a_val = a[row * k + p];
            const float *b_row = b + p * n;
            __m256 a8 = _mm256_set1_ps(a_val);

            size_t col = 0;
            for (; col < vector_end; col += 8) {
                /* loadu/storeu do not require 32-byte alignment. */
                __m256 b8 = _mm256_loadu_ps(b_row + col);
                __m256 c8 = _mm256_loadu_ps(c_row + col);
                (void)a8; /* Remove these casts when the TODO uses the values. */
                (void)b8;

				__m256 product = _mm256_mul_ps(a8, b8);
				__m256 updated = _mm256_add_ps(c8, product);
				_mm256_storeu_ps(c_row + col, updated);
            }

            /* Zero to seven leftover columns, including the whole row if N<8. */
            for (; col < n; ++col)
                c_row[col] += a_val * b_row[col];
        }
    }
}
