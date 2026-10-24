#include "matmul.h"
#include <immintrin.h>
#include <math.h>

/* Eight-float AVX SIMD with explicit FMA, including the scalar tail.
 * The registry checks both AVX and FMA support before this function is used.
 */
void matmul_simd_avx(size_t m, size_t n, size_t k,
                         const float *a, const float *b, float *c) {
    if (!m || !n) return;
    for (size_t i = 0; i < m*n; ++i) c[i] = 0.0f;
    size_t vector_end = n - n % 8;
    for (size_t row = 0; row < m; ++row) {
        float *cr = c + row*n;
        for (size_t p = 0; p < k; ++p) {
            float av = a[row*k + p];
            __m256 a8 = _mm256_set1_ps(av);
            const float *br = b + p*n;
            size_t col = 0;
            for (; col < vector_end; col += 8) {
                __m256 b8 = _mm256_loadu_ps(br + col);
                __m256 c8 = _mm256_loadu_ps(cr + col);
                _mm256_storeu_ps(cr + col, _mm256_fmadd_ps(a8, b8, c8));
				// for reference the version without FMA would be:
				// __m256 prod = _mm256_mul_ps(a8, b8);
				// _mm256_storeu_ps(cr + col, _mm256_add_ps(prod, c8));
            }
            for (; col < n; ++col) cr[col] = fmaf(av, br[col], cr[col]);
        }
    }
}
