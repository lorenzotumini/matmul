#include "matmul.h"
#include <immintrin.h>
#include <math.h>

/* Same traversal as the eight-lane AVX version, widened to sixteen floats and with
 * fused multiply-add, including the scalar tail.
 */
void matmul_simd_avx512(size_t m, size_t n, size_t k,
                        const float *a, const float *b, float *c) {
    if (!m || !n) return;
    for (size_t i = 0; i < m*n; ++i) c[i] = 0.0f;
    size_t vector_end = n - n % 16;
    for (size_t row = 0; row < m; ++row) {
        float *cr = c + row*n;
        for (size_t p = 0; p < k; ++p) {
            float av = a[row*k + p];
            __m512 a16 = _mm512_set1_ps(av);
            const float *br = b + p*n;
            size_t col = 0;
            for (; col < vector_end; col += 16) {
                __m512 b16 = _mm512_loadu_ps(br + col);
                __m512 c16 = _mm512_loadu_ps(cr + col);
			    _mm512_storeu_ps(cr + col, _mm512_fmadd_ps(a16, b16, c16));
				// for reference the version without FMA would be:
				// __m512 prod = _mm512_mul_ps(a16, b16);
				// _mm512_storeu_ps(cr + col, _mm512_add_ps(prod, c16));
            }
            for (; col < n; ++col) cr[col] = fmaf(av, br[col], cr[col]);
        }
    }
}
