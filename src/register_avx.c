#include "matmul.h"
#include <immintrin.h>
#include <math.h>
#if defined(_MSC_VER) && !defined(__clang__)
#define CPU_RESTRICT __restrict
#else
#define CPU_RESTRICT restrict
#endif

/* Eight-lane FMA arithmetic as in simd_avx, with a 4x16 register tile.
 * No cache tiling, input packing or OpenMP: the change is register reuse.
 */
enum { MR = 4, NR = 16 };

static void kernel_4x16(size_t count, size_t k, size_t n,
                        const float *CPU_RESTRICT a, const float *CPU_RESTRICT b,
                        float *CPU_RESTRICT c, int accumulate) {
    __m256 c00 = accumulate ? _mm256_loadu_ps(c) : _mm256_setzero_ps();
    __m256 c01 = accumulate ? _mm256_loadu_ps(c + 8) : _mm256_setzero_ps();
    __m256 c10 = accumulate ? _mm256_loadu_ps(c + n) : _mm256_setzero_ps();
    __m256 c11 = accumulate ? _mm256_loadu_ps(c + n + 8) : _mm256_setzero_ps();
    __m256 c20 = accumulate ? _mm256_loadu_ps(c + 2*n) : _mm256_setzero_ps();
    __m256 c21 = accumulate ? _mm256_loadu_ps(c + 2*n + 8) : _mm256_setzero_ps();
    __m256 c30 = accumulate ? _mm256_loadu_ps(c + 3*n) : _mm256_setzero_ps();
    __m256 c31 = accumulate ? _mm256_loadu_ps(c + 3*n + 8) : _mm256_setzero_ps();

    for (size_t p = 0; p < count; ++p) {
        __m256 b0 = _mm256_loadu_ps(b + p*n);
        __m256 b1 = _mm256_loadu_ps(b + p*n + 8);
        __m256 a0 = _mm256_set1_ps(a[p]);
        __m256 a1 = _mm256_set1_ps(a[k + p]);
        __m256 a2 = _mm256_set1_ps(a[2*k + p]);
        __m256 a3 = _mm256_set1_ps(a[3*k + p]);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
    }
    _mm256_storeu_ps(c, c00);
    _mm256_storeu_ps(c + 8, c01);
    _mm256_storeu_ps(c + n, c10);
    _mm256_storeu_ps(c + n + 8, c11);
    _mm256_storeu_ps(c + 2*n, c20);
    _mm256_storeu_ps(c + 2*n + 8, c21);
    _mm256_storeu_ps(c + 3*n, c30);
    _mm256_storeu_ps(c + 3*n + 8, c31);
}

/* Incomplete register tiles: eight-column vectors, then individual columns.
 * Neither loads nor stores cross the actual row/column boundaries.
 */
static void edge_tile(size_t rows, size_t cols, size_t count, size_t k, size_t n,
                       const float *CPU_RESTRICT a, const float *CPU_RESTRICT b,
                       float *CPU_RESTRICT c, int accumulate) {
    for (size_t row = 0; row < rows; ++row) {
        size_t col = 0;
        for (; cols - col >= 8; col += 8) {
            __m256 sum = accumulate ? _mm256_loadu_ps(c + row*n + col) :
                                      _mm256_setzero_ps();
            for (size_t p = 0; p < count; ++p) {
                __m256 av = _mm256_set1_ps(a[row*k + p]);
                __m256 bv = _mm256_loadu_ps(b + p*n + col);
                sum = _mm256_fmadd_ps(av, bv, sum);
            }
            _mm256_storeu_ps(c + row*n + col, sum);
        }
        for (; col < cols; ++col) {
            float sum = accumulate ? c[row*n + col] : 0.0f;
            for (size_t p = 0; p < count; ++p)
                sum = fmaf(a[row*k + p], b[p*n + col], sum);
            c[row*n + col] = sum;
        }
    }
}

void matmul_register_avx(size_t m, size_t n, size_t k,
                           const float *a, const float *b, float *c) {
    if (!m || !n) return;
    if (!k) {
        /* Do not even form offsets into the empty A/B arrays. */
        for (size_t i = 0; i < m*n; ++i) c[i] = 0.0f;
        return;
    }
    for (size_t row = 0; row < m;) {
        size_t rows = m - row < MR ? m - row : MR;
        for (size_t col = 0; col < n;) {
            size_t cols = n - col < NR ? n - col : NR;
            if (rows == MR && cols == NR)
                kernel_4x16(k, k, n, a + row*k, b + col, c + row*n + col, 0);
            else
                edge_tile(rows, cols, k, k, n, a + row*k, b + col, c + row*n + col, 0);
            col += cols;
        }
        row += rows;
    }
}
