#include "matmul.h"
#include <immintrin.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(_MSC_VER) && !defined(__clang__)
#define CPU_RESTRICT __restrict
#else
#define CPU_RESTRICT restrict
#endif

/* Three levels of work:
 *   output tiles: independent work for CPU threads
 *   cache tiles: 64 rows x 128 columns x 256 reduction terms
 *   register tiles: 4 rows x 16 columns held in eight YMM accumulators
 *
 * These are starting heuristics, not machine-independent optimal sizes.
 * No packing/allocation, fast-math, or FMA is used. Each output accumulates
 * products in increasing reduction-index order, just like the baselines.
 */
enum { MC = 64, NC = 128, KC = 256, MR = 4, NR = 16 };

static size_t clipped_end(size_t start, size_t limit, size_t block) {
    size_t remaining = limit - start;
    return start + (remaining < block ? remaining : block);
}

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
        c00 = _mm256_add_ps(c00, _mm256_mul_ps(a0, b0));
        c01 = _mm256_add_ps(c01, _mm256_mul_ps(a0, b1));
        c10 = _mm256_add_ps(c10, _mm256_mul_ps(a1, b0));
        c11 = _mm256_add_ps(c11, _mm256_mul_ps(a1, b1));
        c20 = _mm256_add_ps(c20, _mm256_mul_ps(a2, b0));
        c21 = _mm256_add_ps(c21, _mm256_mul_ps(a2, b1));
        c30 = _mm256_add_ps(c30, _mm256_mul_ps(a3, b0));
        c31 = _mm256_add_ps(c31, _mm256_mul_ps(a3, b1));
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
                sum = _mm256_add_ps(sum, _mm256_mul_ps(av, bv));
            }
            _mm256_storeu_ps(c + row*n + col, sum);
        }
        for (; col < cols; ++col) {
            float sum = accumulate ? c[row*n + col] : 0.0f;
            for (size_t p = 0; p < count; ++p)
                sum += a[row*k + p] * b[p*n + col];
            c[row*n + col] = sum;
        }
    }
}

static void output_tile(size_t row0, size_t row_end, size_t col0, size_t col_end,
                         size_t k, size_t n, size_t reduction_block,
                         const float *a, const float *b, float *c) {
    for (size_t p0 = 0; p0 < k;) {
        size_t p_end = clipped_end(p0, k, reduction_block);
        for (size_t row = row0; row < row_end;) {
            size_t rows = row_end - row < MR ? row_end - row : MR;
            for (size_t col = col0; col < col_end;) {
                size_t cols = col_end - col < NR ? col_end - col : NR;
                const float *at = a + row*k + p0;
                const float *bt = b + p0*n + col;
                float *ct = c + row*n + col;
                if (rows == MR && cols == NR)
                    kernel_4x16(p_end - p0, k, n, at, bt, ct, p0 != 0);
                else
                    edge_tile(rows, cols, p_end - p0, k, n, at, bt, ct, p0 != 0);
                col += cols;
            }
            row += rows;
        }
        p0 = p_end;
    }
}

void matmul_cpu_combined(size_t m, size_t n, size_t k,
                          const float *a, const float *b, float *c) {
    if (!m || !n) return;
    if (!k) {
        for (size_t i = 0; i < m*n; ++i) c[i] = 0.0f;
        return;
    }

    /* Small problems skip thread-team setup and output cache partitioning.
     * B panels larger than about 256 KiB also get reduction-axis blocking.
     */
    int large = (double)m * (double)n * (double)k >= 8.0 * 1024 * 1024;
    size_t row_block = large ? MC : m;
    size_t col_block = large ? NC : n;
    size_t reduction_block = (double)n * (double)k > 65536.0 ? KC : k;
    size_t row_tiles = m / row_block + (m % row_block != 0);
    size_t col_tiles = n / col_block + (n % col_block != 0);
    size_t tiles = row_tiles * col_tiles;

    /* Keep the signed OpenMP loop portable to older MSVC implementations. */
    if (tiles > (size_t)PTRDIFF_MAX) {
        for (size_t r = 0; r < m;) {
            size_t re = clipped_end(r, m, row_block);
            for (size_t j = 0; j < n;) {
                size_t je = clipped_end(j, n, col_block);
                output_tile(r, re, j, je, k, n, reduction_block, a, b, c);
                j = je;
            }
            r = re;
        }
        return;
    }
    ptrdiff_t tasks = (ptrdiff_t)tiles;
#ifdef _OPENMP
    int threads = large && !omp_in_parallel() ? omp_get_max_threads() : 1;
    if ((size_t)threads > tiles) threads = (int)tiles;
    /* Do not even enter an OpenMP region for the serial path. */
    if (threads > 1) {
        #pragma omp parallel for schedule(static) num_threads(threads) default(none) \
            shared(tasks, col_tiles, row_block, col_block, m, n, k, reduction_block, a, b, c)
        for (ptrdiff_t task = 0; task < tasks; ++task) {
            size_t r = ((size_t)task / col_tiles) * row_block;
            size_t j = ((size_t)task % col_tiles) * col_block;
            output_tile(r, clipped_end(r, m, row_block),
                        j, clipped_end(j, n, col_block), k, n, reduction_block, a, b, c);
        }
        return;
    }
#endif
    for (ptrdiff_t task = 0; task < tasks; ++task) {
        size_t r = ((size_t)task / col_tiles) * row_block;
        size_t j = ((size_t)task % col_tiles) * col_block;
        output_tile(r, clipped_end(r, m, row_block),
                    j, clipped_end(j, n, col_block), k, n, reduction_block, a, b, c);
    }
}
