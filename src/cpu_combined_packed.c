#include "cpu_combined_internal.h"
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Large-matrix reference: packed micro-panels + sixteen-float FMA + threads.
 * B is packed ONCE per column/reduction panel and shared by the team. Each
 * worker packs its own A rows. Packing and stack workspace are inside the call.
 * The dispatcher calls this only when enough work can amortize packing.
 */
enum { MC = 32, NC = 256, KC = 256, MR = 8, NR = 32 };

static size_t end_at(size_t start, size_t limit, size_t block) {
    return start + (limit - start < block ? limit - start : block);
}

/* Layout: [column micro-panel][reduction term][32 contiguous columns]. */
static void pack_b(size_t col0, size_t ce, size_t p0, size_t depth, size_t n,
                     const float *b, float *packed) {
    size_t panels = (ce - col0) / NR + ((ce - col0) % NR != 0);
    for (size_t panel = 0; panel < panels; ++panel) {
        size_t col = col0 + panel*NR;
        size_t width = ce - col < NR ? ce - col : NR;
        for (size_t p = 0; p < depth; ++p) {
            float *dst = packed + (panel*depth + p)*NR;
            memcpy(dst, b + (p0 + p)*n + col, width*sizeof(float));
            for (size_t j = width; j < NR; ++j) dst[j] = 0.0f;
        }
    }
}

/* Layout: [row micro-panel][reduction term][8 rows' A values]. */
static void pack_a(size_t row0, size_t re, size_t p0, size_t depth, size_t k,
                     const float *a, float *packed) {
    size_t panels = (re - row0) / MR + ((re - row0) % MR != 0);
    for (size_t panel = 0; panel < panels; ++panel) {
        size_t row = row0 + panel*MR;
        size_t height = re - row < MR ? re - row : MR;
        for (size_t p = 0; p < depth; ++p)
            for (size_t i = 0; i < MR; ++i)
                packed[(panel*depth + p)*MR + i] =
                    i < height ? a[(row + i)*k + p0 + p] : 0.0f;
    }
}

/* Sixteen independent accumulators. Padded inputs allow a regular arithmetic
 * loop; masked stores and guarded row addresses keep edge outputs in bounds.
 */
static void kernel_8x32(size_t rows, size_t cols, size_t depth, size_t n,
                         const float *ap, const float *bp, float *c, int accumulate) {
    unsigned low = cols < 16 ? (1u << cols) - 1u : 0xffffu;
    unsigned upper = cols > 16 ? cols - 16 : 0;
    unsigned high = upper < 16 ? (1u << upper) - 1u : 0xffffu;
    __mmask16 m0 = (__mmask16)low, m1 = (__mmask16)high;
    __m512 c0a = accumulate && rows > 0 ? _mm512_maskz_loadu_ps(m0, c + 0*n) : _mm512_setzero_ps();
    __m512 c0b = accumulate && rows > 0 && cols > 16 ? _mm512_maskz_loadu_ps(m1, c + 0*n + 16) : _mm512_setzero_ps();
    __m512 c1a = accumulate && rows > 1 ? _mm512_maskz_loadu_ps(m0, c + 1*n) : _mm512_setzero_ps();
    __m512 c1b = accumulate && rows > 1 && cols > 16 ? _mm512_maskz_loadu_ps(m1, c + 1*n + 16) : _mm512_setzero_ps();
    __m512 c2a = accumulate && rows > 2 ? _mm512_maskz_loadu_ps(m0, c + 2*n) : _mm512_setzero_ps();
    __m512 c2b = accumulate && rows > 2 && cols > 16 ? _mm512_maskz_loadu_ps(m1, c + 2*n + 16) : _mm512_setzero_ps();
    __m512 c3a = accumulate && rows > 3 ? _mm512_maskz_loadu_ps(m0, c + 3*n) : _mm512_setzero_ps();
    __m512 c3b = accumulate && rows > 3 && cols > 16 ? _mm512_maskz_loadu_ps(m1, c + 3*n + 16) : _mm512_setzero_ps();
    __m512 c4a = accumulate && rows > 4 ? _mm512_maskz_loadu_ps(m0, c + 4*n) : _mm512_setzero_ps();
    __m512 c4b = accumulate && rows > 4 && cols > 16 ? _mm512_maskz_loadu_ps(m1, c + 4*n + 16) : _mm512_setzero_ps();
    __m512 c5a = accumulate && rows > 5 ? _mm512_maskz_loadu_ps(m0, c + 5*n) : _mm512_setzero_ps();
    __m512 c5b = accumulate && rows > 5 && cols > 16 ? _mm512_maskz_loadu_ps(m1, c + 5*n + 16) : _mm512_setzero_ps();
    __m512 c6a = accumulate && rows > 6 ? _mm512_maskz_loadu_ps(m0, c + 6*n) : _mm512_setzero_ps();
    __m512 c6b = accumulate && rows > 6 && cols > 16 ? _mm512_maskz_loadu_ps(m1, c + 6*n + 16) : _mm512_setzero_ps();
    __m512 c7a = accumulate && rows > 7 ? _mm512_maskz_loadu_ps(m0, c + 7*n) : _mm512_setzero_ps();
    __m512 c7b = accumulate && rows > 7 && cols > 16 ? _mm512_maskz_loadu_ps(m1, c + 7*n + 16) : _mm512_setzero_ps();
    for (size_t p = 0; p < depth; ++p) {
        __m512 b0 = _mm512_loadu_ps(bp + p*NR);
        __m512 b1 = _mm512_loadu_ps(bp + p*NR + 16);
        __m512 a0 = _mm512_set1_ps(ap[p*MR + 0]);
        c0a = _mm512_fmadd_ps(a0, b0, c0a);
        c0b = _mm512_fmadd_ps(a0, b1, c0b);
        __m512 a1 = _mm512_set1_ps(ap[p*MR + 1]);
        c1a = _mm512_fmadd_ps(a1, b0, c1a);
        c1b = _mm512_fmadd_ps(a1, b1, c1b);
        __m512 a2 = _mm512_set1_ps(ap[p*MR + 2]);
        c2a = _mm512_fmadd_ps(a2, b0, c2a);
        c2b = _mm512_fmadd_ps(a2, b1, c2b);
        __m512 a3 = _mm512_set1_ps(ap[p*MR + 3]);
        c3a = _mm512_fmadd_ps(a3, b0, c3a);
        c3b = _mm512_fmadd_ps(a3, b1, c3b);
        __m512 a4 = _mm512_set1_ps(ap[p*MR + 4]);
        c4a = _mm512_fmadd_ps(a4, b0, c4a);
        c4b = _mm512_fmadd_ps(a4, b1, c4b);
        __m512 a5 = _mm512_set1_ps(ap[p*MR + 5]);
        c5a = _mm512_fmadd_ps(a5, b0, c5a);
        c5b = _mm512_fmadd_ps(a5, b1, c5b);
        __m512 a6 = _mm512_set1_ps(ap[p*MR + 6]);
        c6a = _mm512_fmadd_ps(a6, b0, c6a);
        c6b = _mm512_fmadd_ps(a6, b1, c6b);
        __m512 a7 = _mm512_set1_ps(ap[p*MR + 7]);
        c7a = _mm512_fmadd_ps(a7, b0, c7a);
        c7b = _mm512_fmadd_ps(a7, b1, c7b);
    }
    if (rows > 0) {
        _mm512_mask_storeu_ps(c + 0*n, m0, c0a);
        if (cols > 16) _mm512_mask_storeu_ps(c + 0*n + 16, m1, c0b);
    }
    if (rows > 1) {
        _mm512_mask_storeu_ps(c + 1*n, m0, c1a);
        if (cols > 16) _mm512_mask_storeu_ps(c + 1*n + 16, m1, c1b);
    }
    if (rows > 2) {
        _mm512_mask_storeu_ps(c + 2*n, m0, c2a);
        if (cols > 16) _mm512_mask_storeu_ps(c + 2*n + 16, m1, c2b);
    }
    if (rows > 3) {
        _mm512_mask_storeu_ps(c + 3*n, m0, c3a);
        if (cols > 16) _mm512_mask_storeu_ps(c + 3*n + 16, m1, c3b);
    }
    if (rows > 4) {
        _mm512_mask_storeu_ps(c + 4*n, m0, c4a);
        if (cols > 16) _mm512_mask_storeu_ps(c + 4*n + 16, m1, c4b);
    }
    if (rows > 5) {
        _mm512_mask_storeu_ps(c + 5*n, m0, c5a);
        if (cols > 16) _mm512_mask_storeu_ps(c + 5*n + 16, m1, c5b);
    }
    if (rows > 6) {
        _mm512_mask_storeu_ps(c + 6*n, m0, c6a);
        if (cols > 16) _mm512_mask_storeu_ps(c + 6*n + 16, m1, c6b);
    }
    if (rows > 7) {
        _mm512_mask_storeu_ps(c + 7*n, m0, c7a);
        if (cols > 16) _mm512_mask_storeu_ps(c + 7*n + 16, m1, c7b);
    }
}

/* Keep large automatic buffers in this function so the small-problem path
 * does not pay for their stack probing. B uses 256 KiB; each A buffer 32 KiB.
 */
void matmul_cpu_packed(size_t m, size_t n, size_t k,
                         const float *a, const float *b, float *c) {
    float packed_b[NC * KC];
    ptrdiff_t row_tiles = (ptrdiff_t)(m / MC + (m % MC != 0));
#ifdef _OPENMP
    int threads = omp_in_parallel() ? 1 : omp_get_max_threads();
    if ((ptrdiff_t)threads > row_tiles) threads = (int)row_tiles;
    #pragma omp parallel num_threads(threads) if(threads > 1) default(none) \
        shared(m, n, k, a, b, c, packed_b, row_tiles)
#endif
    {
        float packed_a[MC * KC]; /* Private to each worker. */
        for (size_t col0 = 0; col0 < n;) {
            size_t ce = end_at(col0, n, NC);
            for (size_t p0 = 0; p0 < k;) {
                size_t pe = end_at(p0, k, KC);
                size_t depth = pe - p0;
#ifdef _OPENMP
                #pragma omp single
#endif
                {
                    pack_b(col0, ce, p0, depth, n, b, packed_b);
                } /* Implicit barrier: the packed B panel is ready for readers. */
#ifdef _OPENMP
                #pragma omp for schedule(static)
#endif
                for (ptrdiff_t tile = 0; tile < row_tiles; ++tile) {
                    size_t row0 = (size_t)tile*MC;
                    size_t re = end_at(row0, m, MC);
                    pack_a(row0, re, p0, depth, k, a, packed_a);
                    for (size_t row = row0; row < re; row += MR) {
                        size_t rows = re - row < MR ? re - row : MR;
                        const float *ap = packed_a + ((row - row0)/MR)*depth*MR;
                        for (size_t col = col0; col < ce; col += NR) {
                            size_t cols = ce - col < NR ? ce - col : NR;
                            const float *bp = packed_b + ((col - col0)/NR)*depth*NR;
                            kernel_8x32(rows, cols, depth, n, ap, bp,
                                        c + row*n + col, p0 != 0);
                        }
                    }
                } /* Implicit barrier: no worker still reads B before repacking. */
                p0 = pe;
            }
            col0 = ce;
        }
    }
}

