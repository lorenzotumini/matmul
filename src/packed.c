#include "matmul.h"
#include <string.h>

/* Same 128-element cache tiles and row/p/col arithmetic as the blocked
 * baseline. The additional technique is a copy of each B panel into compact
 * workspace. Copying is timed, and there is no explicit SIMD or threading.
 */
void matmul_packed(size_t m, size_t n, size_t k,
                    const float *a, const float *b, float *c) {
    enum { TILE = 128 };
    if (!m || !n) return;
    for (size_t i = 0; i < m*n; ++i) c[i] = 0.0f;
    if (!k) return;
    float panel[TILE * TILE]; /* Bounded automatic workspace: 64 KiB. */

    for (size_t row0 = 0; row0 < m;) {
        size_t re = row0 + (m - row0 < TILE ? m - row0 : TILE);
        for (size_t col0 = 0; col0 < n;) {
            size_t ce = col0 + (n - col0 < TILE ? n - col0 : TILE);
            size_t width = ce - col0;
            for (size_t p0 = 0; p0 < k;) {
                size_t pe = p0 + (k - p0 < TILE ? k - p0 : TILE);
                for (size_t p = p0; p < pe; ++p)
                    memcpy(panel + (p - p0)*width, b + p*n + col0,
                           width * sizeof(float));
                for (size_t row = row0; row < re; ++row) {
                    for (size_t p = p0; p < pe; ++p) {
                        float av = a[row*k + p];
                        const float *bp = panel + (p - p0)*width;
                        for (size_t j = 0; j < width; ++j)
                            c[row*n + col0 + j] += av * bp[j];
                    }
                }
                p0 = pe;
            }
            col0 = ce;
        }
        row0 = re;
    }
}
