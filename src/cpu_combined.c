#include "cpu_combined_internal.h"
#include "cpu_features.h"
#include <stdint.h>

/* One public implementation, two complementary strategies:
 *   direct: original layout, 4x16 AVX/FMA tiles, optional cache tiles/threads;
 *   packed: shared B + private A panels, 8x32 AVX-512/FMA tiles and threads.
 * Small calls never enter the packed function or reserve its large workspace.
 * These thresholds are readable starting heuristics, not an autotuner.
 */
void matmul_cpu_combined(size_t m, size_t n, size_t k,
                          const float *a, const float *b, float *c) {
    if (!m || !n) return;
    if (!k) {
        for (size_t i = 0; i < m*n; ++i) c[i] = 0.0f;
        return;
    }
#ifdef MATMUL_HAVE_CPU_PACKED
    /* Enough row reuse and arithmetic to repay copying and synchronization.
     * Guard the signed row-tile counter used by the portable OpenMP loop.
     */
    if (m >= 128 && n >= 32 &&
        (double)m * (double)n * (double)k >= 8.0 * 1024 * 1024 &&
        m / 32 + (m % 32 != 0) <= (size_t)PTRDIFF_MAX && matmul_has_avx512()) {
        matmul_cpu_packed(m, n, k, a, b, c);
        return;
    }
#endif
#ifdef MATMUL_HAVE_CPU_DIRECT
    if (matmul_has_fma()) {
        matmul_cpu_direct(m, n, k, a, b, c);
        return;
    }
#endif
    /* Portable build / older CPU: still a usable entry, never illegal SIMD. */
    matmul_blocked(m, n, k, a, b, c);
}
