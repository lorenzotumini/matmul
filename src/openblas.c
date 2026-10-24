#include "matmul.h"
#include <cblas.h>
#include <limits.h>
#include <stdint.h>

void matmul_openblas(size_t m, size_t n, size_t k,
                      const float *a, const float *b, float *c) {
    if (!m || !n) return;
    if (!k) {
        for (size_t i = 0; i < m*n; ++i) c[i] = 0.0f;
        return;
    }

    /* OpenBLAS can use 32-bit (LP64) or 64-bit (ILP64) dimension arguments.
     * Keep the size_t kernel contract without silently truncating dimensions.
     * Extremely large dimensions outside this library ABI use the baseline.
     */
    _Static_assert(sizeof(blasint) == 4 || sizeof(blasint) == 8,
                   "Unsupported OpenBLAS integer ABI");
    uintmax_t limit = sizeof(blasint) == 8 ? (uintmax_t)INT64_MAX : (uintmax_t)INT32_MAX;
    if ((uintmax_t)m > limit || (uintmax_t)n > limit || (uintmax_t)k > limit) {
        matmul_loop_interchange(m, n, k, a, b, c);
        return;
    }

    /* Leading dimensions are row strides. beta=0 overwrites C even if its
     * previous contents are NaNs. Internal packing/workspace costs are part
     * of this ordinary synchronous SGEMM call and stay inside the timer.
     */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (blasint)m, (blasint)n, (blasint)k,
                1.0f, a, (blasint)k, b, (blasint)n, 0.0f, c, (blasint)n);
}

/* Metadata is queried by the runner outside measured/captured kernel calls. */
const char *matmul_openblas_config(void) { return openblas_get_config(); }
const char *matmul_openblas_core(void) { return openblas_get_corename(); }
int matmul_openblas_threads(void) { return openblas_get_num_threads(); }
const char *matmul_openblas_threading(void) {
    switch (openblas_get_parallel()) {
    case OPENBLAS_SEQUENTIAL: return "sequential";
    case OPENBLAS_THREAD: return "native-threads";
    case OPENBLAS_OPENMP: return "OpenMP";
    default: return "unknown";
    }
}
