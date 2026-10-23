#include "matmul.h"

#ifdef MATMUL_HAVE_AVX
#include <stdbool.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif

/* This file is compiled WITHOUT AVX. Check both CPU capability and OS support
 * for saving/restoring XMM and YMM registers before exposing the AVX kernel.
 */
static bool avx_available(void) {
#ifdef _MSC_VER
    int features[4];
    __cpuidex(features, 1, 0);
    const unsigned int required = (1u << 26) | (1u << 27) | (1u << 28);
    if (((unsigned int)features[2] & required) != required) return false;
    return (_xgetbv(0) & 6) == 6;
#else
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx") != 0;
#endif
}
#endif

const MatmulImplementation *matmul_implementations(size_t *count) {
    static const MatmulImplementation implementations[] = {
        {"naive", matmul_naive, "Direct definition, row-major float loops"},
        {"loop_interchange", matmul_loop_interchange, "Loop interchange to reduce cache misses"},
        {"blocked", matmul_blocked, "Cache blocking, 32-element tiles"},
#if defined(__x86_64__) || defined(_M_X64)
        {"simd_sse", matmul_simd_sse, "Four-float SSE intrinsics"},
#endif
#ifdef MATMUL_HAVE_AVX
        /* Keep this optional entry last: unsupported hosts omit it by count. */
        {"simd_avx", matmul_simd_avx, "Eight-float AVX intrinsics (exercise)"},
#endif
    };
    *count = sizeof implementations / sizeof implementations[0];
#ifdef MATMUL_HAVE_AVX
    if (!avx_available()) --*count;
#endif
    return implementations;
}
