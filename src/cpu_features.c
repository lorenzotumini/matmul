#include "cpu_features.h"
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h> /* Interlocked operations work without experimental C atomics. */
#else
#include <stdatomic.h>
#endif
#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#elif defined(__x86_64__)
#include <cpuid.h>
#endif

unsigned matmul_decode_cpu_features(unsigned ecx1, unsigned ebx7, uint64_t xcr0) {
    unsigned avx_bits = (1u << 26) | (1u << 27) | (1u << 28);
    if ((ecx1 & avx_bits) != avx_bits || (xcr0 & 6u) != 6u) return 0;
    unsigned result = CPU_AVX;
    if (ecx1 & (1u << 12)) result |= CPU_FMA;
    if (ebx7 & (1u << 5)) result |= CPU_AVX2;
    if ((ebx7 & (1u << 16)) && (xcr0 & 0xe6u) == 0xe6u) {
        result |= CPU_AVX512F;
        /* MSVC /arch:AVX512 permits F, DQ, CD, BW and VL (plus AVX2/FMA). */
        unsigned full = (1u << 17) | (1u << 28) | (1u << 30) | (1u << 31);
        if ((ebx7 & full) == full && (result & (CPU_FMA | CPU_AVX2)) == (CPU_FMA | CPU_AVX2))
            result |= CPU_AVX512_MSVC;
    }
    return result;
}

static unsigned detect_cpu_features(void) {
    unsigned ecx1 = 0, ebx7 = 0;
    uint64_t xcr0 = 0;
#if defined(_MSC_VER) && defined(_M_X64)
    int r[4];
    __cpuidex(r, 1, 0);
    ecx1 = (unsigned)r[2];
    if ((ecx1 & ((1u << 26) | (1u << 27))) == ((1u << 26) | (1u << 27)))
        xcr0 = _xgetbv(0);
    __cpuidex(r, 0, 0);
    if (r[0] >= 7) { __cpuidex(r, 7, 0); ebx7 = (unsigned)r[1]; }
#elif defined(__x86_64__)
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return 0;
    ecx1 = c;
    if ((ecx1 & ((1u << 26) | (1u << 27))) == ((1u << 26) | (1u << 27))) {
        unsigned low, high;
        __asm__ volatile ("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
        xcr0 = ((uint64_t)high << 32) | low;
    }
    if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) ebx7 = b;
#endif
    return matmul_decode_cpu_features(ecx1, ebx7, xcr0);
}

unsigned matmul_cpu_features(void) {
    /* CPUID is costly compared with a tiny matrix product. Cache only immutable
     * CPU/OS capabilities, safely even if callers race on the first query.
     * The high bit distinguishes "initialized, no features" from "not queried".
     */
    const unsigned initialized = 1u << 31;
#if defined(_MSC_VER) && !defined(__clang__)
    static volatile long cached = 0;
    unsigned features = (unsigned)_InterlockedCompareExchange(&cached, 0, 0);
#else
    static atomic_uint cached = 0;
    unsigned features = atomic_load_explicit(&cached, memory_order_relaxed);
#endif
    if (!features) {
        features = initialized | detect_cpu_features();
#if defined(_MSC_VER) && !defined(__clang__)
        _InterlockedExchange(&cached, (long)features);
#else
        atomic_store_explicit(&cached, features, memory_order_relaxed);
#endif
    }
    return features & ~initialized;
}

bool matmul_has_avx(void) { return (matmul_cpu_features() & CPU_AVX) != 0; }
bool matmul_has_fma(void) {
    unsigned required = CPU_AVX | CPU_FMA;
#ifdef _MSC_VER
    required |= CPU_AVX2; /* /arch:AVX2 is the MSVC FMA target. */
#endif
    return (matmul_cpu_features() & required) == required;
}
bool matmul_has_avx512(void) {
    unsigned required = CPU_AVX | CPU_FMA | CPU_AVX2 | CPU_AVX512F;
#ifdef _MSC_VER
    required |= CPU_AVX512_MSVC;
#endif
    return (matmul_cpu_features() & required) == required;
}
