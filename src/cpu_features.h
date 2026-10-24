#ifndef CPU_FEATURES_H
#define CPU_FEATURES_H
#include <stdbool.h>
#include <stdint.h>

enum {
    CPU_AVX = 1u, CPU_FMA = 2u, CPU_AVX2 = 4u,
    CPU_AVX512F = 8u, CPU_AVX512_MSVC = 16u
};
/* Kept separate from probing so OS-state and missing-feature cases can be tested. */
unsigned matmul_decode_cpu_features(unsigned ecx1, unsigned ebx7, uint64_t xcr0);
unsigned matmul_cpu_features(void);
bool matmul_has_avx(void);
bool matmul_has_fma(void);
bool matmul_has_avx512(void);
#endif
