#ifndef CPU_COMBINED_INTERNAL_H
#define CPU_COMBINED_INTERNAL_H

#include "matmul.h"

/* Private paths of one implementation, compiled with their own ISA flags.
 * The baseline dispatcher checks CPU/OS support before entering either one.
 */
void matmul_cpu_direct(size_t m, size_t n, size_t k,
                       const float *a, const float *b, float *c);
void matmul_cpu_packed(size_t m, size_t n, size_t k,
                       const float *a, const float *b, float *c);

#endif
