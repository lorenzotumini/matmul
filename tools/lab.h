#ifndef LAB_H
#define LAB_H

#include "matmul.h"
#include <stdbool.h>
#include <stdint.h>

typedef struct {
    size_t m, n, k;
    float *a, *b, *c;
    double *reference, *magnitude;
} LabProblem;

/* On failure the problem is empty and can safely be destroyed. */
bool lab_create(LabProblem *p, size_t m, size_t n, size_t k);
/* Profiling needs only A, B, C; do not allocate or run the double oracle. */
bool lab_create_workload(LabProblem *p, size_t m, size_t n, size_t k);
void lab_destroy(LabProblem *p);
void lab_randomize(LabProblem *p, uint32_t seed);
void lab_reference(LabProblem *p);
void lab_poison(LabProblem *p);
/* First mismatch is printed to stderr, including its matrix coordinates. */
bool lab_check(const LabProblem *p, const char *name);
double lab_seconds(void);

#endif
