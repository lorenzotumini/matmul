#define _POSIX_C_SOURCE 200809L
#include "lab.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <time.h>
#endif

static bool product(size_t x, size_t y, size_t *out) {
    if (y != 0 && x > SIZE_MAX / y) return false;
    *out = x * y;
    return true;
}

static void *array(size_t count, size_t element_size) {
    size_t bytes;
    if (!product(count ? count : 1, element_size, &bytes)) return NULL;
    return calloc(1, bytes);
}

static bool create_problem(LabProblem *p, size_t m, size_t n, size_t k,
                           bool with_reference) {
    size_t mk, kn, mn;
    *p = (LabProblem){0};
    if (!product(m, k, &mk) || !product(k, n, &kn) ||
        !product(m, n, &mn)) return false;
    if (mk > SIZE_MAX / sizeof(float) || kn > SIZE_MAX / sizeof(float) ||
        mn > SIZE_MAX / sizeof(float) ||
        (with_reference && mn > SIZE_MAX / sizeof(double)))
        return false;
    p->m = m;
    p->n = n;
    p->k = k;
    p->a = array(mk, sizeof *p->a);
    p->b = array(kn, sizeof *p->b);
    p->c = array(mn, sizeof *p->c);
    if (with_reference) {
        p->reference = array(mn, sizeof *p->reference);
        p->magnitude = array(mn, sizeof *p->magnitude);
    }
    if (!p->a || !p->b || !p->c ||
        (with_reference && (!p->reference || !p->magnitude))) {
        lab_destroy(p);
        return false;
    }
    return true;
}

bool lab_create(LabProblem *p, size_t m, size_t n, size_t k) {
    return create_problem(p, m, n, k, true);
}

bool lab_create_workload(LabProblem *p, size_t m, size_t n, size_t k) {
    return create_problem(p, m, n, k, false);
}

void lab_destroy(LabProblem *p) {
    free(p->a);
    free(p->b);
    free(p->c);
    free(p->reference);
    free(p->magnitude);
    *p = (LabProblem){0};
}

static float random_value(uint32_t *state) {
    /* A defined uint32_t sequence, independent of libc rand(). */
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return (float)(*state >> 8) / 8388608.0f - 1.0f;
}

void lab_randomize(LabProblem *p, uint32_t seed) {
    for (size_t i = 0; i < p->m * p->k; ++i) p->a[i] = random_value(&seed);
    for (size_t i = 0; i < p->k * p->n; ++i) p->b[i] = random_value(&seed);
}

void lab_reference(LabProblem *p) {
    /* Test oracle: double products AND double accumulation. Not a contender. */
    for (size_t row = 0; row < p->m; ++row) {
        for (size_t col = 0; col < p->n; ++col) {
            double sum = 0.0, magnitude = 0.0;
            for (size_t inner = 0; inner < p->k; ++inner) {
                double term = (double)p->a[row * p->k + inner] *
                              (double)p->b[inner * p->n + col];
                sum += term;
                magnitude += fabs(term);
            }
            p->reference[row * p->n + col] = sum;
            p->magnitude[row * p->n + col] = magnitude;
        }
    }
}

void lab_poison(LabProblem *p) {
    for (size_t i = 0; i < p->m * p->n; ++i) p->c[i] = NAN;
}

bool lab_check(const LabProblem *p, const char *name) {
    /* A conservative bound for ordinary float dot products, including
     * cancellation. For very long reductions this bound stops being useful.
     * This lab starts with finite inputs in [-1,1] and modest K.
     */
    double ku = (double)p->k * FLT_EPSILON;
    if (ku >= 0.01) {
        fprintf(stderr, "Validation requires K * FLT_EPSILON < 0.01.\n");
        return false;
    }
    double gamma = ku / (1.0 - ku);
    for (size_t i = 0; i < p->m * p->n; ++i) {
        double got = p->c[i], want = p->reference[i];
        double tolerance = 1e-6 + 2.0 * gamma * p->magnitude[i];
        if (!isfinite(got) || !isfinite(want) ||
            !isfinite(p->magnitude[i]) || fabs(got - want) > tolerance) {
            fprintf(stderr,
                    "%s: C[%zu,%zu]: got %.9g, expected %.17g, tolerance %.3g\n",
                    name, i / p->n, i % p->n, got, want, tolerance);
            return false;
        }
    }
    return true;
}

double lab_seconds(void) {
#ifdef _WIN32
    LARGE_INTEGER frequency, counter;
    if (!QueryPerformanceFrequency(&frequency) ||
        !QueryPerformanceCounter(&counter)) {
        fprintf(stderr, "High-resolution timer failed.\n");
        exit(EXIT_FAILURE);
    }
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
#endif
}
