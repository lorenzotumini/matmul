#include "lab.h"

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void) {
    puts("Usage: matmul_bench [options]\n"
         "  --list                 List compiled implementations\n"
         "  --impl NAME            Implementation name or all (default: all)\n"
         "  --m M --n N --k K      A[M,K] * B[K,N] (default: 256 each)\n"
         "  --repeats R            Measured samples (default: 7)\n"
         "  --warmup W             Untimed calls (default: 2)\n"
         "  --batch B              Calls per measured sample (default: 1)\n"
         "  --seed S               uint32 input seed (default: 42)\n"
         "  --csv                  Write CSV to stdout; metadata to stderr\n"
         "  --profile              Repeat one --impl without the oracle/checks;\n"
         "                         run correctness tests separately first\n"
         "  --help                 Show this help");
}

static bool number(const char *s, size_t *value) {
    size_t v = 0;
    if (!*s) return false;
    for (; *s; ++s) {
        if (*s < '0' || *s > '9') return false;
        size_t digit = (size_t)(*s - '0');
        if (v > (SIZE_MAX - digit) / 10) return false;
        v = v * 10 + digit;
    }
    *value = v;
    return true;
}

static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static bool profile(const MatmulImplementation *impl, LabProblem *p,
                    size_t repeats, size_t warmup, size_t batch) {
    fprintf(stderr, "Profiling workload: no reference validation. Run tests first.\n");
    lab_poison(p);
    for (size_t i = 0; i < warmup; ++i)
        impl->run(p->m, p->n, p->k, p->a, p->b, p->c);

    /* One uninterrupted region: no per-call timers, checks, allocation or I/O
     * from the runner. Whole-process profilers still see setup and cleanup.
     */
    double start = lab_seconds();
    for (size_t i = 0; i < repeats; ++i)
        for (size_t j = 0; j < batch; ++j)
            impl->run(p->m, p->n, p->k, p->a, p->b, p->c);
    double elapsed = lab_seconds() - start;

    double checksum = 0.0;
    for (size_t i = 0; i < p->m * p->n; ++i) {
        if (!isfinite(p->c[i])) {
            fprintf(stderr, "Profile output is nonfinite or unfinished. "
                            "Complete the kernel and run its correctness tests.\n");
            return false;
        }
        checksum += p->c[i];
    }
    printf("profile impl=%s calls=%zu elapsed_seconds=%.6f checksum=%.17g\n",
           impl->name, repeats * batch, elapsed, checksum);
    return true;
}

static bool measure(const MatmulImplementation *impl, LabProblem *p,
                    size_t repeats, size_t warmup, size_t batch,
                    uint32_t seed, bool csv) {
    lab_poison(p);
    impl->run(p->m, p->n, p->k, p->a, p->b, p->c);
    if (!lab_check(p, impl->name)) {
        fprintf(stderr, "No performance result: implement/fix this kernel first.\n");
        return false;
    }
    double *samples = malloc(repeats * sizeof *samples);
    if (!samples) {
        fprintf(stderr, "Cannot allocate timing samples.\n");
        return false;
    }
    for (size_t i = 0; i < warmup; ++i) {
        impl->run(p->m, p->n, p->k, p->a, p->b, p->c);
    }
    for (size_t i = 0; i < repeats; ++i) {
        double start = lab_seconds();
        for (size_t j = 0; j < batch; ++j) {
            impl->run(p->m, p->n, p->k, p->a, p->b, p->c);
        }
        samples[i] = (lab_seconds() - start) / (double)batch;
        /* Validation, the oracle, and allocation are outside the timer. */
        if (samples[i] <= 0.0 || !lab_check(p, impl->name)) {
            fprintf(stderr, "Invalid sample (for tiny matrices, increase --batch).\n");
            free(samples);
            return false;
        }
    }
    qsort(samples, repeats, sizeof *samples, compare_double);
    double median = samples[repeats / 2];
    if (repeats % 2 == 0) median = (median + samples[repeats / 2 - 1]) / 2;
    double gflops = 2.0 * (double)p->m * (double)p->n * (double)p->k /
                    median / 1e9;
    double checksum = 0.0;
    for (size_t i = 0; i < p->m * p->n; ++i) checksum += p->c[i];
    if (csv) {
        printf("%s,%s,%s,%zu,%zu,%zu,%" PRIu32 ",%zu,%zu,%zu,%.9g,%.9g,%.9g,%.9g,%.17g\n",
               impl->name, LAB_COMPILER, LAB_BUILD, p->m, p->n, p->k, seed,
               repeats, warmup, batch, samples[0] * 1e3, median * 1e3,
               samples[repeats - 1] * 1e3, gflops, checksum);
    } else {
        printf("%-12s min/median/max: %.4f / %.4f / %.4f ms; "
               "%.3f GFLOP/s; checksum %.9g\n",
               impl->name, samples[0] * 1e3, median * 1e3,
               samples[repeats - 1] * 1e3, gflops, checksum);
    }
    free(samples);
    return true;
}

int main(int argc, char **argv) {
    size_t m = 256, n = 256, k = 256, repeats = 7, warmup = 2, batch = 1;
    size_t seed = 42;
    const char *selected = "all";
    bool csv = false, profiling = false;
    size_t count;
    const MatmulImplementation *impls = matmul_implementations(&count);
    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (strcmp(arg, "--help") == 0) { usage(); return 0; }
        if (strcmp(arg, "--list") == 0) {
            for (size_t j = 0; j < count; ++j)
                printf("%s: %s\n", impls[j].name, impls[j].description);
            return 0;
        }
        if (strcmp(arg, "--csv") == 0) { csv = true; continue; }
        if (strcmp(arg, "--profile") == 0) { profiling = true; continue; }
        if (strcmp(arg, "--impl") == 0) {
            if (++i >= argc) { fprintf(stderr, "Missing implementation name.\n"); return 1; }
            selected = argv[i];
            continue;
        }
        size_t *destination = NULL;
        if (strcmp(arg, "--m") == 0) destination = &m;
        else if (strcmp(arg, "--n") == 0) destination = &n;
        else if (strcmp(arg, "--k") == 0) destination = &k;
        else if (strcmp(arg, "--repeats") == 0) destination = &repeats;
        else if (strcmp(arg, "--warmup") == 0) destination = &warmup;
        else if (strcmp(arg, "--batch") == 0) destination = &batch;
        else if (strcmp(arg, "--seed") == 0) destination = &seed;
        if (!destination || ++i >= argc || !number(argv[i], destination)) {
            fprintf(stderr, "Invalid option/value: %s. Use --help.\n", arg);
            return 1;
        }
    }
    if (!m || !n || !k || !repeats || !batch || seed > UINT32_MAX ||
        repeats > SIZE_MAX / sizeof(double) || (double)k * FLT_EPSILON >= 0.01) {
        fprintf(stderr, "Dimensions, repeats and batch must be positive and fit "
                        "supported ranges; seed must fit uint32; K must be < 83887.\n");
        return 1;
    }
    bool found = strcmp(selected, "all") == 0;
    for (size_t i = 0; i < count; ++i)
        if (strcmp(selected, impls[i].name) == 0) found = true;
    if (!found) { fprintf(stderr, "Unknown implementation: %s\n", selected); return 1; }
    if (profiling && (csv || strcmp(selected, "all") == 0 || repeats > SIZE_MAX / batch)) {
        fprintf(stderr, "--profile requires one named --impl, no --csv, "
                        "and repeats * batch fitting size_t.\n");
        return 1;
    }

    LabProblem p;
    bool allocated = profiling ? lab_create_workload(&p, m, n, k) :
                                 lab_create(&p, m, n, k);
    if (!allocated) {
        fprintf(stderr, "Matrix dimensions overflow or allocation failed.\n");
        return 1;
    }
    lab_randomize(&p, (uint32_t)seed);
    if (!profiling) lab_reference(&p);
    fprintf(stderr, "mode=%s compiler=%s build=%s M=%zu N=%zu K=%zu seed=%zu "
                    "repeats=%zu warmup=%zu batch=%zu\n",
            profiling ? "profile" : "benchmark", LAB_COMPILER, LAB_BUILD,
            m, n, k, seed, repeats, warmup, batch);
    if (csv) puts("implementation,compiler,build,m,n,k,seed,repeats,warmup,batch,"
                  "min_ms,median_ms,max_ms,gflops,checksum");
    bool ok = true;
    for (size_t i = 0; i < count; ++i) {
        if (strcmp(selected, "all") != 0 && strcmp(selected, impls[i].name) != 0)
            continue;
        bool passed = profiling ? profile(&impls[i], &p, repeats, warmup, batch) :
                                  measure(&impls[i], &p, repeats, warmup, batch,
                                          (uint32_t)seed, csv);
        if (!passed)
            ok = false;
    }
    lab_destroy(&p);
    return ok ? 0 : 1;
}
