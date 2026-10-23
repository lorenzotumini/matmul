#include "lab.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Explicit checks still run in Release builds (unlike assert under NDEBUG). */
#define REQUIRE(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        return false; \
    } \
} while (0)

static bool harness(void) {
    LabProblem p;
    REQUIRE(!lab_create(&p, SIZE_MAX, 2, 2));
    REQUIRE(!lab_create(&p, 1, SIZE_MAX / sizeof(double) + 1, 1));
    REQUIRE(!lab_create_workload(&p, SIZE_MAX, 2, 2));
    REQUIRE(lab_create_workload(&p, 2, 3, 4));
    REQUIRE(p.a && p.b && p.c && !p.reference && !p.magnitude);
    lab_destroy(&p);
    REQUIRE(lab_create(&p, 2, 2, 3));
    const float a[] = {1, 2, 3, 4, 5, 6};
    const float b[] = {7, 8, 9, 10, 11, 12};
    const double expected[] = {58, 64, 139, 154};
    memcpy(p.a, a, sizeof a);
    memcpy(p.b, b, sizeof b);
    lab_reference(&p);
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(p.reference[i] == expected[i]);
        p.c[i] = (float)expected[i];
    }
    REQUIRE(lab_check(&p, "known answer"));
    p.c[0] += 1.0f;
    REQUIRE(!lab_check(&p, "expected rejection: wrong answer"));
    p.c[0] = INFINITY;
    REQUIRE(!lab_check(&p, "expected rejection: infinity"));
    lab_poison(&p);
    REQUIRE(!lab_check(&p, "expected rejection: NaN/unwritten output"));
    lab_randomize(&p, 42);
    float first = p.a[0];
    lab_randomize(&p, 42);
    REQUIRE(first == p.a[0]);
    lab_destroy(&p);
    REQUIRE(p.a == NULL && p.c == NULL);
    puts("PASS harness (the rejection diagnostics above are expected)");
    return true;
}

typedef enum { RANDOM, ZERO, ONES, LEFT_IDENTITY, RIGHT_IDENTITY, CANCEL } Pattern;

static bool kernel_case(const MatmulImplementation *impl,
                        size_t m, size_t n, size_t k, Pattern pattern,
                        uint32_t seed) {
    LabProblem p;
    REQUIRE(lab_create(&p, m, n, k));
    lab_randomize(&p, seed);
    if (pattern == ZERO) {
        for (size_t i = 0; i < m * k; ++i) p.a[i] = 0.0f;
    } else if (pattern == ONES || pattern == CANCEL) {
        for (size_t i = 0; i < m * k; ++i) p.a[i] = 1.0f;
        for (size_t inner = 0; inner < k; ++inner)
            for (size_t col = 0; col < n; ++col)
                p.b[inner * n + col] = pattern == CANCEL && inner % 2 ? -1.0f : 1.0f;
    } else if (pattern == LEFT_IDENTITY) {
        for (size_t row = 0; row < m; ++row)
            for (size_t col = 0; col < k; ++col)
                p.a[row * k + col] = row == col ? 1.0f : 0.0f;
    } else if (pattern == RIGHT_IDENTITY) {
        for (size_t row = 0; row < k; ++row)
            for (size_t col = 0; col < n; ++col)
                p.b[row * n + col] = row == col ? 1.0f : 0.0f;
    }
    lab_reference(&p);
    size_t a_bytes = m * k * sizeof(float), b_bytes = k * n * sizeof(float);
    float *a_copy = malloc(a_bytes ? a_bytes : sizeof(float));
    float *b_copy = malloc(b_bytes ? b_bytes : sizeof(float));
    if (!a_copy || !b_copy) {
        free(a_copy); free(b_copy); lab_destroy(&p);
        REQUIRE(false);
    }
    memcpy(a_copy, p.a, a_bytes);
    memcpy(b_copy, p.b, b_bytes);
    bool ok = true;
    for (int call = 0; call < 2; ++call) {
        if (call == 0) lab_poison(&p);
        else for (size_t i = 0; i < m * n; ++i) p.c[i] = 17.0f;
        impl->run(m, n, k, p.a, p.b, p.c);
        if (!lab_check(&p, impl->name) || memcmp(a_copy, p.a, a_bytes) != 0 ||
            memcmp(b_copy, p.b, b_bytes) != 0) {
            fprintf(stderr, "case M=%zu N=%zu K=%zu pattern=%d seed=%u call=%d\n",
                    m, n, k, (int)pattern, (unsigned)seed, call + 1);
            ok = false;
            break;
        }
    }
    free(a_copy);
    free(b_copy);
    lab_destroy(&p);
    return ok;
}

static bool kernel(const MatmulImplementation *impl) {
    /* Start with a small fixture to make basic indexing failures easy to trace. */
    REQUIRE(kernel_case(impl, 2, 2, 3, ONES, 1));
    const size_t shapes[][3] = {
        /* M, N, K: singleton, rectangular, vector/tile tails, empty axes */
        {1, 1, 1}, {2, 4, 3}, {1, 17, 9}, {13, 1, 7}, {7, 11, 1},
        {7, 9, 15}, {16, 16, 16}, {17, 31, 33}, {33, 17, 65},
        {65, 63, 129}, {5, 259, 263},
        {0, 7, 3}, {4, 0, 3}, {4, 7, 0}, {0, 0, 0}
    };
    for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; ++i)
        for (uint32_t seed = 0; seed < 3; ++seed)
            REQUIRE(kernel_case(impl, shapes[i][0], shapes[i][1], shapes[i][2],
                                RANDOM, seed));
    REQUIRE(kernel_case(impl, 5, 7, 9, ZERO, 42));
    REQUIRE(kernel_case(impl, 5, 7, 9, ONES, 42));
    REQUIRE(kernel_case(impl, 7, 11, 7, LEFT_IDENTITY, 42));
    REQUIRE(kernel_case(impl, 7, 11, 11, RIGHT_IDENTITY, 42));
    REQUIRE(kernel_case(impl, 3, 5, 128, CANCEL, 42));
    /* Cross all combined-kernel tile boundaries and its parallel threshold. */
    REQUIRE(kernel_case(impl, 129, 257, 257, RANDOM, 42));
    printf("PASS %s (52 fixtures, each called twice)\n", impl->name);
    return true;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--harness") == 0)
        return harness() ? 0 : 1;
    if (argc != 1) return 1;
    size_t count;
    const MatmulImplementation *impls = matmul_implementations(&count);
    bool ok = true;
    for (size_t i = 0; i < count; ++i)
        if (!kernel(&impls[i])) ok = false;
    return ok ? 0 : 1;
}
