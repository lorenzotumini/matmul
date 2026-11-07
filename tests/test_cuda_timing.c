#include "lab.h"
#include "cuda_bench.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define REQUIRE(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        return false; \
    } \
} while (0)

static bool prepared_case(MatmulFn implementation, size_t m, size_t n, size_t k) {
    LabProblem p;
    REQUIRE(lab_create(&p, m, n, k));
    lab_randomize(&p, 42);
    lab_reference(&p);
    MatmulCudaBenchmark *device = matmul_cuda_benchmark_create(
        implementation, m, n, k, p.a, p.b);
    REQUIRE(device != NULL);
    /* Prepared inputs must remain the original data even if the host changes.
     * Keep the original double reference/magnitude to check every sample.
     */
    for (size_t i = 0; i < m*k; ++i) p.a[i] = 0.0f;
    for (size_t i = 0; i < k*n; ++i) p.b[i] = 0.0f;
    const size_t batches[] = {1, 4, 2};
    for (size_t i = 0; i < sizeof batches / sizeof batches[0]; ++i) {
        double seconds = matmul_cuda_benchmark_sample(device, batches[i]);
        REQUIRE(isfinite(seconds) && seconds > 0.0);
        lab_poison(&p);
        matmul_cuda_benchmark_read(device, p.c);
        REQUIRE(lab_check(&p, "CUDA prepared timing"));
    }
    matmul_cuda_benchmark_destroy(device);
    lab_destroy(&p);
    return true;
}

static bool checks(void) {
    float dummy = 1.0f;
    REQUIRE(!matmul_cuda_kernel_timing_supported(matmul_naive));
    REQUIRE(matmul_cuda_benchmark_create(matmul_naive, 1, 1, 1, &dummy, &dummy) == NULL);
    REQUIRE(matmul_cuda_benchmark_create(matmul_cuda_naive, 0, 1, 1, &dummy, &dummy) == NULL);
    REQUIRE(matmul_cuda_benchmark_create(matmul_cuda_tiled, 1, 1, 0, &dummy, &dummy) == NULL);
    REQUIRE(matmul_cuda_kernel_timing_supported(matmul_cuda_register_tiled));
    REQUIRE(matmul_cuda_benchmark_create(matmul_cuda_register_tiled, 1, 0, 1, &dummy, &dummy) == NULL);
    REQUIRE(matmul_cuda_kernel_timing_supported(matmul_cuda_coalesced));
    REQUIRE(matmul_cuda_benchmark_create(matmul_cuda_coalesced, 0, 1, 1, &dummy, &dummy) == NULL);
    matmul_cuda_benchmark_destroy(NULL);
    const size_t shapes[][3] = {{1, 1, 1}, {17, 31, 33}, {33, 17, 65}, {129, 257, 259}};
    /* Completed baselines only here. The unfinished coalescing exercise
     * has its own registered correctness test, which must fail until completed.
     */
    const MatmulFn implementations[] = {matmul_cuda_naive, matmul_cuda_tiled,
                                      matmul_cuda_register_tiled};
    for (size_t i = 0; i < sizeof implementations / sizeof implementations[0]; ++i) {
        REQUIRE(matmul_cuda_kernel_timing_supported(implementations[i]));
        for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; ++s)
            REQUIRE(prepared_case(implementations[i], shapes[s][0], shapes[s][1], shapes[s][2]));
    }
    return true;
}

int main(void) {
    if (!matmul_cuda_available()) {
        puts("SKIP CUDA timing (no CUDA device/runtime available)");
        return 77;
    }
    if (!checks()) return EXIT_FAILURE;
    puts("PASS CUDA event timing (completed kernels, resident inputs, repeated samples and batches)");
    return EXIT_SUCCESS;
}
