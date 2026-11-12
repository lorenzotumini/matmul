#include "lab.h"
#include "gpu_bench.h"
#include "opencl_bench.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static MatmulFn selected = matmul_opencl_naive;

#define REQUIRE(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        return false; \
    } \
} while (0)

static bool prepared_case(size_t m, size_t n, size_t k) {
    LabProblem p;
    REQUIRE(lab_create(&p, m, n, k));
    lab_randomize(&p, 42);
    lab_reference(&p);
    LabGpuBenchmark *device = lab_gpu_benchmark_create(selected,
        m, n, k, p.a, p.b);
    REQUIRE(device != NULL);
    /* The prepared copy must survive host mutation. This checks the common
     * benchmark adapter too; the ordinary 61 fixtures check the host wrapper.
     */
    for (size_t i = 0; i < m*k; ++i) p.a[i] = 0.0f;
    for (size_t i = 0; i < k*n; ++i) p.b[i] = 0.0f;
    bool ok = true;
    const size_t batches[] = {1, 4, 2};
    for (size_t i = 0; i < sizeof batches / sizeof batches[0]; ++i) {
        double seconds = lab_gpu_benchmark_sample(device, batches[i]);
        lab_poison(&p);
        lab_gpu_benchmark_read(device, p.c);
        if (!isfinite(seconds) || seconds <= 0 || !lab_check(&p, "OpenCL prepared timing")) {
            ok = false;
            break;
        }
    }
    lab_gpu_benchmark_destroy(device);
    lab_destroy(&p);
    return ok;
}

static bool checks(void) {
    float dummy = 1;
    REQUIRE(!lab_gpu_kernel_timing_supported(matmul_naive));
    REQUIRE(lab_gpu_kernel_timing_supported(selected));
    REQUIRE(matmul_opencl_benchmark_create(matmul_naive, 1, 1, 1, &dummy, &dummy) == NULL);
    REQUIRE(matmul_opencl_benchmark_create(selected, 0, 1, 1, &dummy, &dummy) == NULL);
    REQUIRE(matmul_opencl_benchmark_create(selected, 1, 1, 0, &dummy, &dummy) == NULL);
    lab_gpu_benchmark_destroy(NULL);
    /* The host suite covers singleton/tiny correctness. Device timing uses
     * enough work to avoid zero-duration timestamps on the Intel driver.
     */
    const size_t shapes[][3] = {{16,16,16}, {17,31,33}, {33,17,65}, {65,63,129}, {129,257,259}};
    for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; ++i)
        REQUIRE(prepared_case(shapes[i][0], shapes[i][1], shapes[i][2]));
    if (selected == matmul_opencl_combined) {
        REQUIRE(prepared_case(256,256,129)); /* 64x64 path. */
        REQUIRE(prepared_case(255,257,129)); /* 32x64 path below area threshold. */
        REQUIRE(prepared_case(257,257,129)); /* 64x64 path with all tails. */
        REQUIRE(prepared_case(513,515,129)); /* Large path and all tails. */
        REQUIRE(prepared_case(32,513,129)); /* Short-wide path. */
    }
    return true;
}
int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--tiled") == 0) selected = matmul_opencl_tiled;
    else if (argc == 2 && strcmp(argv[1], "--register") == 0) selected = matmul_opencl_register_tiled;
    else if (argc == 2 && strcmp(argv[1], "--combined") == 0) selected = matmul_opencl_combined;
#ifdef MATMUL_HAVE_CLBLAST
    else if (argc == 2 && strcmp(argv[1], "--clblast") == 0) selected = matmul_clblast;
#endif
    else if (argc != 1) return EXIT_FAILURE;
    if (!matmul_opencl_available()) {
        puts("SKIP OpenCL timing (no available Intel GPU/compiler)");
        return 77;
    }
    if (!checks()) return EXIT_FAILURE;
    puts("PASS OpenCL event timing (resident inputs, repeated samples and batches)");
    return EXIT_SUCCESS;
}
