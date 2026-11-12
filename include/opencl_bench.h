#ifndef OPENCL_BENCH_H
#define OPENCL_BENCH_H
#include "matmul.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Optional, single-host-thread API. Targets the first available Intel GPU,
 * never an OpenCL CPU device or a different vendor. Runtime errors are fatal.
 */
typedef struct MatmulOpenCLBenchmark MatmulOpenCLBenchmark;
const char *matmul_opencl_device_name(void);
const char *matmul_opencl_driver_version(void);
bool matmul_opencl_host_unified_memory(void);
const char *matmul_opencl_combined_strategy(size_t m, size_t n, size_t k);
#ifdef MATMUL_HAVE_CLBLAST
const char *matmul_clblast_version(void);
const char *matmul_clblast_build_options(void);
#endif
bool matmul_opencl_kernel_timing_supported(MatmulFn implementation);
/* NULL for unsupported functions, empty shapes or host OOM. */
MatmulOpenCLBenchmark *matmul_opencl_benchmark_create(MatmulFn implementation,
    size_t m, size_t n, size_t k, const float *a, const float *b);
/* Custom kernels: first START to last END / batch, in device seconds.
 * CLBlast: enclosing marker END to marker END, covering its entire operation.
 * Both include intervening enqueue/scheduling gaps; fill/copies are excluded.
 */
double matmul_opencl_benchmark_sample(MatmulOpenCLBenchmark *context, size_t batch);
void matmul_opencl_benchmark_read(MatmulOpenCLBenchmark *context, float *c);
void matmul_opencl_benchmark_destroy(MatmulOpenCLBenchmark *context);
#ifdef __cplusplus
}
#endif
#endif
