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
bool matmul_opencl_kernel_timing_supported(MatmulFn implementation);
/* NULL for unsupported functions, empty shapes or host OOM. */
MatmulOpenCLBenchmark *matmul_opencl_benchmark_create(MatmulFn implementation,
    size_t m, size_t n, size_t k, const float *a, const float *b);
/* First kernel START to last kernel END / batch, in device seconds. Includes
 * intervening scheduling/launch gaps. Fill and transfers are outside the window.
 */
double matmul_opencl_benchmark_sample(MatmulOpenCLBenchmark *context, size_t batch);
void matmul_opencl_benchmark_read(MatmulOpenCLBenchmark *context, float *c);
void matmul_opencl_benchmark_destroy(MatmulOpenCLBenchmark *context);
#ifdef __cplusplus
}
#endif
#endif
