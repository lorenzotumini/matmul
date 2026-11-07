#ifndef LAB_GPU_BENCH_H
#define LAB_GPU_BENCH_H
#include "matmul.h"

/* Runner adapter: CUDA and OpenCL keep their native timing implementations. */
typedef struct LabGpuBenchmark LabGpuBenchmark;
bool lab_gpu_kernel_timing_supported(MatmulFn implementation);
LabGpuBenchmark *lab_gpu_benchmark_create(MatmulFn implementation,
    size_t m, size_t n, size_t k, const float *a, const float *b);
double lab_gpu_benchmark_sample(LabGpuBenchmark *device, size_t batch);
void lab_gpu_benchmark_read(LabGpuBenchmark *device, float *c);
void lab_gpu_benchmark_destroy(LabGpuBenchmark *device);
#endif
