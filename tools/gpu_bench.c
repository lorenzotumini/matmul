#include "gpu_bench.h"
#include <stdlib.h>
#ifdef MATMUL_HAVE_CUDA
#include "cuda_bench.h"
#endif
#ifdef MATMUL_HAVE_OPENCL
#include "opencl_bench.h"
#endif

struct LabGpuBenchmark {
    enum { GPU_CUDA, GPU_OPENCL } backend;
    void *handle;
};

bool lab_gpu_kernel_timing_supported(MatmulFn implementation) {
    (void)implementation;
#ifdef MATMUL_HAVE_CUDA
    if (matmul_cuda_kernel_timing_supported(implementation)) return true;
#endif
#ifdef MATMUL_HAVE_OPENCL
    if (matmul_opencl_kernel_timing_supported(implementation)) return true;
#endif
    return false;
}

LabGpuBenchmark *lab_gpu_benchmark_create(MatmulFn implementation,
    size_t m, size_t n, size_t k, const float *a, const float *b) {
    if (!lab_gpu_kernel_timing_supported(implementation) || !m || !n || !k) return NULL;
    LabGpuBenchmark *device = malloc(sizeof *device);
    if (!device) return NULL;
    device->handle = NULL;
    (void)a; (void)b;
#ifdef MATMUL_HAVE_CUDA
    if (matmul_cuda_kernel_timing_supported(implementation)) {
        device->backend = GPU_CUDA;
        device->handle = matmul_cuda_benchmark_create(implementation, m, n, k, a, b);
    }
#endif
#ifdef MATMUL_HAVE_OPENCL
    if (matmul_opencl_kernel_timing_supported(implementation)) {
        device->backend = GPU_OPENCL;
        device->handle = matmul_opencl_benchmark_create(implementation, m, n, k, a, b);
    }
#endif
    if (!device->handle) { free(device); return NULL; }
    return device;
}

double lab_gpu_benchmark_sample(LabGpuBenchmark *device, size_t batch) {
    (void)device; (void)batch;
#ifdef MATMUL_HAVE_CUDA
    if (device->backend == GPU_CUDA) return matmul_cuda_benchmark_sample(device->handle, batch);
#endif
#ifdef MATMUL_HAVE_OPENCL
    if (device->backend == GPU_OPENCL) return matmul_opencl_benchmark_sample(device->handle, batch);
#endif
    abort(); /* A context can only be created for a compiled backend. */
}
void lab_gpu_benchmark_read(LabGpuBenchmark *device, float *c) {
    (void)device; (void)c;
#ifdef MATMUL_HAVE_CUDA
    if (device->backend == GPU_CUDA) matmul_cuda_benchmark_read(device->handle, c);
#endif
#ifdef MATMUL_HAVE_OPENCL
    if (device->backend == GPU_OPENCL) matmul_opencl_benchmark_read(device->handle, c);
#endif
}
void lab_gpu_benchmark_destroy(LabGpuBenchmark *device) {
    if (!device) return;
#ifdef MATMUL_HAVE_CUDA
    if (device->backend == GPU_CUDA) matmul_cuda_benchmark_destroy(device->handle);
#endif
#ifdef MATMUL_HAVE_OPENCL
    if (device->backend == GPU_OPENCL) matmul_opencl_benchmark_destroy(device->handle);
#endif
    free(device);
}
