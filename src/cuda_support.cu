#include "matmul.h"
#include "cuda_bench.h"
#include "cuda_internal.cuh"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* The C kernel interface has no error return. Runtime errors are fatal and
 * visible on stderr, rather than producing a misleading performance result.
 */
static void checked(cudaError_t error, const char *operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA %s: %s\n", operation, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(operation) checked((operation), #operation)

bool matmul_cuda_available(void) {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

const char *matmul_cuda_device_name(void) {
    /* The exercise explicitly uses device 0; no automatic GPU selection. */
    static cudaDeviceProp device;
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
    return device.name;
}

static size_t bytes_for(size_t elements) {
    if (elements > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "CUDA buffer size overflows size_t.\n");
        exit(EXIT_FAILURE);
    }
    return elements * sizeof(float);
}

static dim3 output_grid(size_t m, size_t n) {
    cudaDeviceProp device;
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));

    /* A block has 16 columns x 16 rows. Round up without overflowing n+15.
     * The student's bounds check must handle the last partial blocks.
     */
    const dim3 block(CUDA_TILE, CUDA_TILE);
    size_t gx = n / block.x + (n % block.x != 0);
    size_t gy = m / block.y + (m % block.y != 0);
    if (gx > (size_t)device.maxGridSize[0] || gy > (size_t)device.maxGridSize[1]) {
        fprintf(stderr, "CUDA output exceeds this device's 2D grid limits.\n");
        exit(EXIT_FAILURE);
    }
    return dim3((unsigned)gx, (unsigned)gy);
}

/* Both exercises pay for exactly the same host-side preparation and transfers. */
static void matmul_cuda(size_t m, size_t n, size_t k,
                         const float *a, const float *b, float *c,
                         MatmulCudaLaunchFn launch) {
    if (!m || !n) return;
    if (!k) {
        for (size_t i = 0; i < m*n; ++i) c[i] = 0.0f;
        return;
    }
    CUDA_CHECK(cudaSetDevice(0));
    const dim3 block(CUDA_TILE, CUDA_TILE);
    const dim3 grid = output_grid(m, n);
    size_t a_bytes = bytes_for(m*k), b_bytes = bytes_for(k*n), c_bytes = bytes_for(m*n);
    float *da = NULL, *db = NULL, *dc = NULL;
    CUDA_CHECK(cudaMalloc((void **)&da, a_bytes));
    CUDA_CHECK(cudaMalloc((void **)&db, b_bytes));
    CUDA_CHECK(cudaMalloc((void **)&dc, c_bytes));
    CUDA_CHECK(cudaMemcpy(da, a, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b, b_bytes, cudaMemcpyHostToDevice));
    /* All-one bits form float NaNs: missing writes cannot pass by accident.
     * This diagnostic initialization is also included in host-call timings.
     */
    CUDA_CHECK(cudaMemset(dc, 0xff, c_bytes));
    launch(grid, block, m, n, k, da, db, dc);
    CUDA_CHECK(cudaGetLastError());       /* Launch configuration / binary errors. */
    CUDA_CHECK(cudaDeviceSynchronize());  /* Wait; report errors from GPU execution. */
    CUDA_CHECK(cudaMemcpy(c, dc, c_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dc));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(da));
}

void matmul_cuda_naive(size_t m, size_t n, size_t k,
                        const float *a, const float *b, float *c) {
    matmul_cuda(m, n, k, a, b, c, matmul_cuda_naive_launch);
}

void matmul_cuda_tiled(size_t m, size_t n, size_t k,
                        const float *a, const float *b, float *c) {
    matmul_cuda(m, n, k, a, b, c, matmul_cuda_tiled_launch);
}

/* Benchmark context is plain host data; CUDA handles/buffers are owned here.
 * Keep all timing machinery out of the student's device kernels.
 */
struct MatmulCudaBenchmark {
    size_t m, n, k, c_bytes;
    unsigned gx, gy;
    float *a, *b, *c;
    MatmulCudaLaunchFn launch;
    cudaEvent_t start, stop;
};

static MatmulCudaLaunchFn benchmark_launch(MatmulFn implementation) {
    if (implementation == matmul_cuda_naive) return matmul_cuda_naive_launch;
    if (implementation == matmul_cuda_tiled) return matmul_cuda_tiled_launch;
    return NULL;
}

bool matmul_cuda_kernel_timing_supported(MatmulFn implementation) {
    return benchmark_launch(implementation) != NULL;
}

MatmulCudaBenchmark *matmul_cuda_benchmark_create(MatmulFn implementation,
    size_t m, size_t n, size_t k, const float *a, const float *b) {
    MatmulCudaLaunchFn launch = benchmark_launch(implementation);
    if (!launch || !m || !n || !k) return NULL;
    MatmulCudaBenchmark *context = (MatmulCudaBenchmark *)malloc(sizeof *context);
    if (!context) return NULL;
    CUDA_CHECK(cudaSetDevice(0));
    dim3 grid = output_grid(m, n);
    context->m = m; context->n = n; context->k = k;
    context->gx = grid.x; context->gy = grid.y;
    context->c_bytes = bytes_for(m*n);
    context->launch = launch;
    size_t a_bytes = bytes_for(m*k), b_bytes = bytes_for(k*n);
    CUDA_CHECK(cudaMalloc((void **)&context->a, a_bytes));
    CUDA_CHECK(cudaMalloc((void **)&context->b, b_bytes));
    CUDA_CHECK(cudaMalloc((void **)&context->c, context->c_bytes));
    CUDA_CHECK(cudaMemcpy(context->a, a, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(context->b, b, b_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventCreate(&context->start));
    CUDA_CHECK(cudaEventCreate(&context->stop));
    return context;
}

double matmul_cuda_benchmark_sample(MatmulCudaBenchmark *context, size_t batch) {
    if (!batch) {
        fprintf(stderr, "CUDA kernel timing requires a positive batch.\n");
        exit(EXIT_FAILURE);
    }
    const dim3 block(CUDA_TILE, CUDA_TILE), grid(context->gx, context->gy);
    /* All operations use the same default stream. The poison finishes BEFORE
     * the start timestamp; no memset/copy/synchronization between the events.
     */
    CUDA_CHECK(cudaMemset(context->c, 0xff, context->c_bytes));
    CUDA_CHECK(cudaEventRecord(context->start, 0));
    for (size_t call = 0; call < batch; ++call) {
        context->launch(grid, block, context->m, context->n, context->k,
                        context->a, context->b, context->c);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(context->stop, 0));
    CUDA_CHECK(cudaEventSynchronize(context->stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, context->start, context->stop));
    return (double)milliseconds / 1000.0 / (double)batch;
}

void matmul_cuda_benchmark_read(MatmulCudaBenchmark *context, float *c) {
    CUDA_CHECK(cudaMemcpy(c, context->c, context->c_bytes, cudaMemcpyDeviceToHost));
}

void matmul_cuda_benchmark_destroy(MatmulCudaBenchmark *context) {
    if (!context) return;
    CUDA_CHECK(cudaEventDestroy(context->stop));
    CUDA_CHECK(cudaEventDestroy(context->start));
    CUDA_CHECK(cudaFree(context->c));
    CUDA_CHECK(cudaFree(context->b));
    CUDA_CHECK(cudaFree(context->a));
    free(context);
}
