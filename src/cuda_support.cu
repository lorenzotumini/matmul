#include "matmul.h"
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
    const dim3 grid((unsigned)gx, (unsigned)gy);
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
