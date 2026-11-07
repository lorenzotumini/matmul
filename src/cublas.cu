#include "matmul.h"
#include "cuda_internal.cuh"
#include <cublas_v2.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

static void checked(cublasStatus_t status, const char *operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS %s: %s\n", operation, cublasGetStatusString(status));
        exit(EXIT_FAILURE);
    }
}
#define CUBLAS_CHECK(operation) checked((operation), #operation)

/* Warm library state, just as with CPU BLAS. Keep handle/workspace setup out
 * of steady-state measurements. Each host thread owns its handle on device 0.
 */
struct LocalHandle {
    cublasHandle_t value;
    LocalHandle() {
        cudaError_t error = cudaSetDevice(0);
        if (error != cudaSuccess) {
            fprintf(stderr, "cuBLAS device selection: %s\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
        CUBLAS_CHECK(cublasCreate(&value));
        CUBLAS_CHECK(cublasSetStream(value, 0));
        CUBLAS_CHECK(cublasSetPointerMode(value, CUBLAS_POINTER_MODE_HOST));
        /* Keep the FP32 baseline explicit, including on newer desktop GPUs. */
        CUBLAS_CHECK(cublasSetMathMode(value, CUBLAS_PEDANTIC_MATH));
    }
    ~LocalHandle() { (void)cublasDestroy(value); }
};

static cublasHandle_t handle(void) {
    static thread_local LocalHandle state;
    return state.value;
}

void matmul_cublas_prepare(size_t m, size_t n, size_t k) {
    if (m > INT_MAX || n > INT_MAX || k > INT_MAX) {
        fprintf(stderr, "cuBLAS SGEMM dimensions exceed the int interface.\n");
        exit(EXIT_FAILURE);
    }
    (void)handle();
}

int matmul_cublas_version(void) {
    int version = 0;
    CUBLAS_CHECK(cublasGetVersion(handle(), &version));
    return version;
}

void matmul_cublas_launch(dim3 grid, dim3 block, size_t m, size_t n, size_t k,
    const float *a, const float *b, float *c) {
    (void)grid; (void)block; /* cuBLAS chooses its own kernel/grid configuration. */
    const float alpha = 1.0f, beta = 0.0f;
    /* Row-major C=A*B is column-major C^T=B^T*A^T over the SAME buffers.
     * Swap operands and M/N; no physical transpose or extra copy is needed.
     * First operand B has leading dimension N, second A has leading K.
     */
    CUBLAS_CHECK(cublasSgemm(handle(), CUBLAS_OP_N, CUBLAS_OP_N,
        (int)n, (int)m, (int)k, &alpha, b, (int)n, a, (int)k, &beta, c, (int)n));
}
