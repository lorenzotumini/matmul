#ifndef CUDA_INTERNAL_CUH
#define CUDA_INTERNAL_CUH

#include <cuda_runtime.h>
#include <stddef.h>

enum { CUDA_TILE = 16 }; /* Must match the shared-memory exercise's block shape. */
enum { CUDA_REGISTER_MICRO = 2, CUDA_REGISTER_TILE = CUDA_TILE * CUDA_REGISTER_MICRO };

typedef void (*MatmulCudaLaunchFn)(dim3 grid, dim3 block, size_t m, size_t n, size_t k,
                                    const float *a, const float *b, float *c);

/* Private bridge: host buffers are handled in cuda_support.cu; this launches
 * the student's kernel on device buffers, asynchronously on the default stream.
 */
void matmul_cuda_naive_launch(dim3 grid, dim3 block, size_t m, size_t n, size_t k,
                              const float *a, const float *b, float *c);
void matmul_cuda_tiled_launch(dim3 grid, dim3 block, size_t m, size_t n, size_t k,
                              const float *a, const float *b, float *c);
void matmul_cuda_register_tiled_launch(dim3 grid, dim3 block, size_t m, size_t n, size_t k,
                                       const float *a, const float *b, float *c);
void matmul_cuda_prefetch_launch(dim3 grid, dim3 block, size_t m, size_t n, size_t k,
                                  const float *a, const float *b, float *c);
dim3 matmul_cuda_combined_grid(size_t m, size_t n, size_t k);
void matmul_cuda_combined_launch(dim3 grid, dim3 block, size_t m, size_t n, size_t k,
    const float *a, const float *b, float *c);
#ifdef MATMUL_HAVE_CUBLAS
void matmul_cublas_prepare(size_t m, size_t n, size_t k);
void matmul_cublas_launch(dim3 grid, dim3 block, size_t m, size_t n, size_t k,
                          const float *a, const float *b, float *c);
#endif

#endif
