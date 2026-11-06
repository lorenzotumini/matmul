#ifndef CUDA_INTERNAL_CUH
#define CUDA_INTERNAL_CUH

#include <cuda_runtime.h>
#include <stddef.h>

/* Private bridge: host buffers are handled in cuda_support.cu; this launches
 * the student's kernel on device buffers, asynchronously on the default stream.
 */
void matmul_cuda_naive_launch(dim3 grid, dim3 block, size_t m, size_t n, size_t k,
                              const float *a, const float *b, float *c);

#endif
