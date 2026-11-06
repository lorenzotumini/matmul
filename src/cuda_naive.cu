#include "cuda_internal.cuh"

/* One GPU thread computes one C[row, col]. */
__global__ static void cuda_naive_kernel(size_t m, size_t n, size_t k,
                                        const float *a, const float *b, float *c) {
    /* compute row and col from blockIdx, blockDim and threadIdx.
     * Map x to columns, y to rows. Think: block's origin + thread's offset.
     * Use size_t for matrix offsets; cast before multiplying built-in indices.
     */
	size_t row = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
	size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    /* return if this thread falls outside the M-by-N output.
     * Rounded-up grids deliberately contain extra threads on boundary blocks.
     */
	if (row >= m || col >= n) return;

    /* accumulate one dot product in a local float.
     * Loop over K only. Same indexing as naive CPU code.
     */
	float sum = 0.0f;
	for (size_t i = 0; i < k; i++) {
		sum += a[row * k + i] * b[i * n + col];
	}

    /* overwrite this thread's single output element with the sum.
     * No atomics or barriers are needed: every valid thread owns one element.
     */
	c[row * n + col] = sum;	
}

/* Supplied launch syntax: grid of blocks, each containing block.x*block.y
 * threads. The call queues GPU work; cuda_support.cu checks and waits for it.
 */
void matmul_cuda_naive_launch(dim3 grid, dim3 block, size_t m, size_t n, size_t k,
                              const float *a, const float *b, float *c) {
    cuda_naive_kernel<<<grid, block>>>(m, n, k, a, b, c);
}
