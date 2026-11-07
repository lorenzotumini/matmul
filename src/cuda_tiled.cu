#include "cuda_internal.cuh"

/* One block cooperates on a 16x16 output tile.
 * Each thread still owns one output; only the input loading changes.
 */
__global__ static void cuda_tiled_kernel(size_t m, size_t n, size_t k,
                                        const float *a, const float *b, float *c) {
    const size_t tx = threadIdx.x, ty = threadIdx.y;
    const size_t row = (size_t)blockIdx.y * CUDA_TILE + ty;
    const size_t col = (size_t)blockIdx.x * CUDA_TILE + tx;
    __shared__ float a_tile[CUDA_TILE][CUDA_TILE];
    __shared__ float b_tile[CUDA_TILE][CUDA_TILE];
    float sum = 0.0f; /* Private to this thread; survives all reduction tiles. */

    for (size_t p0 = 0; p0 < k; p0 += CUDA_TILE) {
        const size_t a_col = p0 + tx;
        const size_t b_row = p0 + ty;

        /* Guarded global-memory loads.
         * Each thread supplies ONE A value and ONE B value to the block.
         * A source coordinates: (row, a_col). B source: (b_row, col).
         * Missing input elements must contribute zero. Guard A and B
         * separately, using each matrix's own shape; avoid invalid reads.
         */
		a_tile[ty][tx] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
		b_tile[ty][tx] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;

        /* Call __syncthreads() before any thread consumes the tiles.
         * Your own loads are not enough: you will read other threads' slots.
         */
		__syncthreads();

        /* Loop over this tile's CUDA_TILE reduction positions.
         * Accumulate products from a row of a_tile and a column of b_tile.
         * These arrays use LOCAL tile coordinates, not global row/col.
         * Keep sum across iterations of the OUTER loop.
         */
		for (size_t p1 = 0; p1 < CUDA_TILE; ++p1) {
			sum += a_tile[ty][p1] * b_tile[p1][tx];
		}

        /* Synchronize again before loading the next reduction tile.
         * Why could overwriting a shared slot too soon corrupt a neighbor?
         * Every thread must reach both barriers, including edge threads.
         */
		__syncthreads();
    }

    /* Only valid output threads store sum to their C element.
     * This is where the row/col output guard belongs. No atomics required.
     */
	if (row < m && col < n) c[row * n + col] = sum;
}

void matmul_cuda_tiled_launch(dim3 grid, dim3 block, size_t m, size_t n, size_t k,
                              const float *a, const float *b, float *c) {
    cuda_tiled_kernel<<<grid, block>>>(m, n, k, a, b, c);
}
