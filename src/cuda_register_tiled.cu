#include "cuda_internal.cuh"

/* Each thread owns four outputs, a 2x2 register tile.
 * A 16x16 thread block covers a 32x32 output tile. K still advances by 16.
 */
__global__ static void cuda_register_tiled_kernel(size_t m, size_t n, size_t k,
    const float *a, const float *b, float *c) {
    const size_t tx = threadIdx.x, ty = threadIdx.y;
    const size_t row_base = (size_t)blockIdx.y * CUDA_REGISTER_TILE;
    const size_t col_base = (size_t)blockIdx.x * CUDA_REGISTER_TILE;
    const size_t local_row = ty * CUDA_REGISTER_MICRO;
    const size_t local_col = tx * CUDA_REGISTER_MICRO;
    const size_t row = row_base + local_row;
    const size_t col = col_base + local_col;

    __shared__ float a_tile[CUDA_REGISTER_TILE][CUDA_TILE]; /* 32 rows x 16 K */
    __shared__ float b_tile[CUDA_TILE][CUDA_REGISTER_TILE]; /* 16 K x 32 cols */
    float sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;

    for (size_t p0 = 0; p0 < k; p0 += CUDA_TILE) {
        /* 512 slots in EACH shared tile, but only 256 threads: two loads
         * per input per thread. Loading ownership differs from output ownership.
         */
        for (size_t half = 0; half < CUDA_REGISTER_MICRO; ++half) {
            const size_t ar = ty + half * CUDA_TILE;
            const size_t bc = tx + half * CUDA_TILE;

            /* Guarded global loads, zero-padding missing elements.
             * A source coordinates: (row_base + ar, p0 + tx).
             * B source coordinates: (p0 + ty, col_base + bc).
             */
			a_tile[ar][tx] = (row_base + ar < m && p0 + tx < k) ? a[(row_base + ar) * k + (p0 + tx)] : 0.0f;
			b_tile[ty][bc] = (p0 + ty < k && col_base + bc < n) ? b[(p0 + ty) * n + (col_base + bc)] : 0.0f;
        }
        __syncthreads();

        for (size_t p = 0; p < CUDA_TILE; ++p) {
            /* Load two A values and two B values into these scalars.
             * Use this thread's TWO local rows in A and TWO local cols in B.
             * The other shared coordinate is p. Do not read global memory here.
             */
			float a0 = a_tile[local_row + 0][p];
			float a1 = a_tile[local_row + 1][p];
			float b0 = b_tile[p][local_col + 0];
			float b1 = b_tile[p][local_col + 1];

            /* Use those FOUR loaded values for FOUR accumulator updates.
             * sum00 is output (row, col), sum01 is (row, col+1),
             * sum10 is (row+1, col), sum11 is (row+1, col+1).
             */
			sum00 += a0 * b0;
			sum01 += a0 * b1;
			sum10 += a1 * b0;
			sum11 += a1 * b1;
        }
        __syncthreads(); /* All readers finish before the next tile overwrites. */
    }

    /* Four independently guarded output stores.
     * A valid top-left element does not guarantee the other three are in C.
     * Overwrite each valid element; no atomics and no accumulation into old C.
     * Deliberately no stores yet: NaN poisoning will fail the correctness test.
     */
	if (row < m && col < n) c[row * n + col] = sum00;
	if (row < m && col + 1 < n) c[row * n + col + 1] = sum01;
	if (row + 1 < m && col < n) c[(row + 1) * n + col] = sum10;
	if (row + 1 < m && col + 1 < n) c[(row + 1) * n + col + 1] = sum11;
}

void matmul_cuda_register_tiled_launch(dim3 grid, dim3 block, size_t m, size_t n, size_t k,
    const float *a, const float *b, float *c) {
    cuda_register_tiled_kernel<<<grid, block>>>(m, n, k, a, b, c);
}
