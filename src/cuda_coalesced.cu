#include "cuda_internal.cuh"

/* Remap the four outputs owned by each thread.
 * Same 16x16 threads, 32x32 output tile, shared tiles, loads and arithmetic
 * as cuda_register_tiled. Only output ownership and its dependent indices change.
 */
__global__ static void cuda_coalesced_kernel(size_t m, size_t n, size_t k,
    const float *a, const float *b, float *c) {
    const size_t tx = threadIdx.x, ty = threadIdx.y;
    const size_t row_base = (size_t)blockIdx.y * CUDA_REGISTER_TILE;
    const size_t col_base = (size_t)blockIdx.x * CUDA_REGISTER_TILE;

    /* Replace the four placeholders with LOCAL output coordinates.
     * Adjacent threads should own adjacent columns for each individual store.
     * Choose one row/column in each HALF of the 32x32 output tile.
     * Rows follow ty, columns follow tx; the two halves are 16 elements apart.
     * Do not keep the baseline's 2*ty / 2*tx and +1 ownership.
     */
	const size_t local_row0 = ty, local_row1 = ty + CUDA_TILE;
	const size_t local_col0 = tx, local_col1 = tx + CUDA_TILE;
	const size_t row0 = row_base + local_row0, row1 = row_base + local_row1;
	const size_t col0 = col_base + local_col0, col1 = col_base + local_col1;

    __shared__ float a_tile[CUDA_REGISTER_TILE][CUDA_TILE];
    __shared__ float b_tile[CUDA_TILE][CUDA_REGISTER_TILE];
    float sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;

    for (size_t p0 = 0; p0 < k; p0 += CUDA_TILE) {
        /* Cooperative loading stays exactly as in the baseline. */
        for (size_t half = 0; half < CUDA_REGISTER_MICRO; ++half) {
            const size_t ar = ty + half * CUDA_TILE;
            const size_t bc = tx + half * CUDA_TILE;
            a_tile[ar][tx] = row_base + ar < m && p0 + tx < k ?
                a[(row_base + ar)*k + p0 + tx] : 0.0f;
            b_tile[ty][bc] = p0 + ty < k && col_base + bc < n ?
                b[(p0 + ty)*n + col_base + bc] : 0.0f;
        }
        __syncthreads();
        for (size_t p = 0; p < CUDA_TILE; ++p) {
            /* Reads follow the remapped ownership; arithmetic is unchanged. */
            float a0 = a_tile[local_row0][p], a1 = a_tile[local_row1][p];
            float b0 = b_tile[p][local_col0], b1 = b_tile[p][local_col1];
            sum00 += a0 * b0;
            sum01 += a0 * b1;
            sum10 += a1 * b0;
            sum11 += a1 * b1;
        }
        __syncthreads();
    }

    /* Rour independently guarded stores using row0/row1, col0/col1.
     * sum00 -> (row0,col0), sum01 -> (row0,col1),
     * sum10 -> (row1,col0), sum11 -> (row1,col1).
     * The second row/column is no longer +1. Guard each coordinate pair.
     */
	if (row0 < m && col0 < n) c[row0*n + col0] = sum00;
	if (row0 < m && col1 < n) c[row0*n + col1] = sum01;
	if (row1 < m && col0 < n) c[row1*n + col0] = sum10;
	if (row1 < m && col1 < n) c[row1*n + col1] = sum11;	
}

void matmul_cuda_coalesced_launch(dim3 grid, dim3 block, size_t m, size_t n, size_t k,
    const float *a, const float *b, float *c) {
    cuda_coalesced_kernel<<<grid, block>>>(m, n, k, a, b, c);
}
