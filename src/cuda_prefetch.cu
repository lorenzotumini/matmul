#include "cuda_internal.cuh"

/* Same 16x16 threads, 32x32 output tile, 2x2 adjacent outputs per thread,
 * cooperative input mapping and arithmetic as cuda_register_tiled.
 * Only the schedule changes: fetch the NEXT tile into registers before
 * computing with the CURRENT shared tile. No second shared buffer is needed.
 */
__global__ static void cuda_prefetch_kernel(size_t m, size_t n, size_t k,
    const float *a, const float *b, float *c) {
    const size_t tx = threadIdx.x, ty = threadIdx.y;
    const size_t row_base = (size_t)blockIdx.y * CUDA_REGISTER_TILE;
    const size_t col_base = (size_t)blockIdx.x * CUDA_REGISTER_TILE;
    const size_t local_row = ty * CUDA_REGISTER_MICRO;
    const size_t local_col = tx * CUDA_REGISTER_MICRO;
    const size_t row = row_base + local_row, col = col_base + local_col;

    __shared__ float a_tile[CUDA_REGISTER_TILE][CUDA_TILE];
    __shared__ float b_tile[CUDA_TILE][CUDA_REGISTER_TILE];
    float sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;

    /* Prime the pipeline with the first tile. The host handles K == 0. */
    for (size_t half = 0; half < CUDA_REGISTER_MICRO; ++half) {
        const size_t ar = ty + half * CUDA_TILE;
        const size_t bc = tx + half * CUDA_TILE;
        a_tile[ar][tx] = row_base + ar < m && tx < k ?
            a[(row_base + ar)*k + tx] : 0.0f;
        b_tile[ty][bc] = ty < k && col_base + bc < n ?
            b[ty*n + col_base + bc] : 0.0f;
    }
    __syncthreads();

    for (size_t p0 = 0; p0 < k; p0 += CUDA_TILE) {
        const size_t next = p0 + CUDA_TILE;
        float next_a[CUDA_REGISTER_MICRO], next_b[CUDA_REGISTER_MICRO];

        /* Keep future inputs private while shared memory still holds the
         * current tile. Zero padding applies to the final K tile and edges.
         */
        if (next < k) {
            for (size_t half = 0; half < CUDA_REGISTER_MICRO; ++half) {
                const size_t ar = ty + half * CUDA_TILE;
                const size_t bc = tx + half * CUDA_TILE;
                next_a[half] = row_base + ar < m && next + tx < k ?
                    a[(row_base + ar)*k + next + tx] : 0.0f;
                next_b[half] = next + ty < k && col_base + bc < n ?
                    b[(next + ty)*n + col_base + bc] : 0.0f;
            }
        }

        for (size_t p = 0; p < CUDA_TILE; ++p) {
            float a0 = a_tile[local_row][p], a1 = a_tile[local_row + 1][p];
            float b0 = b_tile[p][local_col], b1 = b_tile[p][local_col + 1];
            sum00 += a0 * b0;
            sum01 += a0 * b1;
            sum10 += a1 * b0;
            sum11 += a1 * b1;
        }
        __syncthreads(); /* Every reader must finish before replacing shared data. */

        if (next < k) {
            for (size_t half = 0; half < CUDA_REGISTER_MICRO; ++half) {
                const size_t ar = ty + half * CUDA_TILE;
                const size_t bc = tx + half * CUDA_TILE;
                a_tile[ar][tx] = next_a[half];
                b_tile[ty][bc] = next_b[half];
            }
            /* next < k is uniform across the block: all threads participate,
             * including those with no valid output. No publish on the last tile.
             */
            __syncthreads();
        }
    }

    if (row < m && col < n) c[row*n + col] = sum00;
    if (row < m && col + 1 < n) c[row*n + col + 1] = sum01;
    if (row + 1 < m && col < n) c[(row + 1)*n + col] = sum10;
    if (row + 1 < m && col + 1 < n) c[(row + 1)*n + col + 1] = sum11;
}

void matmul_cuda_prefetch_launch(dim3 grid, dim3 block, size_t m, size_t n, size_t k,
    const float *a, const float *b, float *c) {
    cuda_prefetch_kernel<<<grid, block>>>(m, n, k, a, b, c);
}
