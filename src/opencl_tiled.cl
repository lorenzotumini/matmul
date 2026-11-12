#define TILE 16

/* A 16x16 work-group cooperatively computes one 16x16 output tile.
 * Global range is rounded UP by the host. Edge work-items still participate
 * in loading and barriers: do not return early when row/col is out of range.
 */
__attribute__((reqd_work_group_size(TILE, TILE, 1)))
__kernel void matmul_opencl_tiled(const ulong m, const ulong n, const ulong k,
    __global const float *a, __global const float *b, __global float *c) {
    const size_t col = get_global_id(0), row = get_global_id(1);
    const size_t tx = get_local_id(0), ty = get_local_id(1);
    __local float a_tile[TILE][TILE];
    __local float b_tile[TILE][TILE];
    float sum = 0.0f;

    for (size_t p0 = 0; p0 < k; p0 += TILE) {
        /* Each work-item loads ONE value into each tile at [ty][tx].
         * A source coordinates: (row, p0 + tx).
         * B source coordinates: (p0 + ty, col).
         * Check each source against its matrix dimensions; otherwise load 0.
         */
		if (row < m && p0 + tx < k) {
			a_tile[ty][tx] = a[row * k + (p0 + tx)];
		} else {
			a_tile[ty][tx] = 0.0f;
		}
		if (p0 + ty < k && col < n) {
			b_tile[ty][tx] = b[(p0 + ty) * n + col];
		} else {
			b_tile[ty][tx] = 0.0f;
		}


        /* Synchronize the work-group, making local writes visible.
         */
		barrier(CLK_LOCAL_MEM_FENCE);

        /* Traverse the TILE reduction positions, accumulating from
         * a_tile and b_tile.
         */
		for (size_t p1 = 0; p1 < TILE; ++p1) {
			sum += a_tile[ty][p1] * b_tile[p1][tx];
		}

        /* Synchronize again before the next iteration replaces tiles. */
		barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Only valid outputs store their sum, overwriting C[row,col]. */
	if (row < m && col < n) {
		c[row * n + col] = sum;
	}
}
