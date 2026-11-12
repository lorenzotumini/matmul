#define TILE 16
#define OUTPUT_TILE 32

/* 16x16 work-items, 32x32 outputs, depth-16 reduction tiles.
 * Each work-item owns an adjacent 2x2 output group, held in four registers.
 * Do not return early: every work-item participates in loads and barriers.
 */
__attribute__((reqd_work_group_size(TILE, TILE, 1)))
__kernel void matmul_opencl_register_tiled(const ulong m, const ulong n, const ulong k,
    __global const float *a, __global const float *b, __global float *c) {
    const size_t tx = get_local_id(0), ty = get_local_id(1);
    const size_t row_base = get_group_id(1) * OUTPUT_TILE;
    const size_t col_base = get_group_id(0) * OUTPUT_TILE;
    const size_t local_row = 2 * ty, local_col = 2 * tx;
    const size_t row = row_base + local_row, col = col_base + local_col;
    __local float a_tile[OUTPUT_TILE][TILE];
    __local float b_tile[TILE][OUTPUT_TILE];
    float sum00 = 0, sum01 = 0, sum10 = 0, sum11 = 0;

    for (size_t p0 = 0; p0 < k; p0 += TILE) {
        for (size_t part = 0; part < 2; ++part) {
            const size_t ar = ty + part * TILE, bc = tx + part * TILE;
            /* Each work-item loads TWO values per input across this loop.
             * A: global (row_base+ar, p0+tx) -> a_tile[ar][tx].
             * B: global (p0+ty, col_base+bc) -> b_tile[ty][bc].
             * Guard each source separately and zero-pad missing elements.
             * Loading ownership differs from this work-item's output ownership.
             */
			if (row_base + ar < m && p0 + tx < k) {
				a_tile[ar][tx] = a[(row_base + ar) * k + (p0 + tx)];
			} else {
				a_tile[ar][tx] = 0;
			}
			if (p0 + ty < k && col_base + bc < n) {
				b_tile[ty][bc] = b[(p0 + ty) * n + (col_base + bc)];
			} else {
				b_tile[ty][bc] = 0;
			}
        }
        /* Publish local loads with a work-group barrier. */
		barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t p = 0; p < TILE; ++p) {
            /* Read two A values for local_row/local_row+1 and two B
             * values for local_col/local_col+1, all from LOCAL memory.
			 */
			float a0 = a_tile[local_row][p];
			float a1 = a_tile[local_row + 1][p];
			float b0 = b_tile[p][local_col];
			float b1 = b_tile[p][local_col + 1];

            /* Use those four scalars to update all four sums.
             * sum00 -> (row,col), sum01 -> (row,col+1),
             * sum10 -> (row+1,col), sum11 -> (row+1,col+1).
             */
			sum00 += a0 * b0;
			sum01 += a0 * b1;
			sum10 += a1 * b0;
			sum11 += a1 * b1;
        }
        /* Protect readers before the next tile overwrites local data. */
		barrier(CLK_LOCAL_MEM_FENCE);
    }
    /* Four independently guarded stores, overwriting C.
     * A valid (row,col) does not imply the other three outputs are valid.
     */
	if (row < m && col < n) c[row * n + col] = sum00;
	if (row < m && col + 1 < n) c[row * n + col + 1] = sum01;
	if (row + 1 < m && col < n) c[(row + 1) * n + col] = sum10;
	if (row + 1 < m && col + 1 < n) c[(row + 1) * n + col + 1] = sum11;
}
