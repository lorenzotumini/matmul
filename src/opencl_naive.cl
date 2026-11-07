/* One work-item computes one output. Global axes: 0 = column, 1 = row.
 * The host supplies row-major float32 buffers and handles empty dimensions.
 * This is OpenCL C, compiled by the GPU driver at program startup.
 */
__kernel void matmul_opencl_naive(const ulong m, const ulong n, const ulong k,
    __global const float *a, __global const float *b, __global float *c) {
    const size_t col = get_global_id(0);
    const size_t row = get_global_id(1);
    if (row >= m || col >= n) return;

	float sum = 0.0f;
	for (size_t p = 0; p < k; ++p) {
		sum += a[row * k + p] * b[p * n + col];
	}
	c[row * n + col] = sum;
}
