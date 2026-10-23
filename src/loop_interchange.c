#include "matmul.h"

void matmul_loop_interchange(size_t m, size_t n, size_t k, 
				  const float *a, const float *b, float *c) {
	if (!m || !n) return;

	// if (!k) 
	for (size_t row = 0; row < m; row++)
		for (size_t col = 0; col < n; col++) 
			c[n * row + col] = 0;

	// exchange col and i loops, it reduces the number of cache misses
	for (size_t row = 0; row < m; row++) {
		for (size_t i = 0; i < k; i++) {
			// save intermediate value to avoid repeated memory access
			float a_ik = a[k * row + i];
			// also b is accessed in a row-major order, so it is more cache-friendly
			for (size_t col = 0; col < n; col++) {
				c[n * row + col] += a_ik * b[n * i + col];
			}
		}
	}
}
