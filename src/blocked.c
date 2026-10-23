#include "matmul.h"

void matmul_blocked(size_t m, size_t n, size_t k,
                    const float *a, const float *b, float *c) {
    enum { TILE_SIZE = 128 };

    if (!m || !n) return;

	// initialize for accumulation
    for (size_t row = 0; row < m; ++row)
        for (size_t col = 0; col < n; ++col)
            c[row * n + col] = 0;

	// separate the computation into tiles to improve cache locality
	// NOTE: this provides a speed only for large matrices that don't fit in cache, but can be slower for small matrices
    for (size_t row0 = 0; row0 < m; row0 += TILE_SIZE) {
		size_t row_end = row0 + TILE_SIZE < m ? row0 + TILE_SIZE : m;
        for (size_t col0 = 0; col0 < n; col0 += TILE_SIZE) {
			size_t col_end = col0 + TILE_SIZE < n ? col0 + TILE_SIZE : n;
            for (size_t p0 = 0; p0 < k; p0 += TILE_SIZE) {
				size_t p_end = p0 + TILE_SIZE < k ? p0 + TILE_SIZE : k;

				for (size_t row = row0; row < row_end; ++row) {
					for (size_t p = p0; p < p_end; ++p) {
						float a_val = a[row * k + p];
						for (size_t col = col0; col < col_end; ++col) {
							c[row * n + col] += a_val * b[p * n + col];
						}
					}
				}
            }
        }
    }
}
