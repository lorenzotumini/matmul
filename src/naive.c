#include "matmul.h"
#include <math.h>

void matmul_naive(size_t m, size_t n, size_t k, 
				  const float *a, const float *b, float *c) {
	if (!k) 
		for (size_t row = 0; row < m; row++)
			for (size_t col = 0; col < n; col++) 
				c[n * row + col] = 0;

    if (!m || !n) return;

    for (size_t row = 0; row < m; row++)
        for (size_t col = 0; col < n; col++) {
            float dot = 0;
            for (size_t i = 0; i < k; i++) dot += a[k * row + i] * b[n * i + col];
            c[n * row + col] = dot;
        }
}
