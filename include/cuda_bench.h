#ifndef CUDA_BENCH_H
#define CUDA_BENCH_H

#include "matmul.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Optional benchmark-only API. It does not change the host-array kernel
 * contract. One context owns prepared device buffers and reusable CUDA events.
 * Use it from one host thread on device 0, without concurrent GPU work.
 */
typedef struct MatmulCudaBenchmark MatmulCudaBenchmark;

bool matmul_cuda_kernel_timing_supported(MatmulFn implementation);
/* Dimensions/buffer sizes must be prevalidated, as for the ordinary kernels.
 * Returns NULL for unsupported implementations, empty shapes or host OOM.
 * Device/runtime errors terminate with diagnostics, as in the host wrappers.
 */
MatmulCudaBenchmark *matmul_cuda_benchmark_create(MatmulFn implementation,
    size_t m, size_t n, size_t k, const float *a, const float *b);
/* Positive batch required. Returns CUDA-event seconds per launch. Output is
 * poisoned before the start event; only launches lie between the two events.
 * Waits for the stop event before returning; no host transfers in the window.
 */
double matmul_cuda_benchmark_sample(MatmulCudaBenchmark *context, size_t batch);
void matmul_cuda_benchmark_read(MatmulCudaBenchmark *context, float *c);
void matmul_cuda_benchmark_destroy(MatmulCudaBenchmark *context);

#ifdef __cplusplus
}
#endif
#endif
