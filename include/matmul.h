#ifndef MATMUL_H
#define MATMUL_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* C[M,N] = A[M,K] * B[K,N], contiguous row-major float arrays.
 * Inputs are read-only; output must be completely overwritten on every call.
 * A, B, C do not overlap. The caller supplies correctly sized buffers and
 * dimensions whose products fit in size_t. No alignment beyond float required.
 * M or N == 0: no output. K == 0: write an M-by-N matrix of zeros.
 * Even for empty arrays, pass valid pointers (the harness does this).
 * CPU learning kernels perform no heap allocation, I/O, timing, or argument parsing.
 * Packed kernels may use bounded automatic workspace, prepared inside the call.
 * External libraries may manage internal workspace; their preparation cost
 * is included in the synchronous call.
 * GPU wrappers may allocate device buffers and transfer data; all work must
 * finish before returning. Runtime failures may emit diagnostics and terminate.
 */
typedef void (*MatmulFn)(size_t m, size_t n, size_t k,
                         const float *a, const float *b, float *c);

void matmul_naive(size_t m, size_t n, size_t k,
                  const float *a, const float *b, float *c);
void matmul_loop_interchange(size_t m, size_t n, size_t k,
                  const float *a, const float *b, float *c);
void matmul_blocked(size_t m, size_t n, size_t k,
                    const float *a, const float *b, float *c);
void matmul_simd_sse(size_t m, size_t n, size_t k,
                     const float *a, const float *b, float *c);
/* Optional: discover availability through matmul_implementations(). */
void matmul_simd_avx(size_t m, size_t n, size_t k,
                     const float *a, const float *b, float *c);
/* Optional: built when an OpenMP C compiler/runtime is available. */
void matmul_openmp(size_t m, size_t n, size_t k,
                   const float *a, const float *b, float *c);
/* Adaptive CPU backend: checks optional ISA support, with a portable fallback. */
void matmul_cpu_combined(size_t m, size_t n, size_t k,
                          const float *a, const float *b, float *c);
void matmul_packed(size_t m, size_t n, size_t k, const float *a, const float *b, float *c);
void matmul_register_avx(size_t m, size_t n, size_t k, const float *a, const float *b, float *c);
void matmul_simd_avx512(size_t m, size_t n, size_t k, const float *a, const float *b, float *c);
/* Optional CUDA exercise: same host-array contract, including transfer costs. */
void matmul_cuda_naive(size_t m, size_t n, size_t k,
                       const float *a, const float *b, float *c);
void matmul_cuda_tiled(size_t m, size_t n, size_t k,
                       const float *a, const float *b, float *c);
void matmul_cuda_register_tiled(size_t m, size_t n, size_t k,
                                const float *a, const float *b, float *c);
void matmul_cuda_prefetch(size_t m, size_t n, size_t k,
                           const float *a, const float *b, float *c);
void matmul_cuda_combined(size_t m, size_t n, size_t k, const float *a, const float *b, float *c);
bool matmul_cuda_available(void);
const char *matmul_cuda_device_name(void);
/* Optional OpenCL exercise: first available Intel GPU, no CPU fallback. */
void matmul_opencl_naive(size_t m, size_t n, size_t k,
                         const float *a, const float *b, float *c);
void matmul_opencl_tiled(size_t m, size_t n, size_t k,
                         const float *a, const float *b, float *c);
bool matmul_opencl_available(void);
void matmul_opencl_register_tiled(size_t m, size_t n, size_t k,
                                  const float *a, const float *b, float *c);
void matmul_clblast(size_t m, size_t n, size_t k,
                    const float *a, const float *b, float *c);
void matmul_opencl_combined(size_t m, size_t n, size_t k,
                            const float *a, const float *b, float *c);
/* Optional CUDA library reference; same host-array contract. */
void matmul_cublas(size_t m, size_t n, size_t k,
                    const float *a, const float *b, float *c);
#ifdef MATMUL_HAVE_CUBLAS
int matmul_cublas_version(void);
#endif
/* Optional OpenBLAS comparison; library metadata is for the runner only. */
void matmul_openblas(size_t m, size_t n, size_t k,
                      const float *a, const float *b, float *c);
#ifdef MATMUL_HAVE_OPENBLAS
const char *matmul_openblas_config(void);
const char *matmul_openblas_core(void);
int matmul_openblas_threads(void);
const char *matmul_openblas_threading(void);
#endif

typedef struct {
    const char *name;
    MatmulFn run;
    const char *description;
    bool (*available)(void); /* NULL for a portable/build-time-only backend. */
} MatmulImplementation;

const MatmulImplementation *matmul_implementations(size_t *count);
/* The registry includes built but unsupported ISA variants. Check before calling. */
bool matmul_implementation_available(const MatmulImplementation *implementation);

#ifdef __cplusplus
}
#endif
#endif
