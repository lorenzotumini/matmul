#include "matmul.h"
#include "cpu_features.h"

/* Availability belongs to each entry, independent of its position in the list.
 * This avoids accidentally exposing an ISA kernel when entries are reordered.
 */
bool matmul_implementation_available(const MatmulImplementation *impl) {
    return !impl->available || impl->available();
}

const MatmulImplementation *matmul_implementations(size_t *count) {
    static const MatmulImplementation implementations[] = {
        {"naive", matmul_naive, "Direct definition, row-major float loops", NULL},
        {"loop_interchange", matmul_loop_interchange, "Contiguous inner-loop accesses", NULL},
        {"blocked", matmul_blocked, "Cache blocking with fixed-size tiles", NULL},
        {"packed", matmul_packed, "Blocked traversal with copied B panels (single thread)", NULL},
#if defined(__x86_64__) || defined(_M_X64)
        {"simd_sse", matmul_simd_sse, "Four-float SSE intrinsics", NULL},
#endif
#ifdef MATMUL_HAVE_SIMD_AVX
        {"simd_avx", matmul_simd_avx, "Eight-float AVX with FMA", matmul_has_fma},
#endif
#ifdef MATMUL_HAVE_AVX512
        {"simd_avx512", matmul_simd_avx512, "Sixteen-float AVX-512 with FMA", matmul_has_avx512},
#endif
#ifdef MATMUL_HAVE_REGISTER_AVX
        {"register_avx", matmul_register_avx, "4x16 AVX/FMA register tiles, no cache blocking or threads", matmul_has_fma},
#endif
#ifdef MATMUL_HAVE_OPENMP
        {"openmp", matmul_openmp, "OpenMP static output-row scheduling", NULL},
#endif
#ifdef MATMUL_HAVE_OPENMP
        {"cpu_combined", matmul_cpu_combined, "Adaptive direct/packed CPU tiles, SIMD/FMA and OpenMP when supported", NULL},
#else
        {"cpu_combined", matmul_cpu_combined, "Adaptive direct/packed CPU tiles, SIMD/FMA when supported (single-thread build)", NULL},
#endif
#ifdef MATMUL_HAVE_OPENBLAS
        {"openblas", matmul_openblas, "OpenBLAS CBLAS SGEMM, row-major, alpha=1 beta=0", NULL},
#endif
#ifdef MATMUL_HAVE_CUDA
        {"cuda_naive", matmul_cuda_naive, "CUDA exercise: one output per thread; host-call timing includes allocation/transfers", matmul_cuda_available},
        {"cuda_tiled", matmul_cuda_tiled, "CUDA exercise: shared-memory tiles; host-call timing includes allocation/transfers", matmul_cuda_available},
        {"cuda_register_tiled", matmul_cuda_register_tiled, "CUDA exercise: 2x2 per-thread register tiles; 32x32 output per block", matmul_cuda_available},
        {"cuda_prefetch", matmul_cuda_prefetch, "Register-tiled baseline with next-tile register prefetch (same layout and tile size)", matmul_cuda_available},
#endif
#ifdef MATMUL_HAVE_CUDA
        {"cuda_combined", matmul_cuda_combined, "Adaptive SIMT tiles, register reuse and software prefetch", matmul_cuda_available},
#endif
#ifdef MATMUL_HAVE_CUBLAS
        {"cublas", matmul_cublas, "cuBLAS SGEMM, FP32 pedantic, alpha=1 beta=0", matmul_cuda_available},
#endif
    };
    *count = sizeof implementations / sizeof implementations[0];
    return implementations;
}
