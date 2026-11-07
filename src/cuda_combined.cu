#include "cuda_internal.cuh"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

/* A warp follows 32 consecutive columns. Each thread holds TM x TN outputs,
 * reusing a broadcast A value and TN B values across independent sums.
 * Global loads for the NEXT K tile are issued before current-tile arithmetic
 * and held in registers until all readers finish with shared memory.
 * No Tensor Cores, reduced precision, packing or library calls.
 */
template<int BM, int BN>
__global__ static void combined_kernel(size_t m, size_t n, size_t k,
    const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c) {
    constexpr int BK = 16, THREADS = 256, TM = BM / 8, TN = BN / 32;
    constexpr int LOAD_A = BM * BK / THREADS, LOAD_B = BK * BN / THREADS;
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const size_t row0 = (size_t)blockIdx.y * BM;
    const size_t col0 = (size_t)blockIdx.x * BN;
    __shared__ float at[BM][BK], bt[BK][BN];
    float sums[TM][TN] = {};

    /* Every shared slot has one loading owner; invalid input is zero padded. */
    for (int q = tid; q < BM * BK; q += THREADS) {
        size_t row = row0 + q / BK, p = q % BK;
        at[q / BK][q % BK] = row < m && p < k ? a[row*k + p] : 0.0f;
    }
    for (int q = tid; q < BK * BN; q += THREADS) {
        size_t p = q / BN, col = col0 + q % BN;
        bt[q / BN][q % BN] = p < k && col < n ? b[p*n + col] : 0.0f;
    }
    __syncthreads();

    for (size_t p0 = 0; p0 < k; p0 += BK) {
        const size_t next = p0 + BK;
        float next_a[LOAD_A], next_b[LOAD_B];
        if (next < k) {
            #pragma unroll
            for (int t = 0; t < LOAD_A; ++t) {
                int q = tid + t * THREADS;
                size_t row = row0 + q / BK, p = next + q % BK;
                next_a[t] = row < m && p < k ? a[row*k + p] : 0.0f;
            }
            #pragma unroll
            for (int t = 0; t < LOAD_B; ++t) {
                int q = tid + t * THREADS;
                size_t p = next + q / BN, col = col0 + q % BN;
                next_b[t] = p < k && col < n ? b[p*n + col] : 0.0f;
            }
        }
        #pragma unroll
        for (int p = 0; p < BK; ++p) {
            float bv[TN];
            #pragma unroll
            for (int j = 0; j < TN; ++j) bv[j] = bt[p][lane + j * 32];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                float av = at[warp * TM + i][p];
                #pragma unroll
                for (int j = 0; j < TN; ++j) sums[i][j] += av * bv[j];
            }
        }
        __syncthreads(); /* All current-tile readers finish before replacement. */
        if (next < k) {
            #pragma unroll
            for (int t = 0; t < LOAD_A; ++t) {
                int q = tid + t * THREADS;
                at[q / BK][q % BK] = next_a[t];
            }
            #pragma unroll
            for (int t = 0; t < LOAD_B; ++t) {
                int q = tid + t * THREADS;
                bt[q / BN][q % BN] = next_b[t];
            }
        }
        __syncthreads(); /* Uniform barrier, including threads outside C. */
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        size_t row = row0 + warp * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            size_t col = col0 + lane + j * 32;
            if (row < m && col < n) c[row*n + col] = sums[i][j];
        }
    }
}

/* One shared selection rule for allocation/validation and timed launch paths.
 * Fixed, readable heuristics; no runtime search or data-dependent decisions.
 */
static int strategy(size_t m, size_t n, size_t k) {
    if (m < 32 || n < 32 || k < 16) return 0;
    if ((double)m * (double)n * (double)k < 128.0 * 128 * 128) return 1;
    if (m < 128) return n >= 512 ? 32 : 1;
    if (m >= 512 && n >= 512 && k >= 128) return 128;
    return 64;
}

dim3 matmul_cuda_combined_grid(size_t m, size_t n, size_t k) {
    int choice = strategy(m, n, k);
    size_t rows = choice == 0 ? CUDA_TILE : choice == 1 ? CUDA_REGISTER_TILE : choice;
    size_t cols = choice == 0 ? CUDA_TILE : choice == 1 ? CUDA_REGISTER_TILE : choice == 128 ? 128 : 64;
    size_t gx = n / cols + (n % cols != 0), gy = m / rows + (m % rows != 0);
    if (gx > UINT_MAX || gy > UINT_MAX) {
        fprintf(stderr, "CUDA combined grid exceeds dim3 range.\n");
        exit(EXIT_FAILURE);
    }
    return dim3((unsigned)gx, (unsigned)gy);
}

void matmul_cuda_combined_launch(dim3 grid, dim3 block, size_t m, size_t n, size_t k,
    const float *a, const float *b, float *c) {
    switch (strategy(m, n, k)) {
    case 0: matmul_cuda_tiled_launch(grid, block, m, n, k, a, b, c); break;
    case 1: matmul_cuda_register_tiled_launch(grid, block, m, n, k, a, b, c); break;
    case 32: combined_kernel<32, 64><<<grid, dim3(32, 8)>>>(m, n, k, a, b, c); break;
    case 128: combined_kernel<128, 128><<<grid, dim3(32, 8)>>>(m, n, k, a, b, c); break;
    default: combined_kernel<64, 64><<<grid, dim3(32, 8)>>>(m, n, k, a, b, c); break;
    }
}
