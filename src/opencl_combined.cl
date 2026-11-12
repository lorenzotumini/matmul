/* Compile-time geometry. The host selects a few measured configurations.
 * X lanes follow contiguous columns; each work-item owns RM x RN outputs.
 * Next-tile prefetch uses registers, not a second local-memory buffer.
 */
#ifndef WX
#define WX 16
#define WY 8
#define RM 4
#define RN 4
#define PREFETCH 1
#endif
#define BK 16
#define BM (WY * RM)
#define BN (WX * RN)
#define THREADS (WX * WY)
#define LOAD_A ((BM * BK) / THREADS)
#define LOAD_B ((BK * BN) / THREADS)

__attribute__((reqd_work_group_size(WX, WY, 1)))
__kernel void matmul_opencl_combined(const ulong m, const ulong n, const ulong k,
    __global const float *restrict a, __global const float *restrict b,
    __global float *restrict c) {
    const size_t x = get_local_id(0), y = get_local_id(1);
    const size_t tid = y * WX + x;
    const size_t row0 = get_group_id(1) * BM, col0 = get_group_id(0) * BN;
    __local float at[BM][BK], bt[BK][BN];
    float sums[RM][RN] = {{0}};
    for (size_t q = tid; q < BM * BK; q += THREADS) {
        size_t row = row0 + q / BK, p = q % BK;
        at[q/BK][q%BK] = row < m && p < k ? a[row*k+p] : 0.0f;
    }
    for (size_t q = tid; q < BK * BN; q += THREADS) {
        size_t p = q / BN, col = col0 + q % BN;
        bt[q/BN][q%BN] = p < k && col < n ? b[p*n+col] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t p0 = 0; p0 < k; p0 += BK) {
        const size_t next = p0 + BK;
#if PREFETCH
        float next_a[LOAD_A], next_b[LOAD_B];
        if (next < k) {
            #pragma unroll
            for (int t = 0; t < LOAD_A; ++t) {
                size_t q = tid + t*THREADS, row = row0 + q/BK, p = next + q%BK;
                next_a[t] = row < m && p < k ? a[row*k+p] : 0.0f;
            }
            #pragma unroll
            for (int t = 0; t < LOAD_B; ++t) {
                size_t q = tid + t*THREADS, p = next + q/BN, col = col0 + q%BN;
                next_b[t] = p < k && col < n ? b[p*n+col] : 0.0f;
            }
        }
#endif
        #pragma unroll
        for (int p = 0; p < BK; ++p) {
            float bv[RN];
            #pragma unroll
            for (int j = 0; j < RN; ++j) bv[j] = bt[p][x + j*WX];
            #pragma unroll
            for (int i = 0; i < RM; ++i) {
                float av = at[y + i*WY][p];
                #pragma unroll
                for (int j = 0; j < RN; ++j) sums[i][j] += av * bv[j];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE); /* Readers finish before replacement. */
        if (next < k) { /* Uniform condition, including edge work-items. */
            #pragma unroll
            for (int t = 0; t < LOAD_A; ++t) {
                size_t q = tid + t*THREADS;
#if PREFETCH
                at[q/BK][q%BK] = next_a[t];
#else
                size_t row = row0 + q/BK, p = next + q%BK;
                at[q/BK][q%BK] = row < m && p < k ? a[row*k+p] : 0.0f;
#endif
            }
            #pragma unroll
            for (int t = 0; t < LOAD_B; ++t) {
                size_t q = tid + t*THREADS;
#if PREFETCH
                bt[q/BN][q%BN] = next_b[t];
#else
                size_t p = next + q/BN, col = col0 + q%BN;
                bt[q/BN][q%BN] = p < k && col < n ? b[p*n+col] : 0.0f;
#endif
            }
            barrier(CLK_LOCAL_MEM_FENCE); /* Publish before the next iteration. */
        }
    }
    #pragma unroll
    for (int i = 0; i < RM; ++i) {
        size_t row = row0 + y + i*WY;
        #pragma unroll
        for (int j = 0; j < RN; ++j) {
            size_t col = col0 + x + j*WX;
            if (row < m && col < n) c[row*n+col] = sums[i][j];
        }
    }
}
