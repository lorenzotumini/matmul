#include "matmul.h"

const MatmulImplementation *matmul_implementations(size_t *count) {
    static const MatmulImplementation implementations[] = {
        {"naive", matmul_naive, "Direct definition, row-major float loops"},
        {"loop_interchange", matmul_loop_interchange, "Loop interchange to reduce cache misses"},
    };
    *count = sizeof implementations / sizeof implementations[0];
    return implementations;
}
