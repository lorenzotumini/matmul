#include "matmul.h"

const MatmulImplementation *matmul_implementations(size_t *count) {
    static const MatmulImplementation implementations[] = {
        {"naive", matmul_naive, "Direct definition, row-major float loops"},
    };
    *count = sizeof implementations / sizeof implementations[0];
    return implementations;
}
