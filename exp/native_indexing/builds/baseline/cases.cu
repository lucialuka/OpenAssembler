#include <stdint.h>

// Original minimal report reproducer: cols deliberately remains a runtime input.
extern "C" __global__ void divide_indices(unsigned* out, unsigned rows, unsigned cols) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) out[i] = i % cols;
}

extern "C" __global__ void quotient_indices(unsigned* out, unsigned rows, unsigned cols) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) out[i] = i / cols;
}
