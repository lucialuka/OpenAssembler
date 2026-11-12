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

extern "C" __global__ void runtime_u32(const uint32_t* a, const uint32_t* b,
                                       uint32_t* q, uint32_t* r, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t x = a[i], d = b[i];
        q[i] = x / d;
        r[i] = x % d;
    }
}

extern "C" __global__ void runtime_s32(const int32_t* a, const int32_t* b,
                                       int32_t* q, int32_t* r, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int32_t x = a[i], d = b[i];
        q[i] = x / d;
        r[i] = x % d;
    }
}
