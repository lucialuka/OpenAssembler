extern "C" __global__ void multiply(unsigned long long *out,
                                   const unsigned long long *a,
                                   const unsigned long long *b, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}
