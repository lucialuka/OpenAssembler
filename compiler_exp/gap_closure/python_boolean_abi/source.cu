extern "C" __global__ void scalar_flag(bool flag, int *out) {
    out[0] = flag ? 1 : 0;
}
extern "C" __global__ void boolean_arrays(const bool *a, bool *out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = !a[i];
}
