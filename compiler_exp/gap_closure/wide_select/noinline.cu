extern "C" __device__ __noinline__ unsigned long long mix(unsigned long long a, unsigned x) {
    if (x & 1) return a + 0x123456789abcdefULL;
    return a ^ 0xfedcba9876543210ULL;
}
extern "C" __device__ __noinline__ unsigned next(unsigned x) {
    unsigned result=x;
    for (int i=0;i<3;i++) result = result * 1664525u + 1013904223u;
    return result;
}
extern "C" __global__ void entry(unsigned long long *out, const unsigned long long *a, unsigned n) {
    unsigned i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) out[i]=mix(a[i],next(i));
}
