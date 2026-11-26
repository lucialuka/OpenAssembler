extern "C" __device__ __noinline__ unsigned helper(unsigned x) {
    unsigned result=x;
    for (int i=0;i<3;i++) result=result*1664525u+1013904223u;
    return result;
}
extern "C" __global__ void probe(unsigned long long *out,const unsigned long long *input,unsigned n) {
    unsigned i=blockIdx.x*128+threadIdx.x;
    if (i<n) {unsigned long long value=input[i];unsigned x=helper((unsigned)value);out[i]=(value&0xffffffff00000000ULL)|x;}
}
