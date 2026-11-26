extern "C" __device__ unsigned helper(unsigned);
extern "C" __global__ void linked_probe(unsigned *out,const unsigned *input,unsigned n) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) out[i]=helper(input[i]);
}
