extern "C" __global__ void bias(float* output, const float* input, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i] + 0.25f;
}
