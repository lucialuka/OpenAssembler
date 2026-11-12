extern "C" __global__ void scale(float* output, const float* input, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i] * 2.0f;
}
