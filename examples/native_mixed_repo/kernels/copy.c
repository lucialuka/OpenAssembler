/* This source is compiled as CUDA C/C++ by nvcc's PTX frontend. */
extern "C" __global__ void copy(float* output, const float* input, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i];
}
