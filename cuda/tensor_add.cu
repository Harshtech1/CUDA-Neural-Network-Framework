#include "../include/cuda/tensor_cuda.hpp"

#include <cuda_runtime.h>

__global__
void tensorAddKernel(
    const float* A,
    const float* B,
    float* C,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

void tensorAddCUDA(
    const float* A,
    const float* B,
    float* C,
    int N)
{
    int blockSize = 256;

    int gridSize =
        (N + blockSize - 1) / blockSize;

    tensorAddKernel<<<gridSize, blockSize>>>(
        A,
        B,
        C,
        N
    );

    cudaDeviceSynchronize();
}
