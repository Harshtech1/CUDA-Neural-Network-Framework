#pragma once

#include <vector>
#include <string>

class Tensor
{
private:

    // Host (CPU) memory
    float* cpu_data_;

    // Device (GPU) memory
    float* gpu_data_;

    // Tensor shape
    std::vector<int> shape_;

    // Number of elements
    size_t size_;

    // Current device
    std::string device_;

public:

    Tensor(const std::vector<int>& shape);

    ~Tensor();

    void fill(float value);

    void print() const;

    void info() const;

    size_t size() const;

    // CUDA Memory Functions
    void allocateGPU();

    void toGPU();

    void toCPU();

    void freeGPU();
};
