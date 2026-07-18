#pragma once

#include <string>
#include <vector>

class Tensor
{
private:

    float* cpu_data_;
    float* gpu_data_;

    std::vector<int> shape_;

    size_t size_;

    std::string device_;

public:

    Tensor(const std::vector<int>& shape);

    ~Tensor();

    void fill(float value);

    void print() const;

    void info() const;

    size_t size() const;

    void allocateGPU();

    void toGPU();

    void toCPU();

    void freeGPU();

    // Tensor Operations
    Tensor add(const Tensor& other) const;

    Tensor multiply(const Tensor& other) const;

    static Tensor zeros(const std::vector<int>& shape);

    static Tensor ones(const std::vector<int>& shape);
};
