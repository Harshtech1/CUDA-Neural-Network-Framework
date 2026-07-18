#pragma once

#include <string>
#include <vector>

class Tensor
{
private:
    Tensor* grad_ = nullptr;
    float* cpu_data_;
    float* gpu_data_;

    std::vector<int> shape_;

    size_t size_;

    std::string device_;

public:
    Tensor(const std::vector<int>& shape);

    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor();

    void fill(float value);

    void print() const;

    void info() const;

    size_t size() const;

    const std::vector<int>& shape() const;

    int rows() const;

    int cols() const;

    void set(int row, int col, float value);

    float get(int row, int col) const;

    void allocateGPU();

    void toGPU();

    void toCPU();

    void freeGPU();

    Tensor add(const Tensor& other) const;

    Tensor multiply(const Tensor& other) const;

    Tensor addCUDA(const Tensor& other) const;

    Tensor matmul(const Tensor& other) const;

    float* cpuData();

    const float* cpuData() const;

    float* gpuData();

    const float* gpuData() const;

    static Tensor zeros(const std::vector<int>& shape);

    static Tensor ones(const std::vector<int>& shape);


    void setGrad(Tensor* grad);
    Tensor* grad() const;

};
