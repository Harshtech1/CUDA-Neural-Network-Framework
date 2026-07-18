#include "../include/core/tensor.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

Tensor::Tensor(const std::vector<int>& shape)
{
    shape_ = shape;

    size_ = 1;

    for (int dim : shape_)
    {
        size_ *= dim;
    }

    cpu_data_ = new float[size_];

    gpu_data_ = nullptr;

    device_ = "CPU";

    std::cout << "CPU memory allocated (" << size_
              << " elements)" << std::endl;
}

Tensor::~Tensor()
{
    delete[] cpu_data_;

    if (gpu_data_ != nullptr)
    {
        cudaFree(gpu_data_);
    }

    std::cout << "Tensor memory released." << std::endl;
}

void Tensor::fill(float value)
{
    for (size_t i = 0; i < size_; i++)
    {
        cpu_data_[i] = value;
    }
}

Tensor Tensor::zeros(const std::vector<int>& shape)
{
    Tensor t(shape);

    t.fill(0.0f);

    return t;
}

Tensor Tensor::ones(const std::vector<int>& shape)
{
    Tensor t(shape);

    t.fill(1.0f);

    return t;
}

Tensor Tensor::add(const Tensor& other) const
{
    Tensor result(shape_);

    for(size_t i = 0; i < size_; i++)
    {
        result.cpu_data_[i] =
            cpu_data_[i] +
            other.cpu_data_[i];
    }

    return result;
}

Tensor Tensor::multiply(const Tensor& other) const
{
    Tensor result(shape_);

    for(size_t i = 0; i < size_; i++)
    {
        result.cpu_data_[i] =
            cpu_data_[i] *
            other.cpu_data_[i];
    }

    return result;
}

void Tensor::allocateGPU()
{
    cudaMalloc(&gpu_data_, size_ * sizeof(float));

    std::cout << "GPU memory allocated." << std::endl;
}

void Tensor::toGPU()
{
    cudaMemcpy(
        gpu_data_,
        cpu_data_,
        size_ * sizeof(float),
        cudaMemcpyHostToDevice
    );

    device_ = "GPU";

    std::cout << "Copied CPU -> GPU" << std::endl;
}

void Tensor::toCPU()
{
    cudaMemcpy(
        cpu_data_,
        gpu_data_,
        size_ * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    device_ = "CPU";

    std::cout << "Copied GPU -> CPU" << std::endl;
}

void Tensor::freeGPU()
{
    if(gpu_data_)
    {
        cudaFree(gpu_data_);

        gpu_data_ = nullptr;

        std::cout << "GPU memory released." << std::endl;
    }
}

void Tensor::print() const
{
    std::cout << "Tensor Data:\n";

    if(shape_.size() == 2)
    {
        int rows = shape_[0];
        int cols = shape_[1];

        for(int r = 0; r < rows; r++)
        {
            for(int c = 0; c < cols; c++)
            {
                std::cout << cpu_data_[r * cols + c] << " ";
            }

            std::cout << std::endl;
        }
    }
    else
    {
        for(size_t i = 0; i < size_; i++)
        {
            std::cout << cpu_data_[i] << " ";
        }

        std::cout << std::endl;
    }
}

void Tensor::info() const
{
    std::cout << "\n========== Tensor Info ==========\n";

    std::cout << "Device : "
              << device_
              << std::endl;

    std::cout << "Shape  : ";

    for(size_t i=0;i<shape_.size();i++)
    {
        std::cout << shape_[i];

        if(i != shape_.size()-1)
            std::cout << " x ";
    }

    std::cout << std::endl;

    std::cout << "Size   : "
              << size_
              << std::endl;

    std::cout << "=================================\n";
}

size_t Tensor::size() const
{
    return size_;
}

#include "../include/cuda/tensor_cuda.hpp"

float* Tensor::cpuData()
{
    return cpu_data_;
}

const float* Tensor::cpuData() const
{
    return cpu_data_;
}

float* Tensor::gpuData()
{
    return gpu_data_;
}

const float* Tensor::gpuData() const
{
    return gpu_data_;
}

Tensor Tensor::addCUDA(const Tensor& other) const
{
    Tensor result(shape_);

    const_cast<Tensor*>(this)->allocateGPU();
    const_cast<Tensor&>(other).allocateGPU();
    result.allocateGPU();

    const_cast<Tensor*>(this)->toGPU();
    const_cast<Tensor&>(other).toGPU();

    tensorAddCUDA(
        gpu_data_,
        other.gpu_data_,
        result.gpu_data_,
        static_cast<int>(size_)
    );

    result.toCPU();

    const_cast<Tensor*>(this)->freeGPU();
    const_cast<Tensor&>(other).freeGPU();
    result.freeGPU();

    return result;
}

#include "../include/cuda/gemm.hpp"

const std::vector<int>& Tensor::shape() const
{
    return shape_;
}

Tensor Tensor::matmul(const Tensor& other) const
{
    int M = shape_[0];
    int K = shape_[1];
    int N = other.shape_[1];

    Tensor result({M, N});

    const_cast<Tensor*>(this)->allocateGPU();
    const_cast<Tensor&>(other).allocateGPU();
    result.allocateGPU();

    const_cast<Tensor*>(this)->toGPU();
    const_cast<Tensor&>(other).toGPU();

    launchGEMM(
        gpu_data_,
        other.gpu_data_,
        result.gpu_data_,
        M,
        N,
        K
    );

    result.toCPU();

    const_cast<Tensor*>(this)->freeGPU();
    const_cast<Tensor&>(other).freeGPU();
    result.freeGPU();

    return result;
}

int Tensor::rows() const
{
    return shape_[0];
}

int Tensor::cols() const
{
    return shape_[1];
}

void Tensor::set(int row, int col, float value)
{
    if(row < 0 || row >= rows())
    {
        throw std::out_of_range("Row index out of range");
    }

    if(col < 0 || col >= cols())
    {
        throw std::out_of_range("Column index out of range");
    }

    cpu_data_[row * cols() + col] = value;
}

float Tensor::get(int row, int col) const
{
    if(row < 0 || row >= rows())
    {
        throw std::out_of_range("Row index out of range");
    }

    if(col < 0 || col >= cols())
    {
        throw std::out_of_range("Column index out of range");
    }

    return cpu_data_[row * cols() + col];
}

void Tensor::setGrad(Tensor* grad)
{
    grad_ = grad;
}

Tensor* Tensor::grad() const
{
    return grad_;
}

#include <algorithm>

// Copy Constructor
Tensor::Tensor(const Tensor& other)
{
    shape_ = other.shape_;
    size_ = other.size_;
    device_ = other.device_;
    grad_ = nullptr;

    cpu_data_ = new float[size_];
    std::copy(other.cpu_data_, other.cpu_data_ + size_, cpu_data_);

    gpu_data_ = nullptr;
}

// Copy Assignment
Tensor& Tensor::operator=(const Tensor& other)
{
    if (this == &other)
        return *this;

    delete[] cpu_data_;
    if (gpu_data_)
        cudaFree(gpu_data_);

    shape_ = other.shape_;
    size_ = other.size_;
    device_ = other.device_;
    grad_ = nullptr;

    cpu_data_ = new float[size_];
    std::copy(other.cpu_data_, other.cpu_data_ + size_, cpu_data_);

    gpu_data_ = nullptr;

    return *this;
}

// Move Constructor
Tensor::Tensor(Tensor&& other) noexcept
{
    shape_ = std::move(other.shape_);
    size_ = other.size_;
    device_ = std::move(other.device_);
    grad_ = other.grad_;

    cpu_data_ = other.cpu_data_;
    gpu_data_ = other.gpu_data_;

    other.cpu_data_ = nullptr;
    other.gpu_data_ = nullptr;
    other.grad_ = nullptr;
}

// Move Assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if (this == &other)
        return *this;

    delete[] cpu_data_;
    if (gpu_data_)
        cudaFree(gpu_data_);

    shape_ = std::move(other.shape_);
    size_ = other.size_;
    device_ = std::move(other.device_);
    grad_ = other.grad_;

    cpu_data_ = other.cpu_data_;
    gpu_data_ = other.gpu_data_;

    other.cpu_data_ = nullptr;
    other.gpu_data_ = nullptr;
    other.grad_ = nullptr;

    return *this;
}

