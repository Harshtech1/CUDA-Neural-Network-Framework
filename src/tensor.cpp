#include "../include/tensor.hpp"

#include <cuda_runtime.h>

#include <iostream>

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

    for(size_t i = 0; i < size_; i++)
    {
        std::cout << cpu_data_[i] << " ";
    }

    std::cout << std::endl;
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
