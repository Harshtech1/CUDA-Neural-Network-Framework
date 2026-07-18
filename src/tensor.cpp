#include "../include/tensor.hpp"

#include <iostream>

Tensor::Tensor(const std::vector<int>& shape)
{
    shape_ = shape;
    device_ = "CPU";

    size_ = 1;
    for (int dim : shape_)
    {
        size_ *= dim;
    }

    data_ = new float[size_];

    std::cout << "Tensor allocated on CPU (" << size_
              << " elements)" << std::endl;
}

Tensor::~Tensor()
{
    delete[] data_;

    std::cout << "Tensor memory released." << std::endl;
}

void Tensor::fill(float value)
{
    for (size_t i = 0; i < size_; i++)
    {
        data_[i] = value;
    }
}

void Tensor::print() const
{
    std::cout << "Tensor Data:" << std::endl;

    for (size_t i = 0; i < size_; i++)
    {
        std::cout << data_[i] << " ";
    }

    std::cout << std::endl;
}

void Tensor::info() const
{
    std::cout << "\n========== Tensor Info ==========\n";

    std::cout << "Device : " << device_ << std::endl;

    std::cout << "Shape  : ";

    for (size_t i = 0; i < shape_.size(); i++)
    {
        std::cout << shape_[i];

        if (i != shape_.size() - 1)
            std::cout << " x ";
    }

    std::cout << std::endl;

    std::cout << "Size   : " << size_ << std::endl;

    std::cout << "=================================\n";
}

size_t Tensor::size() const
{
    return size_;
}
