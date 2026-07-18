#pragma once

#include "core/tensor.hpp"

class Module
{
public:
    virtual ~Module() = default;

    virtual Tensor forward(const Tensor& input) = 0;

    virtual Tensor backward(const Tensor& gradOutput) = 0;
};
