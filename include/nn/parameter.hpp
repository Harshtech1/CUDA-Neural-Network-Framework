#pragma once

#include "core/tensor.hpp"

class Parameter : public Tensor
{
public:
    using Tensor::Tensor;

    bool requiresGrad = true;

    void applyGradient(const Tensor& grad, float lr);
};
