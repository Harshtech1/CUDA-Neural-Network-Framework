#pragma once

#include "core/tensor.hpp"

class MSELoss
{
public:
    float forward(const Tensor& prediction, const Tensor& target);

    Tensor backward(const Tensor& prediction, const Tensor& target);
};
