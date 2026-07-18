#pragma once

#include "nn/module.hpp"

class ReLU : public Module
{
private:
    Tensor lastInput;

public:
    ReLU() : lastInput({1,1}) {}

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;
};
