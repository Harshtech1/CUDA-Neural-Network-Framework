#pragma once

#include "nn/module.hpp"
#include "nn/parameter.hpp"

class Linear : public Module
{
private:
    Parameter weight;
    Parameter bias;

    Tensor lastInput;
    Tensor weightGrad;
    Tensor biasGrad;

public:
    Linear(int inFeatures, int outFeatures);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;

    Parameter& getWeight();
    Parameter& getBias();

    Tensor& getWeightGrad();
    Tensor& getBiasGrad();
};
