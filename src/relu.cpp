#include "nn/relu.hpp"

Tensor ReLU::forward(const Tensor& input)
{
    lastInput = input;

    Tensor output = input;

    for (int i = 0; i < output.rows(); i++)
    {
        for (int j = 0; j < output.cols(); j++)
        {
            if (output.get(i, j) < 0)
                output.set(i, j, 0);
        }
    }

    return output;
}

Tensor ReLU::backward(const Tensor& gradOutput)
{
    Tensor grad = gradOutput;

    for (int i = 0; i < grad.rows(); i++)
    {
        for (int j = 0; j < grad.cols(); j++)
        {
            if (lastInput.get(i, j) <= 0)
                grad.set(i, j, 0);
        }
    }

    return grad;
}
