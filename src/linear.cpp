#include "nn/linear.hpp"

Linear::Linear(int inFeatures, int outFeatures)
    : weight({inFeatures, outFeatures}),
      bias({1, outFeatures}),
      lastInput({1, 1}),
      weightGrad({inFeatures, outFeatures}),
      biasGrad({1, outFeatures})
{
    weight.fill(0.1f);
    bias.fill(0.0f);
}

Tensor Linear::forward(const Tensor& input)
{
    lastInput = input;

    Tensor output = input.matmul(weight);

    return output;
}

Tensor Linear::backward(const Tensor& gradOutput)
{
    // Compute dL/dW = X^T * dL/dY

    for (int i = 0; i < weightGrad.rows(); i++)
    {
        for (int j = 0; j < weightGrad.cols(); j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < lastInput.rows(); k++)
            {
                sum += lastInput.get(k, i) * gradOutput.get(k, j);
            }

            weightGrad.set(i, j, sum);
        }
    }

    // Compute dL/dB

    for (int j = 0; j < biasGrad.cols(); j++)
    {
        float sum = 0.0f;

        for (int i = 0; i < gradOutput.rows(); i++)
            sum += gradOutput.get(i, j);

        biasGrad.set(0, j, sum);
    }

    Tensor gradInput({gradOutput.rows(), weight.rows()});

    for (int i = 0; i < gradInput.rows(); i++)
    {
        for (int j = 0; j < gradInput.cols(); j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < gradOutput.cols(); k++)
            {
                sum += gradOutput.get(i, k) * weight.get(j, k);
            }

            gradInput.set(i, j, sum);
        }
    }

    return gradInput;
}

Parameter& Linear::getWeight()
{
    return weight;
}

Parameter& Linear::getBias()
{
    return bias;
}

Tensor& Linear::getWeightGrad()
{
    return weightGrad;
}

Tensor& Linear::getBiasGrad()
{
    return biasGrad;
}
