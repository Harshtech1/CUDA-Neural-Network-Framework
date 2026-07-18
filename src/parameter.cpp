#include "nn/parameter.hpp"

void Parameter::applyGradient(const Tensor& grad, float lr)
{
    for (int i = 0; i < rows(); i++)
    {
        for (int j = 0; j < cols(); j++)
        {
            float value = get(i, j);
            value -= lr * grad.get(i, j);
            set(i, j, value);
        }
    }
}
