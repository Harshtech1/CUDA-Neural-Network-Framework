#include "loss/mse_loss.hpp"

float MSELoss::forward(const Tensor& prediction, const Tensor& target)
{
    float loss = 0.0f;

    for (int i = 0; i < prediction.rows(); i++)
    {
        for (int j = 0; j < prediction.cols(); j++)
        {
            float diff = prediction.get(i,j) - target.get(i,j);
            loss += diff * diff;
        }
    }

    return loss / (prediction.rows() * prediction.cols());
}

Tensor MSELoss::backward(const Tensor& prediction, const Tensor& target)
{
    Tensor grad({prediction.rows(), prediction.cols()});

    for (int i = 0; i < prediction.rows(); i++)
    {
        for (int j = 0; j < prediction.cols(); j++)
        {
            float value =
                2.0f * (prediction.get(i, j) - target.get(i, j)) /
                (prediction.rows() * prediction.cols());

            grad.set(i, j, value);
        }
    }

    return grad;
}
