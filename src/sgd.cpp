#include "optim/sgd.hpp"

SGD::SGD(const std::vector<Linear*>& l, float lr)
    : layers(l), learningRate(lr)
{
}

void SGD::step()
{
    for (auto layer : layers)
    {
        layer->getWeight().applyGradient(
            layer->getWeightGrad(),
            learningRate);

        layer->getBias().applyGradient(
            layer->getBiasGrad(),
            learningRate);
    }
}
