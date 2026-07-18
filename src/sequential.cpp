#include "nn/sequential.hpp"

void Sequential::add(std::shared_ptr<Module> layer)
{
    layers.push_back(layer);
}

Tensor Sequential::forward(const Tensor& input)
{
    Tensor output = input;

    for (auto& layer : layers)
    {
        output = layer->forward(output);
    }

    return output;
}

Tensor Sequential::backward(const Tensor& gradOutput)
{
    Tensor grad = gradOutput;

    for (auto it = layers.rbegin(); it != layers.rend(); ++it)
    {
        grad = (*it)->backward(grad);
    }

    return grad;
}
