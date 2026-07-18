#pragma once

#include <vector>

#include "nn/linear.hpp"

class SGD
{
private:
    std::vector<Linear*> layers;
    float learningRate;

public:
    SGD(const std::vector<Linear*>& layers, float lr);

    void step();
};
