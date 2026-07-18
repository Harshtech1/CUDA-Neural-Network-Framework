#pragma once

#include <memory>
#include <vector>

#include "nn/module.hpp"

class Sequential : public Module
{
private:
    std::vector<std::shared_ptr<Module>> layers;

public:
    void add(std::shared_ptr<Module> layer);

    Tensor forward(const Tensor& input) override;

    Tensor backward(const Tensor& gradOutput) override;
};
