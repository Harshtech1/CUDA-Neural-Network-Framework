#include <iostream>

#include "../include/framework.hpp"
#include "../include/tensor.hpp"
#include "../include/version.hpp"

int main()
{
    std::cout << "=========================================\n";
    std::cout << FRAMEWORK_NAME << "\n";
    std::cout << "Version "
              << FRAMEWORK_VERSION_MAJOR << "."
              << FRAMEWORK_VERSION_MINOR << "."
              << FRAMEWORK_VERSION_PATCH << "\n";
    std::cout << "=========================================\n\n";

    Framework framework;

    framework.initialize();

    Tensor tensor({2,3});

    tensor.fill(7.0f);

    tensor.info();

    tensor.print();

    tensor.allocateGPU();

    tensor.toGPU();

    tensor.toCPU();

    tensor.freeGPU();

    return 0;
}
