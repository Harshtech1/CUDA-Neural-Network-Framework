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

    std::cout << "\nCreating Tensor...\n" << std::endl;

    Tensor tensor({2, 3});

    tensor.info();

    tensor.fill(5.0f);

    tensor.print();

    return 0;
}
