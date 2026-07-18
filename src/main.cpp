#include <iostream>

#include "../include/framework.hpp"
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

    return 0;
}
