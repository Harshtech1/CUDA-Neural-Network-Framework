#include "../include/framework.hpp"
#include "../include/tensor.hpp"
#include "../include/version.hpp"

#include <iostream>

int main()
{
    std::cout << FRAMEWORK_NAME << std::endl;

    Framework framework;

    framework.initialize();

    std::cout << "\nCreating Tensor A\n";

    Tensor A({2,3});

    A.fill(2);

    A.print();

    std::cout << "\nCreating Tensor B\n";

    Tensor B({2,3});

    B.fill(5);

    B.print();

    std::cout << "\nA + B\n";

    Tensor C = A.add(B);

    C.print();

    std::cout << "\nA * B\n";

    Tensor D = A.multiply(B);

    D.print();

    std::cout << "\nZeros Tensor\n";

    Tensor Z = Tensor::zeros({2,3});

    Z.print();

    std::cout << "\nOnes Tensor\n";

    Tensor O = Tensor::ones({2,3});

    O.print();

    return 0;
}
