#pragma once

#include <vector>
#include <string>

class Tensor
{
private:
    // Pointer to CPU memory
    float* data_;

    // Shape of the tensor (e.g., {2, 3})
    std::vector<int> shape_;

    // Total number of elements
    size_t size_;

    // Device name (CPU for now)
    std::string device_;

public:
    // Constructor
    Tensor(const std::vector<int>& shape);

    // Destructor
    ~Tensor();

    // Fill tensor with a value
    void fill(float value);

    // Print tensor
    void print() const;

    // Print tensor metadata
    void info() const;

    // Get total number of elements
    size_t size() const;
};
