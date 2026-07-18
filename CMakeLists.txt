cmake_minimum_required(VERSION 3.18)

project(
    CUDA_Neural_Network_Framework
    VERSION 0.1.1
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(include)

add_executable(
    framework

    src/main.cpp
    src/framework.cpp
    src/tensor.cpp
)

set_target_properties(framework PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
