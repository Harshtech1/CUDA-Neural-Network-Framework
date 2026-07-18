cmake_minimum_required(VERSION 3.18)

project(
    CUDA_Neural_Network_Framework
    VERSION 0.6.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(include)

find_package(CUDAToolkit REQUIRED)

add_executable(
    framework

    src/main.cpp
    src/framework.cpp
    src/tensor.cpp
    src/linear.cpp
    src/parameter.cpp
    src/relu.cpp
    src/sequential.cpp
    src/mse_loss.cpp
    src/sgd.cpp

    cuda/tensor_add.cu
    cuda/gemm.cu
)

target_link_libraries(
    framework
    PRIVATE
    CUDA::cudart
)

set_target_properties(
    framework
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
