cmake_minimum_required(VERSION 3.14)
project(TENSOR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find GoogleTest
find_package(GTest REQUIRED)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Tensor library source files
set(TENSOR_SOURCES
    src/tensor/tensor.cpp
    src/tensor/tensor_imp.cpp
    src/tensor/tensor_storage.cpp
    src/core/grad_fn.cpp
    src/core/cuda_utils.cpp
    src/nn/linear.cpp
    src/nn/relu.cpp
    src/nn/gelu.cpp
    src/nn/embedding.cpp
    src/nn/layernorm.cpp
    src/nn/dropout.cpp
)

# Create Tensor library
add_library(tensor_lib STATIC ${TENSOR_SOURCES})

# Test executable
add_executable(tensor_test src/tests/tensor_test.cpp)
add_executable(autograd_test src/tests/autograd_test.cpp)
add_executable(linear_test src/tests/linear_test.cpp)
add_executable(nn_layers_test src/tests/nn_layers_test.cpp)

# Link libraries
target_link_libraries(tensor_test tensor_lib GTest::GTest GTest::Main ${CUDA_LIBRARIES})
target_link_libraries(autograd_test tensor_lib GTest::GTest GTest::Main ${CUDA_LIBRARIES})
target_link_libraries(linear_test tensor_lib GTest::GTest GTest::Main ${CUDA_LIBRARIES})
target_link_libraries(nn_layers_test tensor_lib GTest::GTest GTest::Main ${CUDA_LIBRARIES})

# Enable testing
enable_testing()

# Add test
add_test(NAME TensorTest COMMAND tensor_test)
add_test(NAME LinearTest COMMAND linear_test)
add_test(NAME NNLayersTest COMMAND nn_layers_test)
