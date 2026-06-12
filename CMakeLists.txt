cmake_minimum_required(VERSION 3.14)
project(TENSOR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find GoogleTest
find_package(GTest REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Tensor library source files
set(TENSOR_SOURCES
    src/tensor/tensor.cpp
    src/tensor/tensor_imp.cpp
    src/tensor/tensor_storage.cpp
)

# Create Tensor library
add_library(tensor_lib STATIC ${TENSOR_SOURCES})

# Test executable
add_executable(tensor_test src/tests/tensor_test.cpp)

# Link libraries
target_link_libraries(tensor_test tensor_lib GTest::GTest GTest::Main)

# Enable testing
enable_testing()

# Add test
add_test(NAME TensorTest COMMAND tensor_test)
