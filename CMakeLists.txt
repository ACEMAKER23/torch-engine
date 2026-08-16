cmake_minimum_required(VERSION 3.14)
project(TENSOR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------
# Dependencies
# -------------------------

include(FetchContent)
include(CheckLanguage)

option(TENSOR_FETCH_GTEST "Download GoogleTest if not found" ON)

find_package(GTest QUIET)
if(NOT GTest_FOUND)
    if(TENSOR_FETCH_GTEST)
        message(STATUS "GoogleTest not found - fetching via FetchContent")
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/release-1.14.0.zip
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    else()
        message(FATAL_ERROR "GoogleTest not found. Install GTest or set -DTENSOR_FETCH_GTEST=ON.")
    endif()
endif()

# -------------------------
# CUDA (optional)
# -------------------------

option(TENSOR_ENABLE_CUDA "Enable CUDA support when available" ON)

set(TENSOR_CUDA_ENABLED OFF)
if(TENSOR_ENABLE_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        # CMake requires this at configure-time for CUDA builds.
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures" FORCE)
        endif()

        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)

        set(TENSOR_CUDA_ENABLED ON)
        add_definitions(-DUSE_CUDA)
        message(STATUS "CUDA enabled")
    else()
        message(STATUS "CUDA compiler not available - building CPU-only version")
    endif()
else()
    message(STATUS "CUDA disabled - building CPU-only version")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Tensor library source files
set(TENSOR_SOURCES
    src/tensor/tensor.cpp
    src/tensor/tensor_imp.cpp
    src/tensor/tensor_storage.cpp
    src/core/grad_fn.cpp
    src/core/dtype_utils.cpp
    src/core/loss_scaler.cpp
    src/nn/linear.cpp
    src/nn/relu.cpp
    src/nn/gelu.cpp
    src/nn/embedding.cpp
    src/nn/layernorm.cpp
    src/nn/dropout.cpp
    src/nn/sequential.cpp
    src/nn/positional_embedding.cpp
    src/nn/scaledot.cpp
    src/nn/multi_head_attention.cpp
    src/nn/ffn.cpp
    src/nn/transformer_block.cpp
    src/nn/gpt.cpp
    src/optimizer/optimizer.cpp
    src/optimizer/sgd.cpp
    src/optimizer/adam.cpp
    src/optimizer/adamw.cpp
    src/loss/CrossEntropyLoss.cpp
    src/loss/MSELoss.cpp
    src/loss/BCELoss.cpp
    src/loss/L1Loss.cpp
)

if(TENSOR_CUDA_ENABLED)
    add_executable(cuda_test src/tests/cuda_test.cpp)
    list(APPEND TENSOR_SOURCES
        src/core/cuda_utils.cpp
        src/cuda/elementwise_cuda.cu
    )
endif()

add_library(tensor_lib STATIC ${TENSOR_SOURCES})

if(TENSOR_CUDA_ENABLED)
    add_executable(cuda_test src/tests/cuda_test.cpp)
    target_link_libraries(tensor_lib PRIVATE CUDA::cudart CUDA::cublas)
endif()

# -------------------------
# Tests and benchmarks
# -------------------------

add_executable(tensor_test src/tests/tensor_test.cpp)
add_executable(autograd_test src/tests/autograd_test.cpp)
add_executable(linear_test src/tests/linear_test.cpp)
add_executable(nn_layers_test src/tests/nn_layers_test.cpp)
add_executable(attention_test src/tests/attention_test.cpp)
add_executable(transformer_test src/tests/transformer_test.cpp)
add_executable(optimizer_test src/tests/optimizer_test.cpp)
add_executable(loss_test src/tests/loss_test.cpp)
add_executable(mixed_precision_test src/tests/mixed_precision_test.cpp)
add_executable(end_to_end_test src/tests/end_to_end_test.cpp)
add_executable(mha_grad_check2 src/tests/mha_grad_check2.cpp)

if(TENSOR_CUDA_ENABLED)
    add_executable(cuda_test src/tests/cuda_test.cpp)
    add_executable(matmul_benchmark src/benchmarks/matmul_benchmark.cpp)
    target_link_libraries(matmul_benchmark PRIVATE tensor_lib CUDA::cudart CUDA::cublas)

    add_executable(matmul_profile src/benchmarks/matmul_profile.cpp)
    target_link_libraries(matmul_profile PRIVATE tensor_lib CUDA::cudart CUDA::cublas)

    add_executable(gpt_benchmark src/benchmarks/gpt_benchmark.cpp)
    target_link_libraries(gpt_benchmark PRIVATE tensor_lib CUDA::cudart CUDA::cublas)

    add_executable(gpt_benchmark_gpu src/benchmarks/gpt_benchmark_gpu.cpp)
    target_link_libraries(gpt_benchmark_gpu PRIVATE tensor_lib CUDA::cudart CUDA::cublas)

    add_executable(cuda_matmul_tensor_test src/tests/cuda_matmul_tensor_test.cpp)
    target_link_libraries(cuda_matmul_tensor_test PRIVATE tensor_lib CUDA::cudart CUDA::cublas)

    add_executable(cuda_softmax_tensor_test src/tests/cuda_softmax_tensor_test.cpp)
    target_link_libraries(cuda_softmax_tensor_test PRIVATE tensor_lib CUDA::cudart CUDA::cublas)

    add_executable(cuda_layernorm_tensor_test src/tests/cuda_layernorm_tensor_test.cpp)
    target_link_libraries(cuda_layernorm_tensor_test PRIVATE tensor_lib CUDA::cudart CUDA::cublas)

    add_executable(cuda_embedding_tensor_test src/tests/cuda_embedding_tensor_test.cpp)
    target_link_libraries(cuda_embedding_tensor_test PRIVATE tensor_lib CUDA::cudart CUDA::cublas)

    add_executable(cuda_crossentropy_tensor_test src/tests/cuda_crossentropy_tensor_test.cpp)
    target_link_libraries(cuda_crossentropy_tensor_test PRIVATE tensor_lib CUDA::cudart CUDA::cublas)

    add_executable(cuda_gelu_tensor_test src/tests/cuda_gelu_tensor_test.cpp)
    target_link_libraries(cuda_gelu_tensor_test PRIVATE tensor_lib CUDA::cudart CUDA::cublas)

    add_executable(cuda_reduction_tensor_test src/tests/cuda_reduction_tensor_test.cpp)
    target_link_libraries(cuda_reduction_tensor_test PRIVATE tensor_lib CUDA::cudart CUDA::cublas)
endif()

# Link GoogleTest
function(tensor_link_gtest target)
    if(TARGET GTest::GTest)
        target_link_libraries(${target} PRIVATE tensor_lib GTest::GTest GTest::Main)
    else()
        target_link_libraries(${target} PRIVATE tensor_lib gtest gtest_main)
    endif()
endfunction()

tensor_link_gtest(tensor_test)
tensor_link_gtest(autograd_test)
tensor_link_gtest(linear_test)
tensor_link_gtest(nn_layers_test)
tensor_link_gtest(attention_test)
tensor_link_gtest(transformer_test)
tensor_link_gtest(optimizer_test)
tensor_link_gtest(loss_test)
tensor_link_gtest(mixed_precision_test)
tensor_link_gtest(end_to_end_test)

if(TENSOR_CUDA_ENABLED)
    tensor_link_gtest(cuda_test)
    target_link_libraries(cuda_test PRIVATE CUDA::cudart)
    add_test(NAME CUDATest COMMAND cuda_test)
endif()

# Non-test helper
if(TENSOR_CUDA_ENABLED)
    add_executable(cuda_test src/tests/cuda_test.cpp)
    target_link_libraries(mha_grad_check2 PRIVATE tensor_lib CUDA::cudart)
else()
    target_link_libraries(mha_grad_check2 PRIVATE tensor_lib)
endif()

enable_testing()

add_test(NAME TensorTest COMMAND tensor_test)
add_test(NAME LinearTest COMMAND linear_test)
add_test(NAME NNLayersTest COMMAND nn_layers_test)
add_test(NAME AttentionTest COMMAND attention_test)
add_test(NAME TransformerTest COMMAND transformer_test)
add_test(NAME OptimizerTest COMMAND optimizer_test)
add_test(NAME LossTest COMMAND loss_test)
add_test(NAME MixedPrecisionTest COMMAND mixed_precision_test)
add_test(NAME EndToEndTest COMMAND end_to_end_test)

