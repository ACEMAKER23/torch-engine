cmake_minimum_required(VERSION 3.14)
project(TENSOR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find GoogleTest
find_package(GTest REQUIRED)

# Find CUDA (optional)
find_package(CUDA)
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA found - building with CUDA support")
else()
    message(STATUS "CUDA not found - building CPU-only version")
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

# Add CUDA sources if CUDA is found
if(CUDA_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 80)
    list(APPEND TENSOR_SOURCES src/core/cuda_utils.cpp)
    list(APPEND TENSOR_SOURCES src/cuda/elementwise_cuda.cu)
endif()

# Create Tensor library
add_library(tensor_lib STATIC ${TENSOR_SOURCES})
if(CUDA_FOUND)
    target_link_libraries(tensor_lib ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})
endif()

# Test executable
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
add_executable(cuda_test src/tests/cuda_test.cpp)
add_executable(mha_grad_check2 src/tests/mha_grad_check2.cpp)

if(CUDA_FOUND)
    add_executable(matmul_benchmark src/benchmarks/matmul_benchmark.cpp)
    target_link_libraries(matmul_benchmark tensor_lib ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})

    add_executable(matmul_profile src/benchmarks/matmul_profile.cpp)
    target_link_libraries(matmul_profile tensor_lib ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})

    add_executable(gpt_benchmark src/benchmarks/gpt_benchmark.cpp)
    target_link_libraries(gpt_benchmark tensor_lib ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})

    add_executable(cuda_matmul_tensor_test src/tests/cuda_matmul_tensor_test.cpp)
    target_link_libraries(cuda_matmul_tensor_test tensor_lib ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})

    add_executable(cuda_softmax_tensor_test src/tests/cuda_softmax_tensor_test.cpp)
    target_link_libraries(cuda_softmax_tensor_test tensor_lib ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})

    add_executable(cuda_layernorm_tensor_test src/tests/cuda_layernorm_tensor_test.cpp)
    target_link_libraries(cuda_layernorm_tensor_test tensor_lib ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})

    add_executable(cuda_embedding_tensor_test src/tests/cuda_embedding_tensor_test.cpp)
    target_link_libraries(cuda_embedding_tensor_test tensor_lib ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})

    add_executable(cuda_crossentropy_tensor_test src/tests/cuda_crossentropy_tensor_test.cpp)
    target_link_libraries(cuda_crossentropy_tensor_test tensor_lib ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})

    add_executable(cuda_gelu_tensor_test src/tests/cuda_gelu_tensor_test.cpp)
    target_link_libraries(cuda_gelu_tensor_test tensor_lib ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})

    add_executable(cuda_reduction_tensor_test src/tests/cuda_reduction_tensor_test.cpp)
    target_link_libraries(cuda_reduction_tensor_test tensor_lib ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})
endif()

# Link libraries
if(CUDA_FOUND)
    target_link_libraries(tensor_test tensor_lib GTest::GTest GTest::Main ${CUDA_LIBRARIES})
    target_link_libraries(autograd_test tensor_lib GTest::GTest GTest::Main ${CUDA_LIBRARIES})
    target_link_libraries(linear_test tensor_lib GTest::GTest GTest::Main ${CUDA_LIBRARIES})
    target_link_libraries(nn_layers_test tensor_lib GTest::GTest GTest::Main ${CUDA_LIBRARIES})
    target_link_libraries(attention_test tensor_lib GTest::GTest GTest::Main ${CUDA_LIBRARIES})
    target_link_libraries(transformer_test tensor_lib GTest::GTest GTest::Main ${CUDA_LIBRARIES})
    target_link_libraries(optimizer_test tensor_lib GTest::GTest GTest::Main ${CUDA_LIBRARIES})
    target_link_libraries(loss_test tensor_lib GTest::GTest GTest::Main ${CUDA_LIBRARIES})
    target_link_libraries(mixed_precision_test tensor_lib GTest::GTest GTest::Main ${CUDA_LIBRARIES})
    target_link_libraries(end_to_end_test tensor_lib GTest::GTest GTest::Main ${CUDA_LIBRARIES})
    target_link_libraries(cuda_test tensor_lib GTest::GTest GTest::Main ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})
    target_link_libraries(mha_grad_check2 tensor_lib ${CUDA_LIBRARIES})
else()
    target_link_libraries(tensor_test tensor_lib GTest::GTest GTest::Main)
    target_link_libraries(autograd_test tensor_lib GTest::GTest GTest::Main)
    target_link_libraries(linear_test tensor_lib GTest::GTest GTest::Main)
    target_link_libraries(nn_layers_test tensor_lib GTest::GTest GTest::Main)
    target_link_libraries(attention_test tensor_lib GTest::GTest GTest::Main)
    target_link_libraries(transformer_test tensor_lib GTest::GTest GTest::Main)
    target_link_libraries(optimizer_test tensor_lib GTest::GTest GTest::Main)
    target_link_libraries(loss_test tensor_lib GTest::GTest GTest::Main)
    target_link_libraries(mixed_precision_test tensor_lib GTest::GTest GTest::Main)
    target_link_libraries(end_to_end_test tensor_lib GTest::GTest GTest::Main)
    target_link_libraries(cuda_test tensor_lib GTest::GTest GTest::Main)
    target_link_libraries(mha_grad_check2 tensor_lib)
endif()

# Enable testing
enable_testing()

# Add test
add_test(NAME TensorTest COMMAND tensor_test)
add_test(NAME LinearTest COMMAND linear_test)
add_test(NAME NNLayersTest COMMAND nn_layers_test)
add_test(NAME AttentionTest COMMAND attention_test)
add_test(NAME TransformerTest COMMAND transformer_test)
add_test(NAME OptimizerTest COMMAND optimizer_test)
add_test(NAME LossTest COMMAND loss_test)
add_test(NAME MixedPrecisionTest COMMAND mixed_precision_test)
add_test(NAME EndToEndTest COMMAND end_to_end_test)
add_test(NAME CUDATest COMMAND cuda_test)
