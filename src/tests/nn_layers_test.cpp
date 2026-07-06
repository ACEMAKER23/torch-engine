#include <gtest/gtest.h>
#include "../nn/embedding.h"
#include "../nn/layernorm.h"
#include "../nn/dropout.h"
#include "../nn/relu.h"
#include "../nn/gelu.h"
#include "../nn/sequential.h"
#include "../nn/linear.h"
#include "../tensor/tensor.h"
#include "../core/dtype.h"

TEST(EmbeddingTest, BasicConstruction) {
    Embedding embedding(100, 16, DType::Float32);
    
    EXPECT_EQ(embedding.weight().shape()[0], 100);
    EXPECT_EQ(embedding.weight().shape()[1], 16);
}

TEST(EmbeddingTest, ForwardPass1D) {
    Embedding embedding(10, 4, DType::Float32);
    
    // Create input tensor [seq_len=3] with indices
    Tensor input({3}, DType::Int64, Device::CPU);
    auto* input_data = static_cast<int64_t*>(input.data());
    input_data[0] = 0;
    input_data[1] = 2;
    input_data[2] = 5;
    
    Tensor output = embedding.forward(input);
    
    // Output should be [seq_len=3, embedding_dim=4]
    EXPECT_EQ(output.shape()[0], 3);
    EXPECT_EQ(output.shape()[1], 4);
}

TEST(EmbeddingTest, ForwardPass2D) {
    Embedding embedding(10, 4, DType::Float32);
    
    // Create input tensor [batch_size=2, seq_len=3] with indices
    Tensor input({2, 3}, DType::Int64, Device::CPU);
    auto* input_data = static_cast<int64_t*>(input.data());
    input_data[0] = 0; input_data[1] = 1; input_data[2] = 2;
    input_data[3] = 3; input_data[4] = 4; input_data[5] = 5;
    
    Tensor output = embedding.forward(input);
    
    // Output should be [batch_size=2, seq_len=3, embedding_dim=4]
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 4);
}

TEST(EmbeddingTest, IndexOutOfBounds) {
    Embedding embedding(10, 4, DType::Float32);
    
    Tensor input({1}, DType::Int64, Device::CPU);
    auto* input_data = static_cast<int64_t*>(input.data());
    input_data[0] = 15;  // Out of bounds
    
    EXPECT_THROW(embedding.forward(input), std::runtime_error);
}

TEST(LayerNormTest, BasicConstruction) {
    LayerNorm layernorm(4, DType::Float32);
    
    EXPECT_EQ(layernorm.weight().shape()[0], 4);
    EXPECT_EQ(layernorm.bias().shape()[0], 4);
}

TEST(LayerNormTest, ForwardPass) {
    LayerNorm layernorm(4, DType::Float32);
    
    // Create input tensor [batch_size=2, normalized_shape=4]
    Tensor input({2, 4}, DType::Float32, Device::CPU);
    auto* input_data = static_cast<float*>(input.data());
    input_data[0] = 1.0f; input_data[1] = 2.0f; input_data[2] = 3.0f; input_data[3] = 4.0f;
    input_data[4] = 5.0f; input_data[5] = 6.0f; input_data[6] = 7.0f; input_data[7] = 8.0f;
    
    Tensor output = layernorm.forward(input);
    
    // Output should have same shape as input
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 4);
}

TEST(LayerNormTest, WeightInitialization) {
    LayerNorm layernorm(4, DType::Float32);
    
    // Check that weight is initialized to 1.0
    const auto* weight_data = static_cast<const float*>(layernorm.weight().data());
    for (size_t i = 0; i < layernorm.weight().numel(); ++i) {
        EXPECT_FLOAT_EQ(weight_data[i], 1.0f);
    }
}

TEST(LayerNormTest, BiasInitialization) {
    LayerNorm layernorm(4, DType::Float32);
    
    // Check that bias is initialized to 0.0
    const auto* bias_data = static_cast<const float*>(layernorm.bias().data());
    for (size_t i = 0; i < layernorm.bias().numel(); ++i) {
        EXPECT_FLOAT_EQ(bias_data[i], 0.0f);
    }
}

TEST(LayerNormTest, InvalidNormalizedShape) {
    EXPECT_THROW(LayerNorm layernorm(0, DType::Float32), std::runtime_error);
}

TEST(DropoutTest, BasicConstruction) {
    Dropout dropout(0.5);
    
    EXPECT_NO_THROW();
}

TEST(DropoutTest, ForwardPassTraining) {
    Dropout dropout(0.5);
    dropout.set_training(true);
    
    // Create input tensor
    Tensor input({4}, DType::Float32, Device::CPU);
    auto* input_data = static_cast<float*>(input.data());
    input_data[0] = 1.0f; input_data[1] = 2.0f; input_data[2] = 3.0f; input_data[3] = 4.0f;
    
    Tensor output = dropout.forward(input);
    
    // Output should have same shape as input
    EXPECT_EQ(output.shape()[0], 4);
}

TEST(DropoutTest, ForwardPassInference) {
    Dropout dropout(0.5);
    dropout.set_training(false);
    
    // Create input tensor
    Tensor input({4}, DType::Float32, Device::CPU);
    auto* input_data = static_cast<float*>(input.data());
    input_data[0] = 1.0f; input_data[1] = 2.0f; input_data[2] = 3.0f; input_data[3] = 4.0f;
    
    Tensor output = dropout.forward(input);
    
    // Output should have same shape as input
    EXPECT_EQ(output.shape()[0], 4);
    
    // During inference, output should be unchanged (no scaling)
    const auto* output_data = static_cast<const float*>(output.data());
    EXPECT_FLOAT_EQ(output_data[0], 1.0f);
    EXPECT_FLOAT_EQ(output_data[1], 2.0f);
    EXPECT_FLOAT_EQ(output_data[2], 3.0f);
    EXPECT_FLOAT_EQ(output_data[3], 4.0f);
}

TEST(DropoutTest, InvalidProbability) {
    EXPECT_THROW(Dropout dropout(1.5), std::runtime_error);
    EXPECT_THROW(Dropout dropout(-0.5), std::runtime_error);
}

TEST(ReLUTest, ForwardPass) {
    ReLU relu;
    
    // Create input tensor
    Tensor input({4}, DType::Float32, Device::CPU);
    auto* input_data = static_cast<float*>(input.data());
    input_data[0] = -1.0f; input_data[1] = 0.0f; input_data[2] = 1.0f; input_data[3] = 2.0f;
    
    Tensor output = relu.forward(input);
    
    // Output should have same shape as input
    EXPECT_EQ(output.shape()[0], 4);
    
    // ReLU should set negative values to 0
    const auto* output_data = static_cast<const float*>(output.data());
    EXPECT_FLOAT_EQ(output_data[0], 0.0f);
    EXPECT_FLOAT_EQ(output_data[1], 0.0f);
    EXPECT_FLOAT_EQ(output_data[2], 1.0f);
    EXPECT_FLOAT_EQ(output_data[3], 2.0f);
}

TEST(GeLUTest, ForwardPass) {
    GeLU gelu;
    
    // Create input tensor
    Tensor input({4}, DType::Float32, Device::CPU);
    auto* input_data = static_cast<float*>(input.data());
    input_data[0] = -1.0f; input_data[1] = 0.0f; input_data[2] = 1.0f; input_data[3] = 2.0f;
    
    Tensor output = gelu.forward(input);
    
    // Output should have same shape as input
    EXPECT_EQ(output.shape()[0], 4);
    
    // GeLU should apply Gaussian Error Linear Unit
    const auto* output_data = static_cast<const float*>(output.data());
    // GeLU(0) = 0
    EXPECT_FLOAT_EQ(output_data[1], 0.0f);
    // GeLU should be smooth and approximately linear for positive values
    EXPECT_GT(output_data[2], 0.0f);
    EXPECT_GT(output_data[3], 0.0f);
}

TEST(SequentialTest, BasicLayers) {
    Sequential seq;
    seq.add(std::make_shared<Linear>(4, 8, DType::Float32));
    seq.add(std::make_shared<ReLU>());
    seq.add(std::make_shared<Linear>(8, 4, DType::Float32));
    
    // Create input tensor
    Tensor input({2, 4}, DType::Float32, Device::CPU);
    auto* input_data = static_cast<float*>(input.data());
    input_data[0] = 1.0f; input_data[1] = 2.0f; input_data[2] = 3.0f; input_data[3] = 4.0f;
    input_data[4] = 5.0f; input_data[5] = 6.0f; input_data[6] = 7.0f; input_data[7] = 8.0f;
    
    Tensor output = seq.forward(input);
    
    // Output should have same shape as input [2, 4]
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 4);
}
