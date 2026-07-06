#include <gtest/gtest.h>
#include "../nn/ffn.h"
#include "../nn/transformer_block.h"
#include "../nn/gpt.h"
#include "../tensor/tensor.h"
#include "../core/dtype.h"

// FeedForward Tests
TEST(FeedForwardTest, BasicConstruction) {
    FeedForward ffn(128, 512, DType::Float32);
    
    EXPECT_NO_THROW();
}

TEST(FeedForwardTest, ForwardBasic) {
    FeedForward ffn(128, 512, DType::Float32);
    
    Tensor input({2, 10, 128}, DType::Float32, Device::CPU);
    auto* input_data = static_cast<float*>(input.data());
    for (size_t i = 0; i < input.numel(); ++i) {
        input_data[i] = 0.1f * static_cast<float>(i);
    }
    
    Tensor output = ffn.forward(input);
    
    // Output should have same shape as input
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 10);
    EXPECT_EQ(output.shape()[2], 128);
}

TEST(FeedForwardTest, ForwardSmallInput) {
    FeedForward ffn(4, 8, DType::Float32);
    
    Tensor input({1, 1, 4}, DType::Float32, Device::CPU);
    auto* input_data = static_cast<float*>(input.data());
    input_data[0] = 1.0f; input_data[1] = 2.0f; input_data[2] = 3.0f; input_data[3] = 4.0f;
    
    Tensor output = ffn.forward(input);
    
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 1);
    EXPECT_EQ(output.shape()[2], 4);
}

TEST(FeedForwardTest, ForwardLargeInput) {
    FeedForward ffn(256, 1024, DType::Float32);
    
    Tensor input({4, 100, 256}, DType::Float32, Device::CPU);
    auto* input_data = static_cast<float*>(input.data());
    for (size_t i = 0; i < input.numel(); ++i) {
        input_data[i] = 0.01f;
    }
    
    Tensor output = ffn.forward(input);
    
    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 100);
    EXPECT_EQ(output.shape()[2], 256);
}

TEST(FeedForwardTest, Parameters) {
    FeedForward ffn(128, 512, DType::Float32);
    
    auto params = ffn.parameters();
    
    // Should have 4 parameters: linear1 weight, linear1 bias, linear2 weight, linear2 bias
    EXPECT_EQ(params.size(), 4);
}

TEST(FeedForwardTest, ZeroGrad) {
    FeedForward ffn(128, 512, DType::Float32);
    
    EXPECT_NO_THROW(ffn.zero_grad());
}

// TransformerBlock Tests
TEST(TransformerBlockTest, BasicConstruction) {
    TransformerBlock block(128, 8, 512, DType::Float32, 0.1);
    
    EXPECT_NO_THROW();
}

TEST(TransformerBlockTest, ForwardBasic) {
    TransformerBlock block(128, 8, 512, DType::Float32, 0.1);
    
    Tensor input({2, 10, 128}, DType::Float32, Device::CPU);
    auto* input_data = static_cast<float*>(input.data());
    for (size_t i = 0; i < input.numel(); ++i) {
        input_data[i] = 0.1f * static_cast<float>(i);
    }
    
    Tensor output = block.forward(input);
    
    // Output should have same shape as input
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 10);
    EXPECT_EQ(output.shape()[2], 128);
}

TEST(TransformerBlockTest, ForwardSmallInput) {
    TransformerBlock block(4, 2, 8, DType::Float32, 0.1);
    
    Tensor input({1, 2, 4}, DType::Float32, Device::CPU);
    auto* input_data = static_cast<float*>(input.data());
    input_data[0] = 1.0f; input_data[1] = 2.0f; input_data[2] = 3.0f; input_data[3] = 4.0f;
    input_data[4] = 5.0f; input_data[5] = 6.0f; input_data[6] = 7.0f; input_data[7] = 8.0f;
    
    Tensor output = block.forward(input);
    
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 2);
    EXPECT_EQ(output.shape()[2], 4);
}

TEST(TransformerBlockTest, ForwardLargeInput) {
    TransformerBlock block(256, 16, 1024, DType::Float32, 0.1);
    
    Tensor input({8, 50, 256}, DType::Float32, Device::CPU);
    auto* input_data = static_cast<float*>(input.data());
    for (size_t i = 0; i < input.numel(); ++i) {
        input_data[i] = 0.01f;
    }
    
    Tensor output = block.forward(input);
    
    EXPECT_EQ(output.shape()[0], 8);
    EXPECT_EQ(output.shape()[1], 50);
    EXPECT_EQ(output.shape()[2], 256);
}

TEST(TransformerBlockTest, ForwardSingleHead) {
    TransformerBlock block(64, 1, 256, DType::Float32, 0.1);
    
    Tensor input({2, 10, 64}, DType::Float32, Device::CPU);
    auto* input_data = static_cast<float*>(input.data());
    for (size_t i = 0; i < input.numel(); ++i) {
        input_data[i] = 0.1f;
    }
    
    Tensor output = block.forward(input);
    
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 10);
    EXPECT_EQ(output.shape()[2], 64);
}

TEST(TransformerBlockTest, ForwardMultipleBatches) {
    TransformerBlock block(64, 4, 256, DType::Float32, 0.1);
    
    Tensor input({16, 20, 64}, DType::Float32, Device::CPU);
    auto* input_data = static_cast<float*>(input.data());
    for (size_t i = 0; i < input.numel(); ++i) {
        input_data[i] = 0.05f;
    }
    
    Tensor output = block.forward(input);
    
    EXPECT_EQ(output.shape()[0], 16);
    EXPECT_EQ(output.shape()[1], 20);
    EXPECT_EQ(output.shape()[2], 64);
}

TEST(TransformerBlockTest, Parameters) {
    TransformerBlock block(128, 8, 512, DType::Float32, 0.1);
    
    auto params = block.parameters();
    
    // Should have parameters from all sub-modules
    EXPECT_GT(params.size(), 0);
}

TEST(TransformerBlockTest, ZeroGrad) {
    TransformerBlock block(128, 8, 512, DType::Float32, 0.1);
    
    EXPECT_NO_THROW(block.zero_grad());
}

// GPT Tests
TEST(GPTTest, BasicConstruction) {
    GPT gpt(1000, 128, 64, 4, 2, 256, DType::Float32, 0.1);
    
    EXPECT_NO_THROW();
}

TEST(GPTTest, ForwardBasic) {
    GPT gpt(1000, 128, 64, 4, 2, 256, DType::Float32, 0.1);
    
    Tensor input({2, 10}, DType::Int64, Device::CPU);
    auto* input_data = static_cast<int64_t*>(input.data());
    for (size_t i = 0; i < input.numel(); ++i) {
        input_data[i] = static_cast<int64_t>(i % 1000);
    }
    
    Tensor output = gpt.forward(input);
    
    // Output should be [batch_size, seq_len, vocab_size]
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 10);
    EXPECT_EQ(output.shape()[2], 1000);
}

TEST(GPTTest, ForwardSmallInput) {
    GPT gpt(100, 32, 32, 2, 1, 128, DType::Float32, 0.1);
    
    Tensor input({1, 5}, DType::Int64, Device::CPU);
    auto* input_data = static_cast<int64_t*>(input.data());
    input_data[0] = 1; input_data[1] = 2; input_data[2] = 3; input_data[3] = 4; input_data[4] = 5;
    
    Tensor output = gpt.forward(input);
    
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.shape()[2], 100);
}

TEST(GPTTest, ForwardLargeInput) {
    GPT gpt(50000, 1024, 512, 8, 6, 2048, DType::Float32, 0.1);
    
    Tensor input({4, 128}, DType::Int64, Device::CPU);
    auto* input_data = static_cast<int64_t*>(input.data());
    for (size_t i = 0; i < input.numel(); ++i) {
        input_data[i] = static_cast<int64_t>(i % 50000);
    }
    
    Tensor output = gpt.forward(input);
    
    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 128);
    EXPECT_EQ(output.shape()[2], 50000);
}

TEST(GPTTest, ForwardSingleLayer) {
    GPT gpt(1000, 64, 32, 2, 1, 128, DType::Float32, 0.1);
    
    Tensor input({2, 8}, DType::Int64, Device::CPU);
    auto* input_data = static_cast<int64_t*>(input.data());
    for (size_t i = 0; i < input.numel(); ++i) {
        input_data[i] = static_cast<int64_t>(i % 1000);
    }
    
    Tensor output = gpt.forward(input);
    
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 8);
    EXPECT_EQ(output.shape()[2], 1000);
}

TEST(GPTTest, ForwardMultipleLayers) {
    GPT gpt(1000, 128, 64, 4, 6, 256, DType::Float32, 0.1);
    
    Tensor input({2, 10}, DType::Int64, Device::CPU);
    auto* input_data = static_cast<int64_t*>(input.data());
    for (size_t i = 0; i < input.numel(); ++i) {
        input_data[i] = static_cast<int64_t>(i % 1000);
    }
    
    Tensor output = gpt.forward(input);
    
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 10);
    EXPECT_EQ(output.shape()[2], 1000);
}

TEST(GPTTest, ForwardDifferentSequenceLengths) {
    GPT gpt(1000, 128, 64, 4, 2, 256, DType::Float32, 0.1);
    
    Tensor input({2, 20}, DType::Int64, Device::CPU);
    auto* input_data = static_cast<int64_t*>(input.data());
    for (size_t i = 0; i < input.numel(); ++i) {
        input_data[i] = static_cast<int64_t>(i % 1000);
    }
    
    Tensor output = gpt.forward(input);
    
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 20);
    EXPECT_EQ(output.shape()[2], 1000);
}

TEST(GPTTest, ForwardMaxSequenceLength) {
    GPT gpt(1000, 32, 32, 2, 1, 128, DType::Float32, 0.1);
    
    Tensor input({1, 32}, DType::Int64, Device::CPU);
    auto* input_data = static_cast<int64_t*>(input.data());
    for (size_t i = 0; i < input.numel(); ++i) {
        input_data[i] = static_cast<int64_t>(i % 1000);
    }
    
    Tensor output = gpt.forward(input);
    
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 32);
    EXPECT_EQ(output.shape()[2], 1000);
}

TEST(GPTTest, Parameters) {
    GPT gpt(1000, 128, 64, 4, 2, 256, DType::Float32, 0.1);
    
    auto params = gpt.parameters();
    
    // Should have many parameters from embeddings, transformer blocks, and lm head
    EXPECT_GT(params.size(), 0);
}

TEST(GPTTest, ZeroGrad) {
    GPT gpt(1000, 128, 64, 4, 2, 256, DType::Float32, 0.1);
    
    EXPECT_NO_THROW(gpt.zero_grad());
}

TEST(GPTTest, ZeroGradAfterForward) {
    GPT gpt(1000, 128, 64, 4, 2, 256, DType::Float32, 0.1);
    
    Tensor input({2, 10}, DType::Int64, Device::CPU);
    auto* input_data = static_cast<int64_t*>(input.data());
    for (size_t i = 0; i < input.numel(); ++i) {
        input_data[i] = static_cast<int64_t>(i % 1000);
    }
    
    Tensor output = gpt.forward(input);
    
    EXPECT_NO_THROW(gpt.zero_grad());
}
