#include <gtest/gtest.h>
#include "../nn/scaledot.h"
#include "../nn/multi_head_attention.h"
#include "../tensor/tensor.h"
#include "../core/dtype.h"

// ScaledDotProductAttention Tests

TEST(ScaledDotTest, BasicConstruction) {
    ScaledDot attention;
    EXPECT_NO_THROW();
}

TEST(ScaledDotTest, ForwardBasic) {
    ScaledDot attention;
    
    // Create Q, K, V tensors [batch=2, seq_len=3, d_model=4]
    Tensor Q({2, 3, 4}, DType::Float32, Device::CPU);
    Tensor K({2, 3, 4}, DType::Float32, Device::CPU);
    Tensor V({2, 3, 4}, DType::Float32, Device::CPU);
    
    // Initialize with some values
    float* q_data = static_cast<float*>(Q.data());
    float* k_data = static_cast<float*>(K.data());
    float* v_data = static_cast<float*>(V.data());
    
    for (size_t i = 0; i < Q.numel(); ++i) {
        q_data[i] = static_cast<float>(i) * 0.1f;
        k_data[i] = static_cast<float>(i) * 0.1f + 0.5f;
        v_data[i] = static_cast<float>(i) * 0.1f + 1.0f;
    }
    
    Tensor output = attention.forward(Q, K, V);
    
    // Output shape should be [batch=2, seq_len=3, d_model=4]
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 4);
}

TEST(ScaledDotTest, ForwardDifferentDimensions) {
    ScaledDot attention;
    
    // Create Q, K, V with different dimensions [batch=1, seq_len=5, d_model=8]
    Tensor Q({1, 5, 8}, DType::Float32, Device::CPU);
    Tensor K({1, 5, 8}, DType::Float32, Device::CPU);
    Tensor V({1, 5, 8}, DType::Float32, Device::CPU);
    
    float* q_data = static_cast<float*>(Q.data());
    float* k_data = static_cast<float*>(K.data());
    float* v_data = static_cast<float*>(V.data());
    
    for (size_t i = 0; i < Q.numel(); ++i) {
        q_data[i] = 0.1f;
        k_data[i] = 0.2f;
        v_data[i] = 0.3f;
    }
    
    Tensor output = attention.forward(Q, K, V);
    
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.shape()[2], 8);
}

TEST(ScaledDotTest, Forward2DInput) {
    ScaledDot attention;
    
    // Create 2D tensors [seq_len=3, d_model=4]
    Tensor Q({3, 4}, DType::Float32, Device::CPU);
    Tensor K({3, 4}, DType::Float32, Device::CPU);
    Tensor V({3, 4}, DType::Float32, Device::CPU);
    
    float* q_data = static_cast<float*>(Q.data());
    for (size_t i = 0; i < Q.numel(); ++i) {
        q_data[i] = static_cast<float>(i) * 0.1f;
    }
    
    float* k_data = static_cast<float*>(K.data());
    for (size_t i = 0; i < K.numel(); ++i) {
        k_data[i] = static_cast<float>(i) * 0.1f + 0.5f;
    }
    
    float* v_data = static_cast<float*>(V.data());
    for (size_t i = 0; i < V.numel(); ++i) {
        v_data[i] = static_cast<float>(i) * 0.1f + 1.0f;
    }
    
    Tensor output = attention.forward(Q, K, V);
    
    // Output shape should be [seq_len=3, d_model=4]
    EXPECT_EQ(output.shape()[0], 3);
    EXPECT_EQ(output.shape()[1], 4);
}

TEST(ScaledDotTest, ParametersEmpty) {
    ScaledDot attention;
    auto params = attention.parameters();
    EXPECT_EQ(params.size(), 0);
}

TEST(ScaledDotTest, ZeroGradNoThrow) {
    ScaledDot attention;
    EXPECT_NO_THROW(attention.zero_grad());
}

// MultiHeadAttention Tests

TEST(MultiHeadAttentionTest, BasicConstruction) {
    MultiHeadAttention mha(512, 8, DType::Float32);
    EXPECT_NO_THROW();
}

TEST(MultiHeadAttentionTest, ConstructionDivisible) {
    // embedDim should be divisible by numHeads
    EXPECT_NO_THROW(MultiHeadAttention mha(512, 8, DType::Float32));
    EXPECT_NO_THROW(MultiHeadAttention mha(256, 4, DType::Float32));
    EXPECT_NO_THROW(MultiHeadAttention mha(128, 2, DType::Float32));
}

TEST(MultiHeadAttentionTest, ConstructionNonDivisible) {
    // embedDim not divisible by numHeads should throw
    EXPECT_THROW(MultiHeadAttention mha(512, 7, DType::Float32), std::runtime_error);
    EXPECT_THROW(MultiHeadAttention mha(100, 3, DType::Float32), std::runtime_error);
}

TEST(MultiHeadAttentionTest, ForwardBasic) {
    MultiHeadAttention mha(512, 8, DType::Float32);
    
    // Create input tensor [batch=2, seq_len=10, embed_dim=512]
    Tensor input({2, 10, 512}, DType::Float32, Device::CPU);
    
    // Initialize with some values
    float* data = static_cast<float*>(input.data());
    for (size_t i = 0; i < input.numel(); ++i) {
        data[i] = static_cast<float>(i) * 0.01f;
    }
    
    Tensor output = mha.forward(input);
    
    // Output shape should be [batch=2, seq_len=10, embed_dim=512]
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 10);
    EXPECT_EQ(output.shape()[2], 512);
}

TEST(MultiHeadAttentionTest, ForwardSmallerModel) {
    MultiHeadAttention mha(64, 4, DType::Float32);
    
    // Create input tensor [batch=1, seq_len=5, embed_dim=64]
    Tensor input({1, 5, 64}, DType::Float32, Device::CPU);
    
    float* data = static_cast<float*>(input.data());
    for (size_t i = 0; i < input.numel(); ++i) {
        data[i] = 0.1f;
    }
    
    Tensor output = mha.forward(input);
    
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.shape()[2], 64);
}

TEST(MultiHeadAttentionTest, ForwardSingleHead) {
    MultiHeadAttention mha(32, 1, DType::Float32);
    
    // Create input tensor [batch=2, seq_len=8, embed_dim=32]
    Tensor input({2, 8, 32}, DType::Float32, Device::CPU);
    
    float* data = static_cast<float*>(input.data());
    for (size_t i = 0; i < input.numel(); ++i) {
        data[i] = static_cast<float>(i) * 0.05f;
    }
    
    Tensor output = mha.forward(input);
    
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 8);
    EXPECT_EQ(output.shape()[2], 32);
}

TEST(MultiHeadAttentionTest, ForwardMultipleBatches) {
    MultiHeadAttention mha(128, 4, DType::Float32);
    
    // Create input tensor [batch=4, seq_len=6, embed_dim=128]
    Tensor input({4, 6, 128}, DType::Float32, Device::CPU);
    
    float* data = static_cast<float*>(input.data());
    for (size_t i = 0; i < input.numel(); ++i) {
        data[i] = 0.2f;
    }
    
    Tensor output = mha.forward(input);
    
    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 6);
    EXPECT_EQ(output.shape()[2], 128);
}

TEST(MultiHeadAttentionTest, ParametersCount) {
    MultiHeadAttention mha(512, 8, DType::Float32);
    
    auto params = mha.parameters();
    
    // Should have 4 linear layers * 2 parameters (weight + bias) = 8 parameters
    EXPECT_EQ(params.size(), 8);
}

TEST(MultiHeadAttentionTest, ParametersShapes) {
    MultiHeadAttention mha(256, 4, DType::Float32);
    
    auto params = mha.parameters();
    
    // Q, K, V projections: each has weight [256, 256] and bias [256]
    // Output projection: weight [256, 256] and bias [256]
    
    // Check that all parameters have correct shapes
    EXPECT_EQ(params[0].shape()[0], 256);  // q_proj weight
    EXPECT_EQ(params[0].shape()[1], 256);
    EXPECT_EQ(params[1].shape()[0], 256);  // q_proj bias
    
    EXPECT_EQ(params[2].shape()[0], 256);  // k_proj weight
    EXPECT_EQ(params[2].shape()[1], 256);
    EXPECT_EQ(params[3].shape()[0], 256);  // k_proj bias
    
    EXPECT_EQ(params[4].shape()[0], 256);  // v_proj weight
    EXPECT_EQ(params[4].shape()[1], 256);
    EXPECT_EQ(params[5].shape()[0], 256);  // v_proj bias
    
    EXPECT_EQ(params[6].shape()[0], 256);  // out_proj weight
    EXPECT_EQ(params[6].shape()[1], 256);
    EXPECT_EQ(params[7].shape()[0], 256);  // out_proj bias
}

TEST(MultiHeadAttentionTest, ZeroGradNoThrow) {
    MultiHeadAttention mha(128, 4, DType::Float32);
    EXPECT_NO_THROW(mha.zero_grad());
}

TEST(MultiHeadAttentionTest, ZeroGradAfterForward) {
    MultiHeadAttention mha(64, 2, DType::Float32);
    
    Tensor input({1, 5, 64}, DType::Float32, Device::CPU);
    float* data = static_cast<float*>(input.data());
    for (size_t i = 0; i < input.numel(); ++i) {
        data[i] = 0.1f;
    }
    
    Tensor output = mha.forward(input);
    
    // zero_grad should work after forward pass
    EXPECT_NO_THROW(mha.zero_grad());
}

TEST(MultiHeadAttentionTest, ForwardDifferentSequenceLengths) {
    MultiHeadAttention mha(128, 4, DType::Float32);
    
    // Test with different sequence lengths
    std::vector<int64_t> seq_lengths = {5, 10, 20, 50};
    
    for (int64_t seq_len : seq_lengths) {
        Tensor input({1, seq_len, 128}, DType::Float32, Device::CPU);
        float* data = static_cast<float*>(input.data());
        for (size_t i = 0; i < input.numel(); ++i) {
            data[i] = 0.1f;
        }
        
        Tensor output = mha.forward(input);
        
        EXPECT_EQ(output.shape()[0], 1);
        EXPECT_EQ(output.shape()[1], seq_len);
        EXPECT_EQ(output.shape()[2], 128);
    }
}

TEST(MultiHeadAttentionTest, ForwardDifferentEmbedDimensions) {
    std::vector<int64_t> embed_dims = {32, 64, 128, 256, 512};
    
    for (int64_t embed_dim : embed_dims) {
        int64_t num_heads = embed_dim / 32;  // Ensure divisibility
        MultiHeadAttention mha(embed_dim, num_heads, DType::Float32);
        
        Tensor input({1, 10, embed_dim}, DType::Float32, Device::CPU);
        float* data = static_cast<float*>(input.data());
        for (size_t i = 0; i < input.numel(); ++i) {
            data[i] = 0.1f;
        }
        
        Tensor output = mha.forward(input);
        
        EXPECT_EQ(output.shape()[0], 1);
        EXPECT_EQ(output.shape()[1], 10);
        EXPECT_EQ(output.shape()[2], embed_dim);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
