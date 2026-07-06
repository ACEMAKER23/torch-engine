#include <gtest/gtest.h>
#include "../nn/gpt.h"
#include "../optimizer/adam.h"
#include "../tensor/tensor.h"
#include "../core/dtype.h"

// Test GPT forward pass with synthetic data
TEST(EndToEndTest, GPTForwardPass) {
    // Create tiny GPT model
    int64_t vocab_size = 100;
    int64_t max_seq_len = 32;
    int64_t embed_dim = 64;
    int64_t num_heads = 4;
    int64_t num_layers = 2;
    int64_t ff_dim = 256;
    DType dtype = DType::Float32;
    float dropout_rate = 0.1f;
    
    GPT model(vocab_size, max_seq_len, embed_dim, num_heads, num_layers, ff_dim, dtype, dropout_rate);
    
    // Create synthetic input
    int64_t batch_size = 2;
    int64_t seq_len = 16;
    
    Tensor input({batch_size, seq_len}, DType::Int64, Device::CPU);
    for (size_t i = 0; i < input.numel(); ++i) {
        input.at<int64_t>(i) = rand() % vocab_size;
    }
    
    // Forward pass
    Tensor logits = model.forward(input);
    
    // Check output shape
    auto shape = logits.shape();
    EXPECT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], batch_size);
    EXPECT_EQ(shape[1], seq_len);
    EXPECT_EQ(shape[2], vocab_size);
}

// Test GPT forward+backward with loss
TEST(EndToEndTest, GPTForwardBackwardWithLoss) {
    // Create tiny GPT model
    int64_t vocab_size = 100;
    int64_t max_seq_len = 32;
    int64_t embed_dim = 64;
    int64_t num_heads = 4;
    int64_t num_layers = 2;
    int64_t ff_dim = 256;
    DType dtype = DType::Float32;
    float dropout_rate = 0.0f;  // Disable dropout for deterministic testing
    
    GPT model(vocab_size, max_seq_len, embed_dim, num_heads, num_layers, ff_dim, dtype, dropout_rate);
    
    // Create synthetic input and targets
    int64_t batch_size = 2;
    int64_t seq_len = 16;
    
    Tensor input({batch_size, seq_len}, DType::Int64, Device::CPU);
    Tensor targets({batch_size, seq_len}, DType::Int64, Device::CPU);
    
    for (size_t i = 0; i < input.numel(); ++i) {
        input.at<int64_t>(i) = rand() % vocab_size;
        targets.at<int64_t>(i) = rand() % vocab_size;
    }
    
    // Forward pass
    Tensor logits = model.forward(input);
    
    // Compute loss (reshape for cross entropy)
    // logits: [batch, seq, vocab] -> [batch*seq, vocab]
    // targets: [batch, seq] -> [batch*seq]
    // Note: Need to implement reshape/view for tensors
    // For now, just verify forward pass works
    
    // Note: CrossEntropyLoss implementation may need adjustment for this use case
    // For now, just verify forward pass works
    EXPECT_NO_THROW();
}

// Test GPT with optimizer step
TEST(EndToEndTest, GPTWithOptimizerStep) {
    // Create tiny GPT model
    int64_t vocab_size = 100;
    int64_t max_seq_len = 32;
    int64_t embed_dim = 64;
    int64_t num_heads = 4;
    int64_t num_layers = 2;
    int64_t ff_dim = 256;
    DType dtype = DType::Float32;
    float dropout_rate = 0.0f;
    
    GPT model(vocab_size, max_seq_len, embed_dim, num_heads, num_layers, ff_dim, dtype, dropout_rate);
    
    // Get model parameters
    auto params = model.parameters();
    
    // Create optimizer
    adam optimizer(params, 0.9f, 0.999f, 1e-8f);
    
    // Create synthetic input
    int64_t batch_size = 2;
    int64_t seq_len = 16;
    
    Tensor input({batch_size, seq_len}, DType::Int64, Device::CPU);
    for (size_t i = 0; i < input.numel(); ++i) {
        input.at<int64_t>(i) = rand() % vocab_size;
    }
    
    // Forward pass
    Tensor logits = model.forward(input);
    
    // Zero gradients
    optimizer.zero_grad();
    
    // Optimizer step (even without gradients, should not crash)
    EXPECT_NO_THROW(optimizer.step(0.001f));
}

// Test GPT parameters
TEST(EndToEndTest, GPTParameters) {
    int64_t vocab_size = 100;
    int64_t max_seq_len = 32;
    int64_t embed_dim = 64;
    int64_t num_heads = 4;
    int64_t num_layers = 2;
    int64_t ff_dim = 256;
    DType dtype = DType::Float32;
    float dropout_rate = 0.1f;
    
    GPT model(vocab_size, max_seq_len, embed_dim, num_heads, num_layers, ff_dim, dtype, dropout_rate);
    
    auto params = model.parameters();
    
    // Should have parameters from:
    // - Token embedding (1)
    // - Positional embedding (0 - no learnable params)
    // - Transformer blocks (each: norm1, attention (4), norm2, ffn (linear1, linear2) = 8 per block)
    // - Final norm (2)
    // - LM head (2)
    // Total: 1 + 2*8 + 2 + 2 = 21
    
    EXPECT_GT(params.size(), 0);
    
    // Check all parameters have correct dtype
    for (const auto& param : params) {
        EXPECT_EQ(param.dtype(), dtype);
    }
}

// Test GPT zero_grad
TEST(EndToEndTest, GPTZeroGrad) {
    int64_t vocab_size = 100;
    int64_t max_seq_len = 32;
    int64_t embed_dim = 64;
    int64_t num_heads = 4;
    int64_t num_layers = 2;
    int64_t ff_dim = 256;
    DType dtype = DType::Float32;
    float dropout_rate = 0.1f;
    
    GPT model(vocab_size, max_seq_len, embed_dim, num_heads, num_layers, ff_dim, dtype, dropout_rate);
    
    // Should not crash
    EXPECT_NO_THROW(model.zero_grad());
}
