#include <gtest/gtest.h>
#include "../loss/CrossEntropyLoss.h"
#include "../loss/MSELoss.h"
#include "../loss/BCELoss.h"
#include "../loss/L1Loss.h"
#include "../tensor/tensor.h"
#include <cmath>

class LossTest : public ::testing::Test {
protected:
    void SetUp() override {}
    float epsilon = 1e-5f;
};

// Helper function for finite difference gradient checking
float finite_difference_gradient(const Tensor& predictions, const Tensor& targets, 
                                  size_t index, float h = 1e-5f) {
    // Create perturbed tensors
    Tensor predictions_plus = predictions.clone();
    Tensor predictions_minus = predictions.clone();
    
    predictions_plus.at<float>(index) += h;
    predictions_minus.at<float>(index) -= h;
    
    // Compute loss at perturbed points (using simple loss computation for testing)
    // This is a simplified version - in practice you'd call the loss function
    float loss_plus = 0.0f;
    float loss_minus = 0.0f;
    
    for (size_t i = 0; i < predictions.numel(); ++i) {
        loss_plus += std::pow(predictions_plus.at<float>(i) - targets.at<float>(i), 2.0f);
        loss_minus += std::pow(predictions_minus.at<float>(i) - targets.at<float>(i), 2.0f);
    }
    
    loss_plus /= predictions.numel();
    loss_minus /= predictions.numel();
    
    return (loss_plus - loss_minus) / (2.0f * h);
}

// ============ CrossEntropyLoss Tests ============

TEST_F(LossTest, CrossEntropyForwardWithLogits_Basic) {
    // Test basic cross-entropy with logits
    Tensor logits({3}, DType::Float32, Device::CPU);
    logits.at<float>(0) = 1.0f;
    logits.at<float>(1) = 2.0f;
    logits.at<float>(2) = 3.0f;
    
    Tensor targets({1}, DType::Int64, Device::CPU);
    targets.at<int64_t>(0) = 2; // Target is class 2
    
    crossEntropyLoss loss_fn;
    Tensor loss = loss_fn.forward(logits, targets);
    
    // Compute expected: softmax = [0.09, 0.24, 0.67]
    // loss = -log(0.67) ≈ 0.40
    float expected_loss = -std::log(std::exp(3.0f) / (std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f)));
    
    EXPECT_NEAR(loss.at<float>(0), expected_loss, epsilon);
}

TEST_F(LossTest, CrossEntropyForwardWithProbs_Basic) {
    // Test cross-entropy with probabilities
    Tensor probs({3}, DType::Float32, Device::CPU);
    probs.at<float>(0) = 0.1f;
    probs.at<float>(1) = 0.3f;
    probs.at<float>(2) = 0.6f;
    
    Tensor targets({1}, DType::Int64, Device::CPU);
    targets.at<int64_t>(0) = 2; // Target is class 2
    
    crossEntropyLoss loss_fn;
    Tensor loss = loss_fn.forward_with_probs(probs, targets);
    
    // Expected: -log(0.6) ≈ 0.51
    float expected_loss = -std::log(0.6f);
    
    EXPECT_NEAR(loss.at<float>(0), expected_loss, epsilon);
}

TEST_F(LossTest, CrossEntropyWithLogits_PerfectPrediction) {
    // Test when prediction is perfect (logits very high for target)
    Tensor logits({3}, DType::Float32, Device::CPU);
    logits.at<float>(0) = -100.0f;
    logits.at<float>(1) = -100.0f;
    logits.at<float>(2) = 100.0f;
    
    Tensor targets({1}, DType::Int64, Device::CPU);
    targets.at<int64_t>(0) = 2;
    
    crossEntropyLoss loss_fn;
    Tensor loss = loss_fn.forward(logits, targets);
    
    // Loss should be very close to 0 for perfect prediction
    EXPECT_NEAR(loss.at<float>(0), 0.0f, epsilon);
}

TEST_F(LossTest, CrossEntropyWithProbs_PerfectPrediction) {
    // Test when probability is 1.0 for target
    Tensor probs({3}, DType::Float32, Device::CPU);
    probs.at<float>(0) = 0.0f;
    probs.at<float>(1) = 0.0f;
    probs.at<float>(2) = 1.0f;
    
    Tensor targets({1}, DType::Int64, Device::CPU);
    targets.at<int64_t>(0) = 2;
    
    crossEntropyLoss loss_fn;
    Tensor loss = loss_fn.forward_with_probs(probs, targets);
    
    // Loss should be 0 for perfect prediction
    EXPECT_NEAR(loss.at<float>(0), 0.0f, epsilon);
}

TEST_F(LossTest, CrossEntropyWithProbs_ZeroProbability) {
    // Test edge case with zero probability for target
    Tensor probs({3}, DType::Float32, Device::CPU);
    probs.at<float>(0) = 0.5f;
    probs.at<float>(1) = 0.5f;
    probs.at<float>(2) = 0.0f;
    
    Tensor targets({1}, DType::Int64, Device::CPU);
    targets.at<int64_t>(0) = 2;
    
    crossEntropyLoss loss_fn;
    Tensor loss = loss_fn.forward_with_probs(probs, targets);
    
    // Loss should be very large (infinity in limit)
    EXPECT_GT(loss.at<float>(0), 10.0f);
}

TEST_F(LossTest, CrossEntropyWithLogits_LargeInputs) {
    // Test with large vocabulary size
    const size_t vocab_size = 10000;
    Tensor logits({vocab_size}, DType::Float32, Device::CPU);
    
    for (size_t i = 0; i < vocab_size; ++i) {
        logits.at<float>(i) = static_cast<float>(i) * 0.001f;
    }
    
    Tensor targets({1}, DType::Int64, Device::CPU);
    targets.at<int64_t>(0) = 5000;
    
    crossEntropyLoss loss_fn;
    Tensor loss = loss_fn.forward(logits, targets);
    
    // Should compute without overflow
    EXPECT_FALSE(std::isnan(loss.at<float>(0)));
    EXPECT_FALSE(std::isinf(loss.at<float>(0)));
}

TEST_F(LossTest, CrossEntropyWithLogits_SmallInputs) {
    // Test with very small logits
    Tensor logits({3}, DType::Float32, Device::CPU);
    logits.at<float>(0) = -1000.0f;
    logits.at<float>(1) = -1001.0f;
    logits.at<float>(2) = -1002.0f;
    
    Tensor targets({1}, DType::Int64, Device::CPU);
    targets.at<int64_t>(0) = 0;
    
    crossEntropyLoss loss_fn;
    Tensor loss = loss_fn.forward(logits, targets);
    
    // Should compute without underflow
    EXPECT_FALSE(std::isnan(loss.at<float>(0)));
    EXPECT_FALSE(std::isinf(loss.at<float>(0)));
}

// ============ MSELoss Tests ============

TEST_F(LossTest, MSEForward_Basic) {
    Tensor predictions({3}, DType::Float32, Device::CPU);
    predictions.at<float>(0) = 1.0f;
    predictions.at<float>(1) = 2.0f;
    predictions.at<float>(2) = 3.0f;
    
    Tensor targets({3}, DType::Float32, Device::CPU);
    targets.at<float>(0) = 1.5f;
    targets.at<float>(1) = 2.5f;
    targets.at<float>(2) = 3.5f;
    
    MSELoss loss_fn;
    Tensor loss = loss_fn.forward(predictions, targets);
    
    // Expected: ((1-1.5)^2 + (2-2.5)^2 + (3-3.5)^2) / 3 = 0.25
    float expected_loss = (0.25f + 0.25f + 0.25f) / 3.0f;
    
    EXPECT_NEAR(loss.at<float>(0), expected_loss, epsilon);
}

TEST_F(LossTest, MSEForward_PerfectPrediction) {
    Tensor predictions({3}, DType::Float32, Device::CPU);
    predictions.at<float>(0) = 1.0f;
    predictions.at<float>(1) = 2.0f;
    predictions.at<float>(2) = 3.0f;
    
    Tensor targets({3}, DType::Float32, Device::CPU);
    targets.at<float>(0) = 1.0f;
    targets.at<float>(1) = 2.0f;
    targets.at<float>(2) = 3.0f;
    
    MSELoss loss_fn;
    Tensor loss = loss_fn.forward(predictions, targets);
    
    EXPECT_NEAR(loss.at<float>(0), 0.0f, epsilon);
}

TEST_F(LossTest, MSEForward_LargeError) {
    Tensor predictions({2}, DType::Float32, Device::CPU);
    predictions.at<float>(0) = -1000.0f;
    predictions.at<float>(1) = 1000.0f;
    
    Tensor targets({2}, DType::Float32, Device::CPU);
    targets.at<float>(0) = 0.0f;
    targets.at<float>(1) = 0.0f;
    
    MSELoss loss_fn;
    Tensor loss = loss_fn.forward(predictions, targets);
    
    // Expected: (1000000 + 1000000) / 2 = 1000000
    float expected_loss = 1000000.0f;
    
    EXPECT_NEAR(loss.at<float>(0), expected_loss, epsilon);
}

TEST_F(LossTest, MSEForward_LargeInputs) {
    const size_t n = 10000;
    Tensor predictions({n}, DType::Float32, Device::CPU);
    Tensor targets({n}, DType::Float32, Device::CPU);
    
    for (size_t i = 0; i < n; ++i) {
        predictions.at<float>(i) = static_cast<float>(i);
        targets.at<float>(i) = static_cast<float>(i) + 0.1f;
    }
    
    MSELoss loss_fn;
    Tensor loss = loss_fn.forward(predictions, targets);
    
    // Should compute without issues
    EXPECT_FALSE(std::isnan(loss.at<float>(0)));
    EXPECT_FALSE(std::isinf(loss.at<float>(0)));
    EXPECT_NEAR(loss.at<float>(0), 0.01f, epsilon);
}

// ============ BCELoss Tests ============

TEST_F(LossTest, BCEForward_Basic) {
    Tensor predictions({3}, DType::Float32, Device::CPU);
    predictions.at<float>(0) = 0.9f;
    predictions.at<float>(1) = 0.2f;
    predictions.at<float>(2) = 0.7f;
    
    Tensor targets({3}, DType::Float32, Device::CPU);
    targets.at<float>(0) = 1.0f;
    targets.at<float>(1) = 0.0f;
    targets.at<float>(2) = 1.0f;
    
    BCELoss loss_fn;
    Tensor loss = loss_fn.forward(predictions, targets);
    
    // Expected: -[log(0.9) + log(0.8) + log(0.7)] / 3 ≈ 0.21
    float expected_loss = -(std::log(0.9f) + std::log(0.8f) + std::log(0.7f)) / 3.0f;
    
    EXPECT_NEAR(loss.at<float>(0), expected_loss, epsilon);
}

TEST_F(LossTest, BCEForward_PerfectPrediction) {
    Tensor predictions({3}, DType::Float32, Device::CPU);
    predictions.at<float>(0) = 1.0f;
    predictions.at<float>(1) = 0.0f;
    predictions.at<float>(2) = 1.0f;
    
    Tensor targets({3}, DType::Float32, Device::CPU);
    targets.at<float>(0) = 1.0f;
    targets.at<float>(1) = 0.0f;
    targets.at<float>(2) = 1.0f;
    
    BCELoss loss_fn;
    Tensor loss = loss_fn.forward(predictions, targets);
    
    // Should be close to 0 (clamping might affect slightly)
    EXPECT_NEAR(loss.at<float>(0), 0.0f, 1e-3f);
}

TEST_F(LossTest, BCEForward_EdgeCaseZero) {
    Tensor predictions({2}, DType::Float32, Device::CPU);
    predictions.at<float>(0) = 0.0f;  // Will be clamped
    predictions.at<float>(1) = 0.5f;
    
    Tensor targets({2}, DType::Float32, Device::CPU);
    targets.at<float>(0) = 1.0f;
    targets.at<float>(1) = 0.0f;
    
    BCELoss loss_fn;
    Tensor loss = loss_fn.forward(predictions, targets);
    
    // Should not crash due to log(0) clamping
    EXPECT_FALSE(std::isnan(loss.at<float>(0)));
    EXPECT_FALSE(std::isinf(loss.at<float>(0)));
}

TEST_F(LossTest, BCEForward_EdgeCaseOne) {
    Tensor predictions({2}, DType::Float32, Device::CPU);
    predictions.at<float>(0) = 1.0f;  // Will be clamped
    predictions.at<float>(1) = 0.5f;
    
    Tensor targets({2}, DType::Float32, Device::CPU);
    targets.at<float>(0) = 0.0f;
    targets.at<float>(1) = 1.0f;
    
    BCELoss loss_fn;
    Tensor loss = loss_fn.forward(predictions, targets);
    
    // Should not crash due to log(0) clamping
    EXPECT_FALSE(std::isnan(loss.at<float>(0)));
    EXPECT_FALSE(std::isinf(loss.at<float>(0)));
}

TEST_F(LossTest, BCEForward_LargeInputs) {
    const size_t n = 10000;
    Tensor predictions({n}, DType::Float32, Device::CPU);
    Tensor targets({n}, DType::Float32, Device::CPU);
    
    for (size_t i = 0; i < n; ++i) {
        predictions.at<float>(i) = 0.5f;
        targets.at<float>(i) = (i % 2 == 0) ? 1.0f : 0.0f;
    }
    
    BCELoss loss_fn;
    Tensor loss = loss_fn.forward(predictions, targets);
    
    // Expected: -log(0.5) ≈ 0.693
    // Use relaxed epsilon for large input due to floating point accumulation
    EXPECT_NEAR(loss.at<float>(0), std::log(2.0f), 1e-4f);
}

// ============ L1Loss Tests ============

TEST_F(LossTest, L1Forward_Basic) {
    Tensor predictions({3}, DType::Float32, Device::CPU);
    predictions.at<float>(0) = 1.0f;
    predictions.at<float>(1) = 2.0f;
    predictions.at<float>(2) = 3.0f;
    
    Tensor targets({3}, DType::Float32, Device::CPU);
    targets.at<float>(0) = 1.5f;
    targets.at<float>(1) = 2.5f;
    targets.at<float>(2) = 3.5f;
    
    L1Loss loss_fn;
    Tensor loss = loss_fn.forward(predictions, targets);
    
    // Expected: (|1-1.5| + |2-2.5| + |3-3.5|) / 3 = 0.5
    float expected_loss = (0.5f + 0.5f + 0.5f) / 3.0f;
    
    EXPECT_NEAR(loss.at<float>(0), expected_loss, epsilon);
}

TEST_F(LossTest, L1Forward_PerfectPrediction) {
    Tensor predictions({3}, DType::Float32, Device::CPU);
    predictions.at<float>(0) = 1.0f;
    predictions.at<float>(1) = 2.0f;
    predictions.at<float>(2) = 3.0f;
    
    Tensor targets({3}, DType::Float32, Device::CPU);
    targets.at<float>(0) = 1.0f;
    targets.at<float>(1) = 2.0f;
    targets.at<float>(2) = 3.0f;
    
    L1Loss loss_fn;
    Tensor loss = loss_fn.forward(predictions, targets);
    
    EXPECT_NEAR(loss.at<float>(0), 0.0f, epsilon);
}

TEST_F(LossTest, L1Forward_LargeError) {
    Tensor predictions({2}, DType::Float32, Device::CPU);
    predictions.at<float>(0) = -1000.0f;
    predictions.at<float>(1) = 1000.0f;
    
    Tensor targets({2}, DType::Float32, Device::CPU);
    targets.at<float>(0) = 0.0f;
    targets.at<float>(1) = 0.0f;
    
    L1Loss loss_fn;
    Tensor loss = loss_fn.forward(predictions, targets);
    
    // Expected: (1000 + 1000) / 2 = 1000
    float expected_loss = 1000.0f;
    
    EXPECT_NEAR(loss.at<float>(0), expected_loss, epsilon);
}

TEST_F(LossTest, L1Forward_ZeroError) {
    Tensor predictions({2}, DType::Float32, Device::CPU);
    predictions.at<float>(0) = 0.0f;
    predictions.at<float>(1) = 0.0f;
    
    Tensor targets({2}, DType::Float32, Device::CPU);
    targets.at<float>(0) = 0.0f;
    targets.at<float>(1) = 0.0f;
    
    L1Loss loss_fn;
    Tensor loss = loss_fn.forward(predictions, targets);
    
    EXPECT_NEAR(loss.at<float>(0), 0.0f, epsilon);
}

TEST_F(LossTest, L1Forward_LargeInputs) {
    const size_t n = 10000;
    Tensor predictions({n}, DType::Float32, Device::CPU);
    Tensor targets({n}, DType::Float32, Device::CPU);
    
    for (size_t i = 0; i < n; ++i) {
        predictions.at<float>(i) = static_cast<float>(i);
        targets.at<float>(i) = static_cast<float>(i) + 1.0f;
    }
    
    L1Loss loss_fn;
    Tensor loss = loss_fn.forward(predictions, targets);
    
    // Expected: 1.0
    EXPECT_NEAR(loss.at<float>(0), 1.0f, epsilon);
}

// ============ Gradient Tests ============

TEST_F(LossTest, MSEBackward_GradientCheck) {
    Tensor predictions({2}, DType::Float32, Device::CPU);
    predictions.at<float>(0) = 1.0f;
    predictions.at<float>(1) = 2.0f;
    predictions.setRequiresGrad(true);
    
    Tensor targets({2}, DType::Float32, Device::CPU);
    targets.at<float>(0) = 1.5f;
    targets.at<float>(1) = 2.5f;
    
    MSELoss loss_fn;
    Tensor loss = loss_fn.forward(predictions, targets);
    loss.backward();
    
    Tensor grad = *predictions.grad();
    
    // Expected gradient: 2 * (predictions - targets) / n = 2 * (-0.5, -0.5) / 2 = (-0.5, -0.5)
    EXPECT_NEAR(grad.at<float>(0), -0.5f, epsilon);
    EXPECT_NEAR(grad.at<float>(1), -0.5f, epsilon);
}

TEST_F(LossTest, L1Backward_GradientCheck) {
    Tensor predictions({3}, DType::Float32, Device::CPU);
    predictions.at<float>(0) = 1.0f;
    predictions.at<float>(1) = 2.0f;
    predictions.at<float>(2) = 3.0f;
    predictions.setRequiresGrad(true);
    
    Tensor targets({3}, DType::Float32, Device::CPU);
    targets.at<float>(0) = 1.5f;
    targets.at<float>(1) = 2.0f;
    targets.at<float>(2) = 2.5f;
    
    L1Loss loss_fn;
    Tensor loss = loss_fn.forward(predictions, targets);
    loss.backward();
    
    Tensor grad = *predictions.grad();
    
    // Expected gradient: sign(predictions - targets) / n
    // = (-1, 0, 1) / 3
    EXPECT_NEAR(grad.at<float>(0), -1.0f / 3.0f, epsilon);
    EXPECT_NEAR(grad.at<float>(1), 0.0f, epsilon);
    EXPECT_NEAR(grad.at<float>(2), 1.0f / 3.0f, epsilon);
}
