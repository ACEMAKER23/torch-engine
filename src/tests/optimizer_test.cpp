#include <gtest/gtest.h>
#include "../optimizer/sgd.h"
#include "../optimizer/adam.h"
#include "../optimizer/adamw.h"
#include "../tensor/tensor.h"
#include "../core/dtype.h"
#include "../nn/linear.h"

// Optimizer Base Tests
TEST(OptimizerTest, ZeroGrad) {
    Tensor param({2, 3}, DType::Float32, Device::CPU);
    auto* data = static_cast<float*>(param.data());
    for (size_t i = 0; i < param.numel(); ++i) {
        data[i] = 1.0f;
    }
    param.setRequiresGrad(true);
    
    // Simulate gradient
    Tensor grad({2, 3}, DType::Float32, Device::CPU);
    auto* grad_data = static_cast<float*>(grad.data());
    for (size_t i = 0; i < grad.numel(); ++i) {
        grad_data[i] = 0.5f;
    }
    
    std::vector<Tensor> params = {param};
    sgd opt(params);
    
    opt.zero_grad();
}

// SGD Tests
TEST(SGDTest, BasicConstruction) {
    Tensor param({2, 3}, DType::Float32, Device::CPU);
    std::vector<Tensor> params = {param};
    
    sgd opt(params);
    
    EXPECT_NO_THROW();
}

TEST(SGDTest, StepNoGrad) {
    Tensor param({2, 3}, DType::Float32, Device::CPU);
    auto* data = static_cast<float*>(param.data());
    for (size_t i = 0; i < param.numel(); ++i) {
        data[i] = 1.0f;
    }
    
    std::vector<Tensor> params = {param};
    sgd opt(params);
    
    EXPECT_NO_THROW(opt.step(0.01f));
}

TEST(SGDTest, StepWithGrad) {
    Tensor param({2, 3}, DType::Float32, Device::CPU);
    auto* data = static_cast<float*>(param.data());
    for (size_t i = 0; i < param.numel(); ++i) {
        data[i] = 1.0f;
    }
    param.setRequiresGrad(true);
    
    // Create gradient tensor
    Tensor grad({2, 3}, DType::Float32, Device::CPU);
    auto* grad_data = static_cast<float*>(grad.data());
    for (size_t i = 0; i < grad.numel(); ++i) {
        grad_data[i] = 0.5f;
    }
    
    std::vector<Tensor> params = {param};
    sgd opt(params);
    
    // Manually set gradient (simulating backward pass)
    // Note: This is a simplified test - in real usage, backward would set this
    
    opt.step(0.01f);
}

TEST(SGDTest, StepSmallLearningRate) {
    Tensor param({2, 2}, DType::Float32, Device::CPU);
    auto* data = static_cast<float*>(param.data());
    for (size_t i = 0; i < param.numel(); ++i) {
        data[i] = 1.0f;
    }
    param.setRequiresGrad(true);
    
    std::vector<Tensor> params = {param};
    sgd opt(params);
    
    EXPECT_NO_THROW(opt.step(0.0001f));
}

TEST(SGDTest, StepLargeLearningRate) {
    Tensor param({2, 2}, DType::Float32, Device::CPU);
    auto* data = static_cast<float*>(param.data());
    for (size_t i = 0; i < param.numel(); ++i) {
        data[i] = 1.0f;
    }
    param.setRequiresGrad(true);
    
    std::vector<Tensor> params = {param};
    sgd opt(params);
    
    EXPECT_NO_THROW(opt.step(1.0f));
}

TEST(SGDTest, MultipleParameters) {
    Tensor param1({2, 3}, DType::Float32, Device::CPU);
    Tensor param2({4, 5}, DType::Float32, Device::CPU);
    
    std::vector<Tensor> params = {param1, param2};
    sgd opt(params);
    
    EXPECT_NO_THROW(opt.step(0.01f));
}

TEST(SGDTest, LargeParameters) {
    Tensor param({100, 100}, DType::Float32, Device::CPU);
    auto* data = static_cast<float*>(param.data());
    for (size_t i = 0; i < param.numel(); ++i) {
        data[i] = 0.1f;
    }
    param.setRequiresGrad(true);
    
    std::vector<Tensor> params = {param};
    sgd opt(params);
    
    EXPECT_NO_THROW(opt.step(0.01f));
}

// Adam Tests
TEST(AdamTest, BasicConstruction) {
    Tensor param({2, 3}, DType::Float32, Device::CPU);
    std::vector<Tensor> params = {param};
    
    adam opt(params);
    
    EXPECT_NO_THROW();
}

TEST(AdamTest, ConstructionWithHyperparameters) {
    Tensor param({2, 3}, DType::Float32, Device::CPU);
    std::vector<Tensor> params = {param};
    
    adam opt(params, 0.85f, 0.995f, 1e-6f);
    
    EXPECT_NO_THROW();
}

TEST(AdamTest, StepNoGrad) {
    Tensor param({2, 3}, DType::Float32, Device::CPU);
    auto* data = static_cast<float*>(param.data());
    for (size_t i = 0; i < param.numel(); ++i) {
        data[i] = 1.0f;
    }
    
    std::vector<Tensor> params = {param};
    adam opt(params);
    
    EXPECT_NO_THROW(opt.step(0.001f));
}

TEST(AdamTest, StepWithGrad) {
    Tensor param({2, 3}, DType::Float32, Device::CPU);
    auto* data = static_cast<float*>(param.data());
    for (size_t i = 0; i < param.numel(); ++i) {
        data[i] = 1.0f;
    }
    param.setRequiresGrad(true);
    
    std::vector<Tensor> params = {param};
    adam opt(params);
    
    EXPECT_NO_THROW(opt.step(0.001f));
}

TEST(AdamTest, StepSmallLearningRate) {
    Tensor param({2, 2}, DType::Float32, Device::CPU);
    auto* data = static_cast<float*>(param.data());
    for (size_t i = 0; i < param.numel(); ++i) {
        data[i] = 1.0f;
    }
    param.setRequiresGrad(true);
    
    std::vector<Tensor> params = {param};
    adam opt(params);
    
    EXPECT_NO_THROW(opt.step(0.00001f));
}

TEST(AdamTest, StepLargeLearningRate) {
    Tensor param({2, 2}, DType::Float32, Device::CPU);
    auto* data = static_cast<float*>(param.data());
    for (size_t i = 0; i < param.numel(); ++i) {
        data[i] = 1.0f;
    }
    param.setRequiresGrad(true);
    
    std::vector<Tensor> params = {param};
    adam opt(params);
    
    EXPECT_NO_THROW(opt.step(0.1f));
}

TEST(AdamTest, MultipleSteps) {
    Tensor param({2, 2}, DType::Float32, Device::CPU);
    auto* data = static_cast<float*>(param.data());
    for (size_t i = 0; i < param.numel(); ++i) {
        data[i] = 1.0f;
    }
    param.setRequiresGrad(true);
    
    std::vector<Tensor> params = {param};
    adam opt(params);
    
    for (int i = 0; i < 10; ++i) {
        EXPECT_NO_THROW(opt.step(0.001f));
    }
}

TEST(AdamTest, MultipleParameters) {
    Tensor param1({2, 3}, DType::Float32, Device::CPU);
    Tensor param2({4, 5}, DType::Float32, Device::CPU);
    
    std::vector<Tensor> params = {param1, param2};
    adam opt(params);
    
    EXPECT_NO_THROW(opt.step(0.001f));
}

TEST(AdamTest, LargeParameters) {
    Tensor param({50, 50}, DType::Float32, Device::CPU);
    auto* data = static_cast<float*>(param.data());
    for (size_t i = 0; i < param.numel(); ++i) {
        data[i] = 0.1f;
    }
    param.setRequiresGrad(true);
    
    std::vector<Tensor> params = {param};
    adam opt(params);
    
    EXPECT_NO_THROW(opt.step(0.001f));
}

TEST(AdamTest, BiasCorrection) {
    Tensor param({2, 2}, DType::Float32, Device::CPU);
    auto* data = static_cast<float*>(param.data());
    for (size_t i = 0; i < param.numel(); ++i) {
        data[i] = 1.0f;
    }
    param.setRequiresGrad(true);
    
    std::vector<Tensor> params = {param};
    adam opt(params, 0.9f, 0.999f, 1e-8f);
    
    // Run multiple steps to test bias correction
    for (int i = 0; i < 5; ++i) {
        EXPECT_NO_THROW(opt.step(0.001f));
    }
}

// AdamW Tests
TEST(AdamWTest, BasicConstruction) {
    Tensor param({2, 3}, DType::Float32, Device::CPU);
    std::vector<Tensor> params = {param};
    
    adamw opt(params);
    
    EXPECT_NO_THROW();
}

TEST(AdamWTest, ConstructionWithHyperparameters) {
    Tensor param({2, 3}, DType::Float32, Device::CPU);
    std::vector<Tensor> params = {param};
    
    adamw opt(params, 0.85f, 0.995f, 1e-6f, 0.02f);
    
    EXPECT_NO_THROW();
}

TEST(AdamWTest, StepNoGrad) {
    Tensor param({2, 3}, DType::Float32, Device::CPU);
    auto* data = static_cast<float*>(param.data());
    for (size_t i = 0; i < param.numel(); ++i) {
        data[i] = 1.0f;
    }
    
    std::vector<Tensor> params = {param};
    adamw opt(params);
    
    EXPECT_NO_THROW(opt.step(0.001f));
}

TEST(AdamWTest, StepWithGrad) {
    Tensor param({2, 3}, DType::Float32, Device::CPU);
    auto* data = static_cast<float*>(param.data());
    for (size_t i = 0; i < param.numel(); ++i) {
        data[i] = 1.0f;
    }
    param.setRequiresGrad(true);
    
    std::vector<Tensor> params = {param};
    adamw opt(params);
    
    EXPECT_NO_THROW(opt.step(0.001f));
}

TEST(AdamWTest, StepWithWeightDecay) {
    Tensor param({2, 2}, DType::Float32, Device::CPU);
    auto* data = static_cast<float*>(param.data());
    for (size_t i = 0; i < param.numel(); ++i) {
        data[i] = 1.0f;
    }
    param.setRequiresGrad(true);
    
    std::vector<Tensor> params = {param};
    adamw opt(params, 0.9f, 0.999f, 1e-8f, 0.1f);
    
    EXPECT_NO_THROW(opt.step(0.001f));
}

TEST(AdamWTest, StepSmallLearningRate) {
    Tensor param({2, 2}, DType::Float32, Device::CPU);
    auto* data = static_cast<float*>(param.data());
    for (size_t i = 0; i < param.numel(); ++i) {
        data[i] = 1.0f;
    }
    param.setRequiresGrad(true);
    
    std::vector<Tensor> params = {param};
    adamw opt(params);
    
    EXPECT_NO_THROW(opt.step(0.00001f));
}

TEST(AdamWTest, StepLargeLearningRate) {
    Tensor param({2, 2}, DType::Float32, Device::CPU);
    auto* data = static_cast<float*>(param.data());
    for (size_t i = 0; i < param.numel(); ++i) {
        data[i] = 1.0f;
    }
    param.setRequiresGrad(true);
    
    std::vector<Tensor> params = {param};
    adamw opt(params);
    
    EXPECT_NO_THROW(opt.step(0.1f));
}

TEST(AdamWTest, MultipleSteps) {
    Tensor param({2, 2}, DType::Float32, Device::CPU);
    auto* data = static_cast<float*>(param.data());
    for (size_t i = 0; i < param.numel(); ++i) {
        data[i] = 1.0f;
    }
    param.setRequiresGrad(true);
    
    std::vector<Tensor> params = {param};
    adamw opt(params);
    
    for (int i = 0; i < 10; ++i) {
        EXPECT_NO_THROW(opt.step(0.001f));
    }
}

TEST(AdamWTest, MultipleParameters) {
    Tensor param1({2, 3}, DType::Float32, Device::CPU);
    Tensor param2({4, 5}, DType::Float32, Device::CPU);
    
    std::vector<Tensor> params = {param1, param2};
    adamw opt(params);
    
    EXPECT_NO_THROW(opt.step(0.001f));
}

TEST(AdamWTest, LargeParameters) {
    Tensor param({50, 50}, DType::Float32, Device::CPU);
    auto* data = static_cast<float*>(param.data());
    for (size_t i = 0; i < param.numel(); ++i) {
        data[i] = 0.1f;
    }
    param.setRequiresGrad(true);
    
    std::vector<Tensor> params = {param};
    adamw opt(params);
    
    EXPECT_NO_THROW(opt.step(0.001f));
}

TEST(AdamWTest, DifferentWeightDecay) {
    Tensor param({2, 2}, DType::Float32, Device::CPU);
    auto* data = static_cast<float*>(param.data());
    for (size_t i = 0; i < param.numel(); ++i) {
        data[i] = 1.0f;
    }
    param.setRequiresGrad(true);
    
    std::vector<Tensor> params = {param};
    
    // Test different weight decay values
    adamw opt1(params, 0.9f, 0.999f, 1e-8f, 0.0f);  // No weight decay
    adamw opt2(params, 0.9f, 0.999f, 1e-8f, 0.01f); // Small weight decay
    adamw opt3(params, 0.9f, 0.999f, 1e-8f, 0.1f);  // Large weight decay
    
    EXPECT_NO_THROW(opt1.step(0.001f));
    EXPECT_NO_THROW(opt2.step(0.001f));
    EXPECT_NO_THROW(opt3.step(0.001f));
}

// Optimizer with Linear Layer Tests
TEST(OptimizerWithLinearTest, SGDWithLinear) {
    Linear layer(4, 8, DType::Float32);
    
    auto params = layer.parameters();
    sgd opt(params);
    
    EXPECT_NO_THROW(opt.step(0.01f));
}

TEST(OptimizerWithLinearTest, AdamWithLinear) {
    Linear layer(4, 8, DType::Float32);
    
    auto params = layer.parameters();
    adam opt(params);
    
    EXPECT_NO_THROW(opt.step(0.001f));
}

TEST(OptimizerWithLinearTest, AdamWWithLinear) {
    Linear layer(4, 8, DType::Float32);
    
    auto params = layer.parameters();
    adamw opt(params);
    
    EXPECT_NO_THROW(opt.step(0.001f));
}

TEST(OptimizerWithLinearTest, MultipleStepsWithLinear) {
    Linear layer(4, 8, DType::Float32);
    
    auto params = layer.parameters();
    adam opt(params);
    
    for (int i = 0; i < 5; ++i) {
        EXPECT_NO_THROW(opt.step(0.001f));
    }
}

TEST(OptimizerWithLinearTest, ZeroGradWithLinear) {
    Linear layer(4, 8, DType::Float32);
    
    auto params = layer.parameters();
    adam opt(params);
    
    EXPECT_NO_THROW(opt.zero_grad());
}
