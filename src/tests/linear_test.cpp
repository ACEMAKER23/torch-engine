#include <gtest/gtest.h>
#include "../nn/linear.h"
#include "../tensor/tensor.h"
#include "../core/dtype.h"

TEST(LinearTest, BasicConstruction) {
    Linear layer(3, 4, DType::Float32);
    
    EXPECT_EQ(layer.weight().shape()[0], 4);
    EXPECT_EQ(layer.weight().shape()[1], 3);
    EXPECT_EQ(layer.bias().shape()[0], 4);
}

TEST(LinearTest, ForwardPass) {
    Linear layer(3, 4, DType::Float32);
    
    // Create input tensor [batch_size=2, in_features=3]
    Tensor input({2, 3}, DType::Float32, Device::CPU);
    auto* input_data = static_cast<float*>(input.data());
    input_data[0] = 1.0f; input_data[1] = 2.0f; input_data[2] = 3.0f;
    input_data[3] = 4.0f; input_data[4] = 5.0f; input_data[5] = 6.0f;
    
    Tensor output = layer.forward(input);
    
    // Output should be [batch_size=2, out_features=4]
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 4);
}

TEST(LinearTest, WeightInitialization) {
    Linear layer(10, 5, DType::Float32);
    
    // Check that weights are initialized (not all zeros)
    const auto* weight_data = static_cast<const float*>(layer.weight().data());
    bool has_non_zero = false;
    for (size_t i = 0; i < layer.weight().numel(); ++i) {
        if (weight_data[i] != 0.0f) {
            has_non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(has_non_zero);
}

TEST(LinearTest, BiasInitialization) {
    Linear layer(10, 5, DType::Float32);
    
    // Check that bias is initialized to zeros
    const auto* bias_data = static_cast<const float*>(layer.bias().data());
    for (size_t i = 0; i < layer.bias().numel(); ++i) {
        EXPECT_FLOAT_EQ(bias_data[i], 0.0f);
    }
}

TEST(LinearTest, DifferentDtype) {
    EXPECT_THROW(Linear layer(3, 4, DType::Int32), std::runtime_error);
}

TEST(LinearTest, InvalidInFeatures) {
    EXPECT_THROW(Linear layer(0, 4, DType::Float32), std::runtime_error);
}

TEST(LinearTest, GetterMethodsConst) {
    const Linear layer(3, 4, DType::Float32);
    
    // Should be able to call const getters on const object
    const Tensor& weight = layer.weight();
    const Tensor& bias = layer.bias();
    
    EXPECT_EQ(weight.shape()[0], 4);
    EXPECT_EQ(bias.shape()[0], 4);
}
