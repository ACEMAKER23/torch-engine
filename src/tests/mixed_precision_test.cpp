#include <gtest/gtest.h>
#include "../core/dtype.h"
#include "../core/dtype_utils.h"
#include "../core/loss_scaler.h"
#include "../tensor/tensor.h"
#include <cmath>

class MixedPrecisionTest : public ::testing::Test {
protected:
    void SetUp() override {}
    float epsilon = 1e-5f;
};

// ============ DType Tests ============

TEST_F(MixedPrecisionTest, DType_Float16_Size) {
    EXPECT_EQ(dtype_size(DType::Float16), 2);
}

TEST_F(MixedPrecisionTest, DType_BFloat16_Size) {
    EXPECT_EQ(dtype_size(DType::BFloat16), 2);
}

TEST_F(MixedPrecisionTest, IsFloatingPoint_Float32) {
    EXPECT_TRUE(is_dtype_floating_point(DType::Float32));
}

TEST_F(MixedPrecisionTest, IsFloatingPoint_Float16) {
    EXPECT_TRUE(is_dtype_floating_point(DType::Float16));
}

TEST_F(MixedPrecisionTest, IsFloatingPoint_BFloat16) {
    EXPECT_TRUE(is_dtype_floating_point(DType::BFloat16));
}

TEST_F(MixedPrecisionTest, IsHalfPrecision_Float16) {
    EXPECT_TRUE(is_half_precision(DType::Float16));
}

TEST_F(MixedPrecisionTest, IsHalfPrecision_BFloat16) {
    EXPECT_TRUE(is_half_precision(DType::BFloat16));
}

TEST_F(MixedPrecisionTest, IsHalfPrecision_Float32) {
    EXPECT_FALSE(is_half_precision(DType::Float32));
}

TEST_F(MixedPrecisionTest, GetMasterDtype_Float16) {
    EXPECT_EQ(get_master_dtype(DType::Float16), DType::Float32);
}

TEST_F(MixedPrecisionTest, GetMasterDtype_BFloat16) {
    EXPECT_EQ(get_master_dtype(DType::BFloat16), DType::Float32);
}

TEST_F(MixedPrecisionTest, GetMasterDtype_Float32) {
    EXPECT_EQ(get_master_dtype(DType::Float32), DType::Float32);
}

// ============ Float16 Conversion Tests ============

TEST_F(MixedPrecisionTest, Float16_FromFloat32_Basic) {
    float f = 1.5f;
    float16_t f16 = float16_t::from_float32(f);
    float converted_back = f16.to_float32();
    
    EXPECT_NEAR(converted_back, f, 0.001f);
}

TEST_F(MixedPrecisionTest, Float16_FromFloat32_Zero) {
    float f = 0.0f;
    float16_t f16 = float16_t::from_float32(f);
    float converted_back = f16.to_float32();
    
    EXPECT_NEAR(converted_back, f, epsilon);
}

TEST_F(MixedPrecisionTest, Float16_FromFloat32_Negative) {
    float f = -2.5f;
    float16_t f16 = float16_t::from_float32(f);
    float converted_back = f16.to_float32();
    
    EXPECT_NEAR(converted_back, f, 0.001f);
}

TEST_F(MixedPrecisionTest, Float16_FromFloat32_Small) {
    float f = 0.001f;
    float16_t f16 = float16_t::from_float32(f);
    float converted_back = f16.to_float32();
    
    // Float16 has limited precision for small numbers
    EXPECT_NEAR(converted_back, f, 0.0001f);
}

// ============ BFloat16 Conversion Tests ============

TEST_F(MixedPrecisionTest, BFloat16_FromFloat32_Basic) {
    float f = 1.5f;
    bfloat16_t bf16 = bfloat16_t::from_float32(f);
    float converted_back = bf16.to_float32();
    
    // BFloat16 keeps exponent but truncates mantissa
    EXPECT_NEAR(converted_back, f, 0.01f);
}

TEST_F(MixedPrecisionTest, BFloat16_FromFloat32_Zero) {
    float f = 0.0f;
    bfloat16_t bf16 = bfloat16_t::from_float32(f);
    float converted_back = bf16.to_float32();
    
    EXPECT_NEAR(converted_back, f, epsilon);
}

TEST_F(MixedPrecisionTest, BFloat16_FromFloat32_Negative) {
    float f = -2.5f;
    bfloat16_t bf16 = bfloat16_t::from_float32(f);
    float converted_back = bf16.to_float32();
    
    EXPECT_NEAR(converted_back, f, 0.01f);
}

// ============ Cast Dtype Tests ============

TEST_F(MixedPrecisionTest, CastDtype_Float32ToFloat32) {
    Tensor tensor({3}, DType::Float32, Device::CPU);
    tensor.at<float>(0) = 1.0f;
    tensor.at<float>(1) = 2.0f;
    tensor.at<float>(2) = 3.0f;
    
    Tensor converted = cast_dtype(tensor, DType::Float32);
    
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(converted.at<float>(i), tensor.at<float>(i), epsilon);
    }
}

TEST_F(MixedPrecisionTest, CastDtype_Float32ToFloat16) {
    Tensor tensor({3}, DType::Float32, Device::CPU);
    tensor.at<float>(0) = 1.0f;
    tensor.at<float>(1) = 2.0f;
    tensor.at<float>(2) = 3.0f;
    
    Tensor converted = cast_dtype(tensor, DType::Float16);
    
    EXPECT_EQ(converted.dtype(), DType::Float16);
    EXPECT_EQ(converted.numel(), 3);
}

TEST_F(MixedPrecisionTest, CastDtype_Float32ToBFloat16) {
    Tensor tensor({3}, DType::Float32, Device::CPU);
    tensor.at<float>(0) = 1.0f;
    tensor.at<float>(1) = 2.0f;
    tensor.at<float>(2) = 3.0f;
    
    Tensor converted = cast_dtype(tensor, DType::BFloat16);
    
    EXPECT_EQ(converted.dtype(), DType::BFloat16);
    EXPECT_EQ(converted.numel(), 3);
}

// ============ Loss Scaler Tests ============

TEST_F(MixedPrecisionTest, LossScaler_InitialScale) {
    LossScaler scaler(2.0f);
    EXPECT_NEAR(scaler.get_scale(), 2.0f, epsilon);
}

TEST_F(MixedPrecisionTest, LossScaler_ScaleLoss) {
    LossScaler scaler(4.0f);
    
    Tensor loss({1}, DType::Float32, Device::CPU);
    loss.at<float>(0) = 2.5f;
    
    Tensor scaled = scaler.scale(loss);
    
    EXPECT_NEAR(scaled.at<float>(0), 10.0f, epsilon);
}

TEST_F(MixedPrecisionTest, LossScaler_UnscaleGrad) {
    LossScaler scaler(4.0f);
    
    Tensor grad({1}, DType::Float32, Device::CPU);
    grad.at<float>(0) = 8.0f;
    
    Tensor unscaled = scaler.unscale(grad);
    
    EXPECT_NEAR(unscaled.at<float>(0), 2.0f, epsilon);
}

TEST_F(MixedPrecisionTest, LossScaler_Reset) {
    LossScaler scaler(2.0f);
    
    // Simulate some scaling operations that might change the scale
    Tensor loss({1}, DType::Float32, Device::CPU);
    loss.at<float>(0) = 1.0f;
    scaler.scale(loss);
    
    scaler.reset();
    
    EXPECT_NEAR(scaler.get_scale(), 2.0f, epsilon);
}

TEST_F(MixedPrecisionTest, LossScaler_CheckValidGrad) {
    LossScaler scaler(2.0f);
    
    Tensor grad({3}, DType::Float32, Device::CPU);
    grad.at<float>(0) = 0.5f;
    grad.at<float>(1) = 1.0f;
    grad.at<float>(2) = 1.5f;
    
    bool valid = scaler.check_and_adjust_scale(grad);
    
    EXPECT_TRUE(valid);
}

TEST_F(MixedPrecisionTest, LossScaler_CheckInfGrad) {
    LossScaler scaler(2.0f);
    
    Tensor grad({2}, DType::Float32, Device::CPU);
    grad.at<float>(0) = 1.0f;
    grad.at<float>(1) = std::numeric_limits<float>::infinity();
    
    bool valid = scaler.check_and_adjust_scale(grad);
    
    EXPECT_FALSE(valid);
    // Scale should be reduced
    EXPECT_NEAR(scaler.get_scale(), 1.0f, epsilon);
}

TEST_F(MixedPrecisionTest, LossScaler_CheckNanGrad) {
    LossScaler scaler(2.0f);
    
    Tensor grad({2}, DType::Float32, Device::CPU);
    grad.at<float>(0) = 1.0f;
    grad.at<float>(1) = std::numeric_limits<float>::quiet_NaN();
    
    bool valid = scaler.check_and_adjust_scale(grad);
    
    EXPECT_FALSE(valid);
    // Scale should be reduced
    EXPECT_NEAR(scaler.get_scale(), 1.0f, epsilon);
}

// ============ Large Input Tests ============

TEST_F(MixedPrecisionTest, CastDtype_LargeInput) {
    const size_t n = 10000;
    Tensor tensor({n}, DType::Float32, Device::CPU);
    
    for (size_t i = 0; i < n; ++i) {
        tensor.at<float>(i) = static_cast<float>(i) * 0.001f;
    }
    
    Tensor converted = cast_dtype(tensor, DType::Float16);
    
    EXPECT_EQ(converted.dtype(), DType::Float16);
    EXPECT_EQ(converted.numel(), n);
}

TEST_F(MixedPrecisionTest, LossScaler_LargeInput) {
    LossScaler scaler(2.0f);
    
    const size_t n = 10000;
    Tensor loss({n}, DType::Float32, Device::CPU);
    
    for (size_t i = 0; i < n; ++i) {
        loss.at<float>(i) = static_cast<float>(i) * 0.001f;
    }
    
    Tensor scaled = scaler.scale(loss);
    
    EXPECT_EQ(scaled.numel(), n);
    EXPECT_NEAR(scaled.at<float>(0), 0.0f, epsilon);
    EXPECT_NEAR(scaled.at<float>(1000), 2.0f, epsilon);
}
