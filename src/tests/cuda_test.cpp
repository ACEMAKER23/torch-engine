#include <gtest/gtest.h>
#include "../tensor/tensor.h"
#include "../core/dtype_utils.h"
#include "../core/cuda_utils.h"

TEST(CUDATest, ToDeviceRoundTrip) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    Tensor a({2, 3}, DType::Float32, Device::CPU);
    for (int i = 0; i < 6; ++i) {
        a.at<float>(i) = static_cast<float>(i);
    }

    Tensor gpu = a.toDevice(Device::CUDA);
    EXPECT_EQ(gpu.device(), Device::CUDA);
    EXPECT_EQ(gpu.shape(), a.shape());

    Tensor back = gpu.toDevice(Device::CPU);
    EXPECT_EQ(back.device(), Device::CPU);

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(back.at<float>(i), a.at<float>(i));
    }
}

TEST(CUDATest, ConstructGPU) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    Tensor a({2, 3}, DType::Float32, Device::CUDA);
    EXPECT_EQ(a.device(), Device::CUDA);
    EXPECT_EQ(a.numel(), 6);
}

TEST(CUDATest, SetDevice) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    cuda_set_device(0);
    EXPECT_EQ(cuda_get_device(), 0);
}

TEST(CUDATest, ElementwiseAdd) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    Tensor a({2, 3}, DType::Float32, Device::CPU);
    Tensor b({2, 3}, DType::Float32, Device::CPU);
    for (int i = 0; i < 6; ++i) {
        a.at<float>(i) = static_cast<float>(i);
        b.at<float>(i) = static_cast<float>(i * 2);
    }

    Tensor a_gpu = a.toDevice(Device::CUDA);
    Tensor b_gpu = b.toDevice(Device::CUDA);
    Tensor c_gpu = a_gpu + b_gpu;
    Tensor c = c_gpu.toDevice(Device::CPU);

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(c.at<float>(i), a.at<float>(i) + b.at<float>(i));
    }
}

TEST(CUDATest, ElementwiseSub) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    Tensor a({2, 3}, DType::Float32, Device::CPU);
    Tensor b({2, 3}, DType::Float32, Device::CPU);
    for (int i = 0; i < 6; ++i) {
        a.at<float>(i) = static_cast<float>(i);
        b.at<float>(i) = static_cast<float>(i * 2);
    }

    Tensor a_gpu = a.toDevice(Device::CUDA);
    Tensor b_gpu = b.toDevice(Device::CUDA);
    Tensor c_gpu = a_gpu - b_gpu;
    Tensor c = c_gpu.toDevice(Device::CPU);

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(c.at<float>(i), a.at<float>(i) - b.at<float>(i));
    }
}

TEST(CUDATest, ElementwiseMul) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    Tensor a({2, 3}, DType::Float32, Device::CPU);
    Tensor b({2, 3}, DType::Float32, Device::CPU);
    for (int i = 0; i < 6; ++i) {
        a.at<float>(i) = static_cast<float>(i);
        b.at<float>(i) = static_cast<float>(i * 2);
    }

    Tensor a_gpu = a.toDevice(Device::CUDA);
    Tensor b_gpu = b.toDevice(Device::CUDA);
    Tensor c_gpu = a_gpu * b_gpu;
    Tensor c = c_gpu.toDevice(Device::CPU);

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(c.at<float>(i), a.at<float>(i) * b.at<float>(i));
    }
}

TEST(CUDATest, ElementwiseDiv) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    Tensor a({2, 3}, DType::Float32, Device::CPU);
    Tensor b({2, 3}, DType::Float32, Device::CPU);
    for (int i = 0; i < 6; ++i) {
        a.at<float>(i) = static_cast<float>(i * 3);
        b.at<float>(i) = static_cast<float>(i + 1);
    }

    Tensor a_gpu = a.toDevice(Device::CUDA);
    Tensor b_gpu = b.toDevice(Device::CUDA);
    Tensor c_gpu = a_gpu / b_gpu;
    Tensor c = c_gpu.toDevice(Device::CPU);

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(c.at<float>(i), a.at<float>(i) / b.at<float>(i));
    }
}

TEST(CUDATest, ElementwiseAddFloat16) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    Tensor a({2, 3}, DType::Float16, Device::CPU);
    Tensor b({2, 3}, DType::Float16, Device::CPU);
    for (int i = 0; i < 6; ++i) {
        a.at<float16_t>(i) = float16_t::from_float32(static_cast<float>(i));
        b.at<float16_t>(i) = float16_t::from_float32(static_cast<float>(i * 2));
    }

    Tensor a_gpu = a.toDevice(Device::CUDA);
    Tensor b_gpu = b.toDevice(Device::CUDA);
    Tensor c_gpu = a_gpu + b_gpu;
    Tensor c = c_gpu.toDevice(Device::CPU);

    for (int i = 0; i < 6; ++i) {
        float expected = a.at<float16_t>(i).to_float32() + b.at<float16_t>(i).to_float32();
        EXPECT_NEAR(c.at<float16_t>(i).to_float32(), expected, 1e-3f);
    }
}

TEST(CUDATest, ElementwiseAddBFloat16) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    Tensor a({2, 3}, DType::BFloat16, Device::CPU);
    Tensor b({2, 3}, DType::BFloat16, Device::CPU);
    for (int i = 0; i < 6; ++i) {
        a.at<bfloat16_t>(i) = bfloat16_t::from_float32(static_cast<float>(i));
        b.at<bfloat16_t>(i) = bfloat16_t::from_float32(static_cast<float>(i * 2));
    }

    Tensor a_gpu = a.toDevice(Device::CUDA);
    Tensor b_gpu = b.toDevice(Device::CUDA);
    Tensor c_gpu = a_gpu + b_gpu;
    Tensor c = c_gpu.toDevice(Device::CPU);

    for (int i = 0; i < 6; ++i) {
        float expected = a.at<bfloat16_t>(i).to_float32() + b.at<bfloat16_t>(i).to_float32();
        EXPECT_NEAR(c.at<bfloat16_t>(i).to_float32(), expected, 1e-3f);
    }
}
