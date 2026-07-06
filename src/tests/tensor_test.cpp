#include <gtest/gtest.h>
#include "../tensor/tensor.h"
#include "../core/dtype.h"
#include <cstring>

TEST(TensorTest, BasicConstruction) {
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    EXPECT_EQ(a.numel(), 6);
    EXPECT_EQ(a.shape().size(), 2);
    EXPECT_EQ(a.shape()[0], 2);
    EXPECT_EQ(a.shape()[1], 3);
}

TEST(TensorTest, CopySemantics) {
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    
    // Shallow copy - should share storage
    Tensor b = a;
    
    EXPECT_EQ(a.numel(), b.numel());
    EXPECT_EQ(a.shape(), b.shape());
    
    // Verify they share the same storage
    EXPECT_EQ(a.impl_->storage(), b.impl_->storage());
}

TEST(TensorTest, MoveSemantics) {
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    auto original_storage = a.impl_->storage();
    
    // Move - should transfer ownership
    Tensor b = std::move(a);
    
    EXPECT_EQ(b.numel(), 6);
    EXPECT_EQ(b.impl_->storage(), original_storage);
}

TEST(TensorTest, CloneDeepCopy) {
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    
    // Initialize some data
    float* data = static_cast<float*>(a.impl_->storage()->data());
    for (int i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    // Deep copy
    Tensor b = a.clone();
    
    // Verify data is copied
    float* data_a = static_cast<float*>(a.impl_->storage()->data());
    float* data_b = static_cast<float*>(b.impl_->storage()->data());
    
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(data_a[i], data_b[i]);
    }
    
    // Verify storage is different
    EXPECT_NE(a.impl_->storage(), b.impl_->storage());
    
    // Modify original, verify clone is independent
    data_a[0] = 999.0f;
    EXPECT_EQ(data_b[0], 0.0f);
}

TEST(TensorTest, ViewReshape) {
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    
    // Initialize data
    float* data = static_cast<float*>(a.impl_->storage()->data());
    for (int i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    auto original_storage = a.impl_->storage();
    
    // View reshape
    a.view({3, 2});
    
    EXPECT_EQ(a.shape().size(), 2);
    EXPECT_EQ(a.shape()[0], 3);
    EXPECT_EQ(a.shape()[1], 2);
    EXPECT_EQ(a.numel(), 6);
    
    // Verify storage is shared (zero-copy)
    EXPECT_EQ(a.impl_->storage(), original_storage);
    
    // Verify data is still accessible
    float* data_after = static_cast<float*>(a.impl_->storage()->data());
    EXPECT_EQ(data_after[0], 0.0f);
    EXPECT_EQ(data_after[5], 5.0f);
}

TEST(TensorTest, ViewInvalidShape) {
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    
    // Try to view with different numel - should throw
    EXPECT_THROW(a.view({4, 2}), std::runtime_error);
}

TEST(TensorTest, SliceBasic) {
    Tensor a({5, 3}, DType::Float32, Device::CPU);
    
    // Initialize data
    float* data = static_cast<float*>(a.impl_->storage()->data());
    for (int i = 0; i < 15; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    auto original_storage = a.impl_->storage();
    
    // Slice rows 1-3
    Tensor b = a.slice(1, 3, 0);
    
    EXPECT_EQ(b.shape()[0], 2);  // 3 - 1 = 2 rows
    EXPECT_EQ(b.shape()[1], 3);  // columns unchanged
    EXPECT_EQ(b.numel(), 6);
    
    // Verify storage is shared (zero-copy)
    EXPECT_EQ(b.impl_->storage(), original_storage);
    
    // Verify offset is correct
    EXPECT_EQ(b.impl_->offset(), 3);  // 1 * stride[0] = 1 * 3 = 3
    
    // Verify data points to correct location
    float* data_b = static_cast<float*>(b.impl_->storage()->data());
    EXPECT_EQ(data_b[3], 3.0f);  // First element of slice
    EXPECT_EQ(data_b[8], 8.0f);  // Last element of slice
}

TEST(TensorTest, NestedSlice) {
    Tensor a({10, 5}, DType::Float32, Device::CPU);
    
    // Initialize data
    float* data = static_cast<float*>(a.impl_->storage()->data());
    for (int i = 0; i < 50; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    auto original_storage = a.impl_->storage();
    
    // First slice: rows 2-6
    Tensor b = a.slice(2, 6, 0);
    EXPECT_EQ(b.impl_->offset(), 10);  // 2 * stride[0] = 2 * 5 = 10
    
    // Second slice: rows 1-3 of the already sliced tensor
    Tensor c = b.slice(1, 3, 0);
    EXPECT_EQ(c.impl_->offset(), 15);  // 10 + 1 * 5 = 15
    
    // Verify storage is still shared
    EXPECT_EQ(c.impl_->storage(), original_storage);
    
    // Verify data points to correct location (row 3 of original)
    float* data_c = static_cast<float*>(c.impl_->storage()->data());
    EXPECT_EQ(data_c[15], 15.0f);
}

TEST(TensorTest, StridesComputation) {
    Tensor a({2, 3, 4}, DType::Float32, Device::CPU);
    
    auto strides = a.strides();
    EXPECT_EQ(strides.size(), 3);
    EXPECT_EQ(strides[0], 12);  // 3 * 4
    EXPECT_EQ(strides[1], 4);   // 4
    EXPECT_EQ(strides[2], 1);   // 1
}

TEST(TensorTest, Numel) {
    Tensor a({2, 3, 4}, DType::Float32, Device::CPU);
    EXPECT_EQ(a.numel(), 24);
    
    Tensor b({10}, DType::Float32, Device::CPU);
    EXPECT_EQ(b.numel(), 10);
}

// Elementwise Operations Tests
TEST(TensorTest, ElementwiseAddFloat32) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    
    float* data_a = static_cast<float*>(a.impl_->storage()->data());
    float* data_b = static_cast<float*>(b.impl_->storage()->data());
    for (int i = 0; i < 4; ++i) {
        data_a[i] = 1.0f;
        data_b[i] = 2.0f;
    }
    
    Tensor c = a + b;
    float* data_c = static_cast<float*>(c.impl_->storage()->data());
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(data_c[i], 3.0f);
    }
}

TEST(TensorTest, ElementwiseAddInt32) {
    Tensor a({2, 2}, DType::Int32, Device::CPU);
    Tensor b({2, 2}, DType::Int32, Device::CPU);
    
    int32_t* data_a = static_cast<int32_t*>(a.impl_->storage()->data());
    int32_t* data_b = static_cast<int32_t*>(b.impl_->storage()->data());
    for (int i = 0; i < 4; ++i) {
        data_a[i] = 1;
        data_b[i] = 2;
    }
    
    Tensor c = a + b;
    int32_t* data_c = static_cast<int32_t*>(c.impl_->storage()->data());
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(data_c[i], 3);
    }
}

TEST(TensorTest, ElementwiseAddInt64) {
    Tensor a({2, 2}, DType::Int64, Device::CPU);
    Tensor b({2, 2}, DType::Int64, Device::CPU);
    
    int64_t* data_a = static_cast<int64_t*>(a.impl_->storage()->data());
    int64_t* data_b = static_cast<int64_t*>(b.impl_->storage()->data());
    for (int i = 0; i < 4; ++i) {
        data_a[i] = 1;
        data_b[i] = 2;
    }
    
    Tensor c = a + b;
    int64_t* data_c = static_cast<int64_t*>(c.impl_->storage()->data());
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(data_c[i], 3);
    }
}

TEST(TensorTest, ElementwiseSubFloat32) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    
    float* data_a = static_cast<float*>(a.impl_->storage()->data());
    float* data_b = static_cast<float*>(b.impl_->storage()->data());
    for (int i = 0; i < 4; ++i) {
        data_a[i] = 5.0f;
        data_b[i] = 2.0f;
    }
    
    Tensor c = a - b;
    float* data_c = static_cast<float*>(c.impl_->storage()->data());
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(data_c[i], 3.0f);
    }
}

TEST(TensorTest, ElementwiseMulFloat32) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    
    float* data_a = static_cast<float*>(a.impl_->storage()->data());
    float* data_b = static_cast<float*>(b.impl_->storage()->data());
    for (int i = 0; i < 4; ++i) {
        data_a[i] = 3.0f;
        data_b[i] = 2.0f;
    }
    
    Tensor c = a * b;
    float* data_c = static_cast<float*>(c.impl_->storage()->data());
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(data_c[i], 6.0f);
    }
}

TEST(TensorTest, ElementwiseDivFloat32) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    
    float* data_a = static_cast<float*>(a.impl_->storage()->data());
    float* data_b = static_cast<float*>(b.impl_->storage()->data());
    for (int i = 0; i < 4; ++i) {
        data_a[i] = 6.0f;
        data_b[i] = 2.0f;
    }
    
    Tensor c = a / b;
    float* data_c = static_cast<float*>(c.impl_->storage()->data());
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(data_c[i], 3.0f);
    }
}

TEST(TensorTest, ElementwiseShapeMismatch) {
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    Tensor b({3, 2}, DType::Float32, Device::CPU);
    
    EXPECT_THROW(a + b, std::runtime_error);
    EXPECT_THROW(a - b, std::runtime_error);
    EXPECT_THROW(a * b, std::runtime_error);
    EXPECT_THROW(a / b, std::runtime_error);
}

TEST(TensorTest, ElementwiseDtypeMismatch) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    Tensor b({2, 2}, DType::Int32, Device::CPU);
    
    EXPECT_THROW(a + b, std::runtime_error);
    EXPECT_THROW(a - b, std::runtime_error);
    EXPECT_THROW(a * b, std::runtime_error);
    EXPECT_THROW(a / b, std::runtime_error);
}

// Reduction Operations Tests
TEST(TensorTest, ReductionSumFloat32) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    float* data = static_cast<float*>(a.impl_->storage()->data());
    for (int i = 0; i < 4; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    EXPECT_FLOAT_EQ(a.sum<float>(), 6.0f);
}

TEST(TensorTest, ReductionMeanFloat32) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    float* data = static_cast<float*>(a.impl_->storage()->data());
    for (int i = 0; i < 4; ++i) {
        data[i] = 1.0f;
    }
    
    EXPECT_FLOAT_EQ(a.mean<float>(), 1.0f);
}

TEST(TensorTest, ReductionMaxFloat32) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    float* data = static_cast<float*>(a.impl_->storage()->data());
    data[0] = 1.0f;
    data[1] = 5.0f;
    data[2] = 3.0f;
    data[3] = 2.0f;
    
    EXPECT_FLOAT_EQ(a.max<float>(), 5.0f);
}

TEST(TensorTest, ReductionMinFloat32) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    float* data = static_cast<float*>(a.impl_->storage()->data());
    data[0] = 1.0f;
    data[1] = 5.0f;
    data[2] = 3.0f;
    data[3] = 2.0f;
    
    EXPECT_FLOAT_EQ(a.min<float>(), 1.0f);
}

// at() Access Tests
TEST(TensorTest, AtFlatIndexFloat32) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    float* data = static_cast<float*>(a.impl_->storage()->data());
    for (int i = 0; i < 4; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    EXPECT_FLOAT_EQ(a.at<float>(0), 0.0f);
    EXPECT_FLOAT_EQ(a.at<float>(1), 1.0f);
    EXPECT_FLOAT_EQ(a.at<float>(3), 3.0f);
}

TEST(TensorTest, AtMultiIndexFloat32) {
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    float* data = static_cast<float*>(a.impl_->storage()->data());
    for (int i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    EXPECT_FLOAT_EQ(a.at<float>({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(a.at<float>({0, 1}), 1.0f);
    EXPECT_FLOAT_EQ(a.at<float>({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(a.at<float>({1, 2}), 5.0f);
}

TEST(TensorTest, AtTypeMismatch) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    
    EXPECT_THROW(a.at<int32_t>(0), std::runtime_error);
    EXPECT_THROW(a.at<int64_t>(0), std::runtime_error);
}

// Transpose Tests
TEST(TensorTest, Transpose2D) {
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    float* data = static_cast<float*>(a.impl_->storage()->data());
    for (int i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    a.transpose();
    
    EXPECT_EQ(a.shape()[0], 3);
    EXPECT_EQ(a.shape()[1], 2);
    EXPECT_EQ(a.numel(), 6);
}

TEST(TensorTest, TransposeZeroCopy) {
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    float* data = static_cast<float*>(a.impl_->storage()->data());
    for (int i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    auto original_storage = a.impl_->storage();
    a.transpose();
    
    // Verify storage is shared (zero-copy)
    EXPECT_EQ(a.impl_->storage(), original_storage);
}

TEST(TensorTest, TransposeInvalid) {
    Tensor a({2, 3, 4}, DType::Float32, Device::CPU);
    
    EXPECT_THROW(a.transpose(), std::runtime_error);
}

// Activation Function Tests
TEST(TensorTest, ActivationRelu) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    float* data = static_cast<float*>(a.impl_->storage()->data());
    data[0] = -1.0f;
    data[1] = 2.0f;
    data[2] = -3.0f;
    data[3] = 4.0f;
    
    Tensor b = a.relu();
    float* data_b = static_cast<float*>(b.impl_->storage()->data());
    
    EXPECT_FLOAT_EQ(data_b[0], 0.0f);
    EXPECT_FLOAT_EQ(data_b[1], 2.0f);
    EXPECT_FLOAT_EQ(data_b[2], 0.0f);
    EXPECT_FLOAT_EQ(data_b[3], 4.0f);
}

TEST(TensorTest, ActivationReluInplace) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    float* data = static_cast<float*>(a.impl_->storage()->data());
    data[0] = -1.0f;
    data[1] = 2.0f;
    data[2] = -3.0f;
    data[3] = 4.0f;
    
    a.relu_<float>();
    
    EXPECT_FLOAT_EQ(data[0], 0.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 0.0f);
    EXPECT_FLOAT_EQ(data[3], 4.0f);
}

TEST(TensorTest, ActivationGelu) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    float* data = static_cast<float*>(a.impl_->storage()->data());
    data[0] = 0.0f;
    data[1] = 1.0f;
    data[2] = -1.0f;
    data[3] = 2.0f;
    
    Tensor b = a.gelu();
    float* data_b = static_cast<float*>(b.impl_->storage()->data());
    
    // GELU(0) ≈ 0
    EXPECT_NEAR(data_b[0], 0.0f, 0.01f);
    // GELU(1) ≈ 0.84
    EXPECT_NEAR(data_b[1], 0.84f, 0.01f);
    // GELU(-1) ≈ -0.16 (GELU is odd function)
    EXPECT_NEAR(data_b[2], -0.16f, 0.01f);
}

TEST(TensorTest, ActivationSigmoid) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    float* data = static_cast<float*>(a.impl_->storage()->data());
    data[0] = 0.0f;
    data[1] = 1.0f;
    
    Tensor b = a.sigmoid();
    float* data_b = static_cast<float*>(b.impl_->storage()->data());
    
    // sigmoid(0) = 0.5
    EXPECT_NEAR(data_b[0], 0.5f, 0.01f);
    // sigmoid(1) ≈ 0.73
    EXPECT_NEAR(data_b[1], 0.73f, 0.01f);
}

TEST(TensorTest, ActivationSqrt) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    float* data = static_cast<float*>(a.impl_->storage()->data());
    data[0] = 4.0f;
    data[1] = 9.0f;
    
    Tensor b = a.sqrt();
    float* data_b = static_cast<float*>(b.impl_->storage()->data());
    
    EXPECT_NEAR(data_b[0], 2.0f, 0.01f);
    EXPECT_NEAR(data_b[1], 3.0f, 0.01f);
}

TEST(TensorTest, ActivationExp) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    float* data = static_cast<float*>(a.impl_->storage()->data());
    data[0] = 0.0f;
    data[1] = 1.0f;
    
    Tensor b = a.exp();
    float* data_b = static_cast<float*>(b.impl_->storage()->data());
    
    EXPECT_NEAR(data_b[0], 1.0f, 0.01f);
    EXPECT_NEAR(data_b[1], 2.718f, 0.01f);
}

TEST(TensorTest, ActivationLog) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    float* data = static_cast<float*>(a.impl_->storage()->data());
    data[0] = 1.0f;
    data[1] = 2.718f;
    
    Tensor b = a.log();
    float* data_b = static_cast<float*>(b.impl_->storage()->data());
    
    EXPECT_NEAR(data_b[0], 0.0f, 0.01f);
    EXPECT_NEAR(data_b[1], 1.0f, 0.01f);
}

TEST(TensorTest, ActivationPow) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    float* data = static_cast<float*>(a.impl_->storage()->data());
    data[0] = 2.0f;
    data[1] = 3.0f;
    
    Tensor b = a.pow<float>(2.0f);
    float* data_b = static_cast<float*>(b.impl_->storage()->data());
    
    EXPECT_NEAR(data_b[0], 4.0f, 0.01f);
    EXPECT_NEAR(data_b[1], 9.0f, 0.01f);
}

// Matrix Multiplication Tests
TEST(TensorTest, MatmulFloat32) {
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    Tensor b({3, 2}, DType::Float32, Device::CPU);
    
    float* data_a = static_cast<float*>(a.impl_->storage()->data());
    float* data_b = static_cast<float*>(b.impl_->storage()->data());
    
    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    for (int i = 0; i < 6; ++i) {
        data_a[i] = static_cast<float>(i + 1);
    }
    
    // B = [[1, 2],
    //      [3, 4],
    //      [5, 6]]
    for (int i = 0; i < 6; ++i) {
        data_b[i] = static_cast<float>(i + 1);
    }
    
    Tensor c = a.matmul(b);
    float* data_c = static_cast<float*>(c.impl_->storage()->data());
    
    // C = A @ B = [[22, 28],
    //               [49, 64]]
    EXPECT_FLOAT_EQ(data_c[0], 22.0f);  // 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
    EXPECT_FLOAT_EQ(data_c[1], 28.0f);  // 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
    EXPECT_FLOAT_EQ(data_c[2], 49.0f);  // 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
    EXPECT_FLOAT_EQ(data_c[3], 64.0f);  // 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64
    
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);
}

TEST(TensorTest, MatmulInt32) {
    Tensor a({2, 2}, DType::Int32, Device::CPU);
    Tensor b({2, 2}, DType::Int32, Device::CPU);
    
    int32_t* data_a = static_cast<int32_t*>(a.impl_->storage()->data());
    int32_t* data_b = static_cast<int32_t*>(b.impl_->storage()->data());
    
    // A = [[1, 2],
    //      [3, 4]]
    data_a[0] = 1; data_a[1] = 2; data_a[2] = 3; data_a[3] = 4;
    
    // B = [[5, 6],
    //      [7, 8]]
    data_b[0] = 5; data_b[1] = 6; data_b[2] = 7; data_b[3] = 8;
    
    Tensor c = a.matmul(b);
    int32_t* data_c = static_cast<int32_t*>(c.impl_->storage()->data());
    
    // C = [[19, 22],
    //      [43, 50]]
    EXPECT_EQ(data_c[0], 19);  // 1*5 + 2*7 = 5 + 14 = 19
    EXPECT_EQ(data_c[1], 22);  // 1*6 + 2*8 = 6 + 16 = 22
    EXPECT_EQ(data_c[2], 43);  // 3*5 + 4*7 = 15 + 28 = 43
    EXPECT_EQ(data_c[3], 50);  // 3*6 + 4*8 = 18 + 32 = 50
}

TEST(TensorTest, MatmulDimensionMismatch) {
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    Tensor b({2, 2}, DType::Float32, Device::CPU);  // Wrong: should be (3, 2)
    
    EXPECT_THROW(a.matmul(b), std::runtime_error);
}

TEST(TensorTest, Matmul3D) {
    // Batch matmul: (2, 2, 3) @ (2, 3, 2) -> (2, 2, 2)
    Tensor a({2, 2, 3}, DType::Float32, Device::CPU);
    Tensor b({2, 3, 2}, DType::Float32, Device::CPU);
    
    float* data_a = static_cast<float*>(a.impl_->storage()->data());
    float* data_b = static_cast<float*>(b.impl_->storage()->data());
    
    // Initialize with sequential values
    for (int i = 0; i < 12; ++i) {
        data_a[i] = static_cast<float>(i + 1);
    }
    for (int i = 0; i < 12; ++i) {
        data_b[i] = static_cast<float>(i + 1);
    }
    
    Tensor c = a.matmul(b);
    
    // Check output shape
    EXPECT_EQ(c.shape().size(), 3);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);
    EXPECT_EQ(c.shape()[2], 2);
    
    // Check that computation ran without error
    EXPECT_EQ(c.numel(), 8);
}

TEST(TensorTest, Matmul4D) {
    // 4D batch matmul: (2, 2, 2, 3) @ (2, 2, 3, 2) -> (2, 2, 2, 2)
    Tensor a({2, 2, 2, 3}, DType::Float32, Device::CPU);
    Tensor b({2, 2, 3, 2}, DType::Float32, Device::CPU);
    
    float* data_a = static_cast<float*>(a.impl_->storage()->data());
    float* data_b = static_cast<float*>(b.impl_->storage()->data());
    
    // Initialize with sequential values
    for (int i = 0; i < 24; ++i) {
        data_a[i] = static_cast<float>(i + 1);
    }
    for (int i = 0; i < 24; ++i) {
        data_b[i] = static_cast<float>(i + 1);
    }
    
    Tensor c = a.matmul(b);
    
    // Check output shape
    EXPECT_EQ(c.shape().size(), 4);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);
    EXPECT_EQ(c.shape()[2], 2);
    EXPECT_EQ(c.shape()[3], 2);
    
    // Check that computation ran without error
    EXPECT_EQ(c.numel(), 16);
}

TEST(TensorTest, MatmulBroadcast) {
    // Broadcast: (2, 3) @ (3, 4) -> (2, 4) (no batch dims, just 2D matmul)
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    Tensor b({3, 4}, DType::Float32, Device::CPU);
    
    float* data_a = static_cast<float*>(a.impl_->storage()->data());
    float* data_b = static_cast<float*>(b.impl_->storage()->data());
    
    for (int i = 0; i < 6; ++i) {
        data_a[i] = static_cast<float>(i + 1);
    }
    for (int i = 0; i < 12; ++i) {
        data_b[i] = static_cast<float>(i + 1);
    }
    
    Tensor c = a.matmul(b);
    
    EXPECT_EQ(c.shape().size(), 2);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 4);
}

TEST(TensorTest, MatmulBroadcastBatch) {
    // Broadcast batch: (3, 2, 3) @ (2, 3, 4) -> (3, 2, 4)
    // First tensor has batch dim (3,), second has (2,)
    // This should fail since 3 and 2 are not broadcastable
    Tensor a({3, 2, 3}, DType::Float32, Device::CPU);
    Tensor b({2, 3, 4}, DType::Float32, Device::CPU);
    
    EXPECT_THROW(a.matmul(b), std::runtime_error);
}

TEST(TensorTest, MatmulBroadcastSingleBatch) {
    // Broadcast: (1, 2, 3) @ (3, 3, 4) -> (3, 2, 4)
    // First tensor has batch dim (1,) which broadcasts to (3,)
    // Matrix dims: (2, 3) @ (3, 4) -> (2, 4)
    Tensor a({1, 2, 3}, DType::Float32, Device::CPU);
    Tensor b({3, 3, 4}, DType::Float32, Device::CPU);
    
    float* data_a = static_cast<float*>(a.data());
    float* data_b = static_cast<float*>(b.data());
    
    for (int i = 0; i < 6; ++i) {
        data_a[i] = static_cast<float>(i + 1);
    }
    for (int i = 0; i < 36; ++i) {
        data_b[i] = static_cast<float>(i + 1);
    }
    
    Tensor c = a.matmul(b);
    
    EXPECT_EQ(c.shape().size(), 3);
    EXPECT_EQ(c.shape()[0], 3);
    EXPECT_EQ(c.shape()[1], 2);
    EXPECT_EQ(c.shape()[2], 4);
}

TEST(TensorTest, MatmulBroadcastIncompatible) {
    // Incompatible broadcast: (2, 2, 3) @ (3, 3, 4)
    // Batch dims (2,) and (3,) are not broadcastable
    Tensor a({2, 2, 3}, DType::Float32, Device::CPU);
    Tensor b({3, 3, 4}, DType::Float32, Device::CPU);
    
    EXPECT_THROW(a.matmul(b), std::runtime_error);
}

TEST(TensorTest, ElementwiseBroadcast) {
    // Broadcast: (2, 3) + (1, 3) -> (2, 3)
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    Tensor b({1, 3}, DType::Float32, Device::CPU);
    
    float* data_a = static_cast<float*>(a.data());
    float* data_b = static_cast<float*>(b.data());
    
    // a = [[1, 2, 3],
    //      [4, 5, 6]]
    for (int i = 0; i < 6; ++i) {
        data_a[i] = static_cast<float>(i + 1);
    }
    
    // b = [[10, 20, 30]]
    data_b[0] = 10.0f;
    data_b[1] = 20.0f;
    data_b[2] = 30.0f;
    
    Tensor c = a + b;
    
    // c = [[11, 22, 33],
    //      [14, 25, 36]]
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 3);
    
    float* data_c = static_cast<float*>(c.data());
    EXPECT_FLOAT_EQ(data_c[0], 11.0f);
    EXPECT_FLOAT_EQ(data_c[1], 22.0f);
    EXPECT_FLOAT_EQ(data_c[2], 33.0f);
    EXPECT_FLOAT_EQ(data_c[3], 14.0f);
    EXPECT_FLOAT_EQ(data_c[4], 25.0f);
    EXPECT_FLOAT_EQ(data_c[5], 36.0f);
}

TEST(TensorTest, ElementwiseBroadcast2D) {
    // Broadcast: (3, 1) + (1, 4) -> (3, 4)
    Tensor a({3, 1}, DType::Float32, Device::CPU);
    Tensor b({1, 4}, DType::Float32, Device::CPU);
    
    float* data_a = static_cast<float*>(a.data());
    float* data_b = static_cast<float*>(b.data());
    
    // a = [[1], [2], [3]]
    data_a[0] = 1.0f;
    data_a[1] = 2.0f;
    data_a[2] = 3.0f;
    
    // b = [[10, 20, 30, 40]]
    data_b[0] = 10.0f;
    data_b[1] = 20.0f;
    data_b[2] = 30.0f;
    data_b[3] = 40.0f;
    
    Tensor c = a + b;
    
    // c = [[11, 21, 31, 41],
    //      [12, 22, 32, 42],
    //      [13, 23, 33, 43]]
    EXPECT_EQ(c.shape()[0], 3);
    EXPECT_EQ(c.shape()[1], 4);
    
    float* data_c = static_cast<float*>(c.data());
    EXPECT_FLOAT_EQ(data_c[0], 11.0f);
    EXPECT_FLOAT_EQ(data_c[1], 21.0f);
    EXPECT_FLOAT_EQ(data_c[2], 31.0f);
    EXPECT_FLOAT_EQ(data_c[3], 41.0f);
    EXPECT_FLOAT_EQ(data_c[4], 12.0f);
    EXPECT_FLOAT_EQ(data_c[5], 22.0f);
    EXPECT_FLOAT_EQ(data_c[6], 32.0f);
    EXPECT_FLOAT_EQ(data_c[7], 42.0f);
    EXPECT_FLOAT_EQ(data_c[8], 13.0f);
    EXPECT_FLOAT_EQ(data_c[9], 23.0f);
    EXPECT_FLOAT_EQ(data_c[10], 33.0f);
    EXPECT_FLOAT_EQ(data_c[11], 43.0f);
}

TEST(TensorTest, ElementwiseBroadcastIncompatible) {
    // Incompatible broadcast: (2, 3) + (4, 5)
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    Tensor b({4, 5}, DType::Float32, Device::CPU);
    
    EXPECT_THROW(a + b, std::runtime_error);
}

TEST(TensorTest, FillFloat32) {
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    
    a.fill_<float>(5.0f);
    
    float* data = static_cast<float*>(a.data());
    for (size_t i = 0; i < a.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 5.0f);
    }
}

TEST(TensorTest, FillInt32) {
    Tensor a({2, 3}, DType::Int32, Device::CPU);
    
    a.fill_<int32_t>(42);
    
    int32_t* data = static_cast<int32_t*>(a.data());
    for (size_t i = 0; i < a.numel(); ++i) {
        EXPECT_EQ(data[i], 42);
    }
}

TEST(TensorTest, FillInt64) {
    Tensor a({2, 3}, DType::Int64, Device::CPU);
    
    a.fill_<int64_t>(100);
    
    int64_t* data = static_cast<int64_t*>(a.data());
    for (size_t i = 0; i < a.numel(); ++i) {
        EXPECT_EQ(data[i], 100);
    }
}

TEST(TensorTest, FillZero) {
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    
    // Initialize with some values
    float* data = static_cast<float*>(a.data());
    for (size_t i = 0; i < a.numel(); ++i) {
        data[i] = static_cast<float>(i);
    }
    
    // Fill with zero
    a.fill_<float>(0.0f);
    
    for (size_t i = 0; i < a.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
}

TEST(TensorTest, ContiguousBasic) {
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    
    // Initialize with some values
    float* data = static_cast<float*>(a.data());
    for (size_t i = 0; i < a.numel(); ++i) {
        data[i] = static_cast<float>(i);
    }
    
    Tensor b = a.contiguous();
    
    // Check shape is preserved
    EXPECT_EQ(b.shape()[0], 2);
    EXPECT_EQ(b.shape()[1], 3);
    
    // Check data is copied correctly
    const float* b_data = static_cast<const float*>(b.data());
    for (size_t i = 0; i < b.numel(); ++i) {
        EXPECT_FLOAT_EQ(b_data[i], static_cast<float>(i));
    }
    
    // Modify original and check contiguous copy is unchanged (proves it's a deep copy)
    data[0] = 999.0f;
    EXPECT_FLOAT_EQ(b_data[0], 0.0f);
}

TEST(TensorTest, ContiguousAfterTranspose) {
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    
    // Initialize with some values
    float* data = static_cast<float*>(a.data());
    for (size_t i = 0; i < a.numel(); ++i) {
        data[i] = static_cast<float>(i);
    }
    
    // Transpose the tensor
    a.transpose(0, 1);
    
    // Make it contiguous
    Tensor b = a.contiguous();
    
    // Check shape is [3, 2] after transpose
    EXPECT_EQ(b.shape()[0], 3);
    EXPECT_EQ(b.shape()[1], 2);
    
    // Check data is accessible
    const float* b_data = static_cast<const float*>(b.data());
    EXPECT_EQ(b.numel(), 6);
}

TEST(TensorTest, ContiguousInt32) {
    Tensor a({2, 3}, DType::Int32, Device::CPU);
    
    // Initialize with some values
    int32_t* data = static_cast<int32_t*>(a.data());
    for (size_t i = 0; i < a.numel(); ++i) {
        data[i] = static_cast<int32_t>(i);
    }
    
    Tensor b = a.contiguous();
    
    // Check shape is preserved
    EXPECT_EQ(b.shape()[0], 2);
    EXPECT_EQ(b.shape()[1], 3);
    
    // Check data is copied correctly
    const int32_t* b_data = static_cast<const int32_t*>(b.data());
    for (size_t i = 0; i < b.numel(); ++i) {
        EXPECT_EQ(b_data[i], static_cast<int32_t>(i));
    }
}

TEST(TensorTest, ContiguousInt64) {
    Tensor a({2, 3}, DType::Int64, Device::CPU);
    
    // Initialize with some values
    int64_t* data = static_cast<int64_t*>(a.data());
    for (size_t i = 0; i < a.numel(); ++i) {
        data[i] = static_cast<int64_t>(i);
    }
    
    Tensor b = a.contiguous();
    
    // Check shape is preserved
    EXPECT_EQ(b.shape()[0], 2);
    EXPECT_EQ(b.shape()[1], 3);
    
    // Check data is copied correctly
    const int64_t* b_data = static_cast<const int64_t*>(b.data());
    for (size_t i = 0; i < b.numel(); ++i) {
        EXPECT_EQ(b_data[i], static_cast<int64_t>(i));
    }
}

TEST(TensorTest, Contiguous1D) {
    Tensor a({5}, DType::Float32, Device::CPU);
    
    float* data = static_cast<float*>(a.data());
    for (size_t i = 0; i < a.numel(); ++i) {
        data[i] = static_cast<float>(i);
    }
    
    Tensor b = a.contiguous();
    
    EXPECT_EQ(b.shape()[0], 5);
    EXPECT_EQ(b.numel(), 5);
    
    const float* b_data = static_cast<const float*>(b.data());
    for (size_t i = 0; i < b.numel(); ++i) {
        EXPECT_FLOAT_EQ(b_data[i], static_cast<float>(i));
    }
}

TEST(TensorTest, Contiguous3D) {
    Tensor a({2, 3, 4}, DType::Float32, Device::CPU);
    
    float* data = static_cast<float*>(a.data());
    for (size_t i = 0; i < a.numel(); ++i) {
        data[i] = static_cast<float>(i);
    }
    
    Tensor b = a.contiguous();
    
    EXPECT_EQ(b.shape()[0], 2);
    EXPECT_EQ(b.shape()[1], 3);
    EXPECT_EQ(b.shape()[2], 4);
    EXPECT_EQ(b.numel(), 24);
    
    const float* b_data = static_cast<const float*>(b.data());
    for (size_t i = 0; i < b.numel(); ++i) {
        EXPECT_FLOAT_EQ(b_data[i], static_cast<float>(i));
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
