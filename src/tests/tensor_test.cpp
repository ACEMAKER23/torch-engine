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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
