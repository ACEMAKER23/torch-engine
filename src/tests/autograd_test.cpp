#include <gtest/gtest.h>
#include "../tensor/tensor.h"
#include "../core/grad_fn.h"

class AutogradTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

// Test graph node creation
TEST_F(AutogradTest, GraphNodeCreationNoGrad) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    
    Tensor c = a + b;
    
    // No graph should be created when requires_grad is false
    EXPECT_FALSE(c.requiredGrad());
    EXPECT_EQ(c.gradFn(), nullptr);
}

TEST_F(AutogradTest, GraphNodeCreationWithGrad) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);
    
    Tensor c = a + b;
    
    // Graph should be created when requires_grad is true
    EXPECT_TRUE(c.requiredGrad());
    EXPECT_NE(c.gradFn(), nullptr);
}

TEST_F(AutogradTest, GraphNodeCreationPartialGrad) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    // b does not require grad
    
    Tensor c = a + b;
    
    // Graph should still be created if any input requires grad
    EXPECT_TRUE(c.requiredGrad());
    EXPECT_NE(c.gradFn(), nullptr);
}

// Test AddBackward directly
TEST_F(AutogradTest, AddBackwardBasic) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);
    
    // Initialize data
    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 1.0f;
        b.at<float>(i) = 2.0f;
    }
    
    Tensor c = a + b;
    auto fn = std::dynamic_pointer_cast<AddBackward>(c.gradFn());
    ASSERT_NE(fn, nullptr);
    fn->inputs = {a, b};
    
    // Create gradient output (all ones)
    Tensor grad_output({2, 2}, DType::Float32, Device::CPU);
    for (size_t i = 0; i < grad_output.numel(); ++i) {
        grad_output.at<float>(i) = 1.0f;
    }
    
    auto grads = fn->backward(grad_output);
    
    // For addition: grad_a = grad_output, grad_b = grad_output
    ASSERT_EQ(grads.size(), 2);
    for (size_t i = 0; i < grads[0].numel(); ++i) {
        EXPECT_FLOAT_EQ(grads[0].at<float>(i), 1.0f);
        EXPECT_FLOAT_EQ(grads[1].at<float>(i), 1.0f);
    }
}

// Test SubBackward directly
TEST_F(AutogradTest, SubBackwardBasic) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);
    
    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 1.0f;
        b.at<float>(i) = 2.0f;
    }
    
    Tensor c = a - b;
    auto fn = std::dynamic_pointer_cast<SubBackward>(c.gradFn());
    ASSERT_NE(fn, nullptr);
    fn->inputs = {a, b};
    
    Tensor grad_output({2, 2}, DType::Float32, Device::CPU);
    for (size_t i = 0; i < grad_output.numel(); ++i) {
        grad_output.at<float>(i) = 1.0f;
    }
    
    auto grads = fn->backward(grad_output);
    
    // For subtraction: grad_a = grad_output, grad_b = -grad_output
    ASSERT_EQ(grads.size(), 2);
    for (size_t i = 0; i < grads[0].numel(); ++i) {
        EXPECT_FLOAT_EQ(grads[0].at<float>(i), 1.0f);
        EXPECT_FLOAT_EQ(grads[1].at<float>(i), -1.0f);
    }
}

// Test MulBackward directly
TEST_F(AutogradTest, MulBackwardBasic) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);
    
    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 2.0f;
        b.at<float>(i) = 3.0f;
    }
    
    Tensor c = a * b;
    auto fn = std::dynamic_pointer_cast<MulBackward>(c.gradFn());
    ASSERT_NE(fn, nullptr);
    fn->inputs = {a, b};
    
    Tensor grad_output({2, 2}, DType::Float32, Device::CPU);
    for (size_t i = 0; i < grad_output.numel(); ++i) {
        grad_output.at<float>(i) = 1.0f;
    }
    
    auto grads = fn->backward(grad_output);
    
    // For multiplication: grad_a = grad_output * b, grad_b = grad_output * a
    ASSERT_EQ(grads.size(), 2);
    for (size_t i = 0; i < grads[0].numel(); ++i) {
        EXPECT_FLOAT_EQ(grads[0].at<float>(i), 3.0f);  // 1.0 * 3.0
        EXPECT_FLOAT_EQ(grads[1].at<float>(i), 2.0f);  // 1.0 * 2.0
    }
}

// Test DivBackward directly
TEST_F(AutogradTest, DivBackwardBasic) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);
    
    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 6.0f;
        b.at<float>(i) = 2.0f;
    }
    
    Tensor c = a / b;
    auto fn = std::dynamic_pointer_cast<DivBackward>(c.gradFn());
    ASSERT_NE(fn, nullptr);
    fn->inputs = {a, b};
    
    Tensor grad_output({2, 2}, DType::Float32, Device::CPU);
    for (size_t i = 0; i < grad_output.numel(); ++i) {
        grad_output.at<float>(i) = 1.0f;
    }
    
    auto grads = fn->backward(grad_output);
    
    // For division: grad_a = grad_output / b, grad_b = -grad_output * a / (b * b)
    ASSERT_EQ(grads.size(), 2);
    for (size_t i = 0; i < grads[0].numel(); ++i) {
        EXPECT_FLOAT_EQ(grads[0].at<float>(i), 0.5f);  // 1.0 / 2.0
        EXPECT_FLOAT_EQ(grads[1].at<float>(i), -1.5f);  // -1.0 * 6.0 / (2.0 * 2.0)
    }
}

// Test MatMulBackward directly
TEST_F(AutogradTest, MatMulBackwardBasic) {
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({3, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);
    
    // Initialize with simple values
    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 1.0f;
    }
    for (size_t i = 0; i < b.numel(); ++i) {
        b.at<float>(i) = 1.0f;
    }
    
    Tensor c = a.matmul(b);
    auto fn = std::dynamic_pointer_cast<MatMulBackward>(c.gradFn());
    ASSERT_NE(fn, nullptr);
    fn->inputs = {a, b};
    
    Tensor grad_output({2, 2}, DType::Float32, Device::CPU);
    for (size_t i = 0; i < grad_output.numel(); ++i) {
        grad_output.at<float>(i) = 1.0f;
    }
    
    auto grads = fn->backward(grad_output);
    
    // For matmul: grad_a = grad_output @ b^T, grad_b = a^T @ grad_output
    ASSERT_EQ(grads.size(), 2);
    EXPECT_EQ(grads[0].shape(), std::vector<int64_t>({2, 3}));
    EXPECT_EQ(grads[1].shape(), std::vector<int64_t>({3, 2}));
    
    // With all ones, grad_a should have shape (2,3) with each row sum = 2 (sum of grad_output columns)
    // grad_b should have shape (3,2) with each column sum = 2 (sum of grad_output rows)
}

// Test broadcasting with gradients
TEST_F(AutogradTest, BroadcastingWithGradients) {
    Tensor a({3,}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 3}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);
    
    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 1.0f;
    }
    for (size_t i = 0; i < b.numel(); ++i) {
        b.at<float>(i) = 2.0f;
    }
    
    Tensor c = a + b;  // a broadcast to (2,3)
    
    EXPECT_TRUE(c.requiredGrad());
    EXPECT_NE(c.gradFn(), nullptr);
}

// Test unary minus
TEST_F(AutogradTest, UnaryMinus) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    
    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 1.0f;
    }
    
    Tensor b = -a;
    
    for (size_t i = 0; i < b.numel(); ++i) {
        EXPECT_FLOAT_EQ(b.at<float>(i), -1.0f);
    }
}

// Test all operators create graph nodes
TEST_F(AutogradTest, AllOperatorsCreateGraph) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);
    
    Tensor c_add = a + b;
    EXPECT_TRUE(c_add.requiredGrad());
    EXPECT_NE(c_add.gradFn(), nullptr);
    
    Tensor c_sub = a - b;
    EXPECT_TRUE(c_sub.requiredGrad());
    EXPECT_NE(c_sub.gradFn(), nullptr);
    
    Tensor c_mul = a * b;
    EXPECT_TRUE(c_mul.requiredGrad());
    EXPECT_NE(c_mul.gradFn(), nullptr);
    
    Tensor c_div = a / b;
    EXPECT_TRUE(c_div.requiredGrad());
    EXPECT_NE(c_div.gradFn(), nullptr);
}

// Test that gradient computation doesn't create new graph nodes
TEST_F(AutogradTest, GradientsNoGraph) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);
    
    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 1.0f;
        b.at<float>(i) = 2.0f;
    }
    
    Tensor c = a * b;
    auto fn = std::dynamic_pointer_cast<MulBackward>(c.gradFn());
    fn->inputs = {a, b};
    
    Tensor grad_output({2, 2}, DType::Float32, Device::CPU);
    for (size_t i = 0; i < grad_output.numel(); ++i) {
        grad_output.at<float>(i) = 1.0f;
    }
    
    auto grads = fn->backward(grad_output);
    
    // Gradients should not require grad
    EXPECT_FALSE(grads[0].requiredGrad());
    EXPECT_FALSE(grads[1].requiredGrad());
}

// Test with different dtypes
TEST_F(AutogradTest, DifferentDtypes) {
    Tensor a_int({2, 2}, DType::Int32, Device::CPU);
    a_int.setRequiresGrad(true);
    Tensor b_int({2, 2}, DType::Int32, Device::CPU);
    b_int.setRequiresGrad(true);
    
    Tensor c_int = a_int + b_int;
    EXPECT_TRUE(c_int.requiredGrad());
    
    Tensor a_float({2, 2}, DType::Float32, Device::CPU);
    a_float.setRequiresGrad(true);
    Tensor b_float({2, 2}, DType::Float32, Device::CPU);
    b_float.setRequiresGrad(true);
    
    Tensor c_float = a_float + b_float;
    EXPECT_TRUE(c_float.requiredGrad());
}

// Test scalar-like tensors
TEST_F(AutogradTest, ScalarTensor) {
    Tensor a({1}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({1}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);
    
    a.at<float>(0) = 2.0f;
    b.at<float>(0) = 3.0f;
    
    Tensor c = a * b;
    auto fn = std::dynamic_pointer_cast<MulBackward>(c.gradFn());
    fn->inputs = {a, b};
    
    Tensor grad_output({1}, DType::Float32, Device::CPU);
    grad_output.at<float>(0) = 1.0f;
    
    auto grads = fn->backward(grad_output);
    
    EXPECT_FLOAT_EQ(grads[0].at<float>(0), 3.0f);
    EXPECT_FLOAT_EQ(grads[1].at<float>(0), 2.0f);
}

// Test zero gradients
TEST_F(AutogradTest, ZeroGradients) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);
    
    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 1.0f;
        b.at<float>(i) = 2.0f;
    }
    
    Tensor c = a + b;
    auto fn = std::dynamic_pointer_cast<AddBackward>(c.gradFn());
    fn->inputs = {a, b};
    
    Tensor grad_output({2, 2}, DType::Float32, Device::CPU);
    for (size_t i = 0; i < grad_output.numel(); ++i) {
        grad_output.at<float>(i) = 0.0f;
    }
    
    auto grads = fn->backward(grad_output);
    
    // Zero gradient output should produce zero gradients
    for (size_t i = 0; i < grads[0].numel(); ++i) {
        EXPECT_FLOAT_EQ(grads[0].at<float>(i), 0.0f);
        EXPECT_FLOAT_EQ(grads[1].at<float>(i), 0.0f);
    }
}

// Test backward() method on simple graph
TEST_F(AutogradTest, BackwardMethodSimple) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);
    
    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 2.0f;
        b.at<float>(i) = 3.0f;
    }
    
    Tensor c = a + b;
    c.backward();
    
    // Check gradients are stored
    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);
    
    // For addition, gradients should be 1.0
    for (size_t i = 0; i < a.grad()->numel(); ++i) {
        EXPECT_FLOAT_EQ(a.grad()->at<float>(i), 1.0f);
        EXPECT_FLOAT_EQ(b.grad()->at<float>(i), 1.0f);
    }
}

// Test backward() method with multiplication
TEST_F(AutogradTest, BackwardMethodMultiplication) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);
    
    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 2.0f;
        b.at<float>(i) = 3.0f;
    }
    
    Tensor c = a * b;
    c.backward();
    
    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);
    
    // For multiplication: grad_a = grad_output * b = 1.0 * 3.0 = 3.0
    // grad_b = grad_output * a = 1.0 * 2.0 = 2.0
    for (size_t i = 0; i < a.grad()->numel(); ++i) {
        EXPECT_FLOAT_EQ(a.grad()->at<float>(i), 3.0f);
        EXPECT_FLOAT_EQ(b.grad()->at<float>(i), 2.0f);
    }
}

// Test backward() method with chained operations
TEST_F(AutogradTest, BackwardMethodChained) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);
    Tensor c({2, 2}, DType::Float32, Device::CPU);
    c.setRequiresGrad(true);
    
    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 1.0f;
        b.at<float>(i) = 2.0f;
        c.at<float>(i) = 3.0f;
    }
    
    // d = (a * b) + c
    Tensor d = (a * b) + c;
    d.backward();
    
    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);
    ASSERT_NE(c.grad(), nullptr);
    
    // For (a * b) + c:
    // grad_c = 1.0
    // grad_(a*b) = 1.0
    // grad_a = grad_(a*b) * b = 1.0 * 2.0 = 2.0
    // grad_b = grad_(a*b) * a = 1.0 * 1.0 = 1.0
    for (size_t i = 0; i < a.grad()->numel(); ++i) {
        EXPECT_FLOAT_EQ(a.grad()->at<float>(i), 2.0f);
        EXPECT_FLOAT_EQ(b.grad()->at<float>(i), 1.0f);
        EXPECT_FLOAT_EQ(c.grad()->at<float>(i), 1.0f);
    }
}

// Test backward() with single input requiring grad
TEST_F(AutogradTest, BackwardMethodPartialGrad) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    // b does not require grad
    
    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 2.0f;
        b.at<float>(i) = 3.0f;
    }
    
    Tensor c = a * b;
    c.backward();
    
    ASSERT_NE(a.grad(), nullptr);
    EXPECT_EQ(b.grad(), nullptr);  // b shouldn't have gradient
    
    for (size_t i = 0; i < a.grad()->numel(); ++i) {
        EXPECT_FLOAT_EQ(a.grad()->at<float>(i), 3.0f);
    }
}

// Test backward() with subtraction
TEST_F(AutogradTest, BackwardMethodSubtraction) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);
    
    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 5.0f;
        b.at<float>(i) = 3.0f;
    }
    
    Tensor c = a - b;
    c.backward();
    
    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);
    
    // For subtraction: grad_a = 1.0, grad_b = -1.0
    for (size_t i = 0; i < a.grad()->numel(); ++i) {
        EXPECT_FLOAT_EQ(a.grad()->at<float>(i), 1.0f);
        EXPECT_FLOAT_EQ(b.grad()->at<float>(i), -1.0f);
    }
}

// Test backward() with division
TEST_F(AutogradTest, BackwardMethodDivision) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);
    
    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 6.0f;
        b.at<float>(i) = 2.0f;
    }
    
    Tensor c = a / b;
    c.backward();
    
    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);
    
    // For division: grad_a = 1.0 / b = 0.5, grad_b = -1.0 * a / (b * b) = -1.5
    for (size_t i = 0; i < a.grad()->numel(); ++i) {
        EXPECT_FLOAT_EQ(a.grad()->at<float>(i), 0.5f);
        EXPECT_FLOAT_EQ(b.grad()->at<float>(i), -1.5f);
    }
}

// Test deeply nested chained operations
TEST_F(AutogradTest, BackwardMethodDeepNesting) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);
    Tensor c({2, 2}, DType::Float32, Device::CPU);
    c.setRequiresGrad(true);
    Tensor d({2, 2}, DType::Float32, Device::CPU);
    d.setRequiresGrad(true);

    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 1.0f;
        b.at<float>(i) = 2.0f;
        c.at<float>(i) = 3.0f;
        d.at<float>(i) = 4.0f;
    }

    // loss = ((a * b) + c) - d
    Tensor loss = ((a * b) + c) - d;
    loss.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);
    ASSERT_NE(c.grad(), nullptr);
    ASSERT_NE(d.grad(), nullptr);

    // dL/da = 1 * b = 2
    // dL/db = 1 * a = 1
    // dL/dc = 1
    // dL/dd = -1
    for (size_t i = 0; i < a.numel(); ++i) {
        EXPECT_FLOAT_EQ(a.grad()->at<float>(i), 2.0f);
        EXPECT_FLOAT_EQ(b.grad()->at<float>(i), 1.0f);
        EXPECT_FLOAT_EQ(c.grad()->at<float>(i), 1.0f);
        EXPECT_FLOAT_EQ(d.grad()->at<float>(i), -1.0f);
    }
}

// Test fan-out: one tensor used multiple times (gradient accumulation)
TEST_F(AutogradTest, BackwardMethodFanOut) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);
    Tensor c({2, 2}, DType::Float32, Device::CPU);
    c.setRequiresGrad(true);

    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 2.0f;
        b.at<float>(i) = 3.0f;
        c.at<float>(i) = 4.0f;
    }

    // loss = (a * b) + (a * c)
    // dL/da = b + c = 3 + 4 = 7
    // dL/db = a = 2
    // dL/dc = a = 2
    Tensor loss = (a * b) + (a * c);
    loss.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);
    ASSERT_NE(c.grad(), nullptr);

    for (size_t i = 0; i < a.numel(); ++i) {
        EXPECT_FLOAT_EQ(a.grad()->at<float>(i), 7.0f);
        EXPECT_FLOAT_EQ(b.grad()->at<float>(i), 2.0f);
        EXPECT_FLOAT_EQ(c.grad()->at<float>(i), 2.0f);
    }
}

// Test chained operations with broadcasting
TEST_F(AutogradTest, BackwardMethodChainedBroadcasting) {
    Tensor a({1, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);

    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 2.0f;
    }
    for (size_t i = 0; i < b.numel(); ++i) {
        b.at<float>(i) = 3.0f;
    }

    // Broadcasting: a (1x2) + b (2x2) -> (2x2)
    Tensor c = a + b;
    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);

    // For broadcasting addition: grad_a is summed over broadcast dimensions
    // a (1,2) is broadcast to (2,2), so each element of a contributes to 2 positions
    // grad_a should be [2, 2] (sum over dimension 0)
    // grad_b is just 1 (no broadcasting)
    for (size_t i = 0; i < a.grad()->numel(); ++i) {
        EXPECT_FLOAT_EQ(a.grad()->at<float>(i), 2.0f);  // sum over broadcast dim
    }
    for (size_t i = 0; i < b.grad()->numel(); ++i) {
        EXPECT_FLOAT_EQ(b.grad()->at<float>(i), 1.0f);
    }
}

// Test chained operations with matmul
TEST_F(AutogradTest, BackwardMethodChainedMatmul) {
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({3, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);
    Tensor c({2, 2}, DType::Float32, Device::CPU);
    c.setRequiresGrad(true);

    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 1.0f;
    }
    for (size_t i = 0; i < b.numel(); ++i) {
        b.at<float>(i) = 1.0f;
    }
    for (size_t i = 0; i < c.numel(); ++i) {
        c.at<float>(i) = 1.0f;
    }

    // loss = (a @ b) + c
    Tensor loss = a.matmul(b) + c;
    loss.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);
    ASSERT_NE(c.grad(), nullptr);

    // dL/dc = 1
    // dL/d(ab) = 1
    // dL/da = 1 @ b^T
    // dL/db = a^T @ 1
    // With all ones, grad_a should have shape (2,3), grad_b shape (3,2)
    EXPECT_EQ(a.grad()->shape(), std::vector<int64_t>({2, 3}));
    EXPECT_EQ(b.grad()->shape(), std::vector<int64_t>({3, 2}));
    EXPECT_EQ(c.grad()->shape(), std::vector<int64_t>({2, 2}));

    for (size_t i = 0; i < c.grad()->numel(); ++i) {
        EXPECT_FLOAT_EQ(c.grad()->at<float>(i), 1.0f);
    }
}

// Test multiple backward calls (gradient accumulation)
TEST_F(AutogradTest, BackwardMethodMultipleCalls) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);

    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 2.0f;
        b.at<float>(i) = 3.0f;
    }

    // First backward
    Tensor loss1 = a * b;
    loss1.backward();

    // Second backward (should accumulate)
    Tensor loss2 = a + b;
    loss2.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);

    // dL1/da = b = 3, dL2/da = 1 -> total = 4
    // dL1/db = a = 2, dL2/db = 1 -> total = 3
    for (size_t i = 0; i < a.numel(); ++i) {
        EXPECT_FLOAT_EQ(a.grad()->at<float>(i), 4.0f);
        EXPECT_FLOAT_EQ(b.grad()->at<float>(i), 3.0f);
    }
}

// Test zero gradients with complex chains
TEST_F(AutogradTest, BackwardMethodZeroGradientChain) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);

    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 1.0f;
        b.at<float>(i) = 2.0f;
    }

    // loss = (a * b) + (a * b) = 2 * a * b
    Tensor loss = (a * b) + (a * b);
    loss.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);

    // dL/da = 2 * b = 4
    // dL/db = 2 * a = 2
    for (size_t i = 0; i < a.numel(); ++i) {
        EXPECT_FLOAT_EQ(a.grad()->at<float>(i), 4.0f);
        EXPECT_FLOAT_EQ(b.grad()->at<float>(i), 2.0f);
    }
}

// Test partial gradients in complex chain
TEST_F(AutogradTest, BackwardMethodPartialGradientChain) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(false);  // b doesn't require grad
    Tensor c({2, 2}, DType::Float32, Device::CPU);
    c.setRequiresGrad(true);

    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 1.0f;
        b.at<float>(i) = 2.0f;
        c.at<float>(i) = 3.0f;
    }

    // loss = (a * b) + c
    Tensor loss = (a * b) + c;
    loss.backward();

    ASSERT_NE(a.grad(), nullptr);
    EXPECT_EQ(b.grad(), nullptr);  // b doesn't require grad
    ASSERT_NE(c.grad(), nullptr);

    // dL/da = b = 2
    // dL/dc = 1
    for (size_t i = 0; i < a.numel(); ++i) {
        EXPECT_FLOAT_EQ(a.grad()->at<float>(i), 2.0f);
    }
    for (size_t i = 0; i < c.numel(); ++i) {
        EXPECT_FLOAT_EQ(c.grad()->at<float>(i), 1.0f);
    }
}

// Test division in complex chain
TEST_F(AutogradTest, BackwardMethodDivisionChain) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);
    Tensor c({2, 2}, DType::Float32, Device::CPU);
    c.setRequiresGrad(true);

    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 8.0f;
        b.at<float>(i) = 2.0f;
        c.at<float>(i) = 1.0f;
    }

    // loss = (a / b) + c
    Tensor loss = (a / b) + c;
    loss.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);
    ASSERT_NE(c.grad(), nullptr);

    // dL/da = 1/b = 0.5
    // dL/db = -a/b^2 = -8/4 = -2
    // dL/dc = 1
    for (size_t i = 0; i < a.grad()->numel(); ++i) {
        EXPECT_FLOAT_EQ(a.grad()->at<float>(i), 0.5f);
        EXPECT_FLOAT_EQ(b.grad()->at<float>(i), -2.0f);
        EXPECT_FLOAT_EQ(c.grad()->at<float>(i), 1.0f);
    }
}

// Test broadcasting with subtraction
TEST_F(AutogradTest, BackwardMethodSubtractionBroadcasting) {
    Tensor a({1, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);

    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 2.0f;
    }
    for (size_t i = 0; i < b.numel(); ++i) {
        b.at<float>(i) = 3.0f;
    }

    // Broadcasting: a (1x2) - b (2x2) -> (2x2)
    Tensor c = a - b;
    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);

    // grad_a is summed over broadcast dimension
    // grad_b is -1 (negative because subtraction)
    for (size_t i = 0; i < a.grad()->numel(); ++i) {
        EXPECT_FLOAT_EQ(a.grad()->at<float>(i), 2.0f);  // sum over broadcast dim
    }
    for (size_t i = 0; i < b.grad()->numel(); ++i) {
        EXPECT_FLOAT_EQ(b.grad()->at<float>(i), -1.0f);
    }
}

// Test broadcasting with multiplication
TEST_F(AutogradTest, BackwardMethodMultiplicationBroadcasting) {
    Tensor a({1, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);

    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 2.0f;
    }
    for (size_t i = 0; i < b.numel(); ++i) {
        b.at<float>(i) = 3.0f;
    }

    // Broadcasting: a (1x2) * b (2x2) -> (2x2)
    Tensor c = a * b;
    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);

    // grad_a = sum(b) over broadcast dim = 3+3 = 6
    // grad_b = a (broadcasted) = 2
    for (size_t i = 0; i < a.grad()->numel(); ++i) {
        EXPECT_FLOAT_EQ(a.grad()->at<float>(i), 6.0f);  // sum over broadcast dim
    }
    for (size_t i = 0; i < b.grad()->numel(); ++i) {
        EXPECT_FLOAT_EQ(b.grad()->at<float>(i), 2.0f);
    }
}

// Test broadcasting with division
TEST_F(AutogradTest, BackwardMethodDivisionBroadcasting) {
    Tensor a({1, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);

    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 8.0f;
    }
    for (size_t i = 0; i < b.numel(); ++i) {
        b.at<float>(i) = 2.0f;
    }

    // Broadcasting: a (1x2) / b (2x2) -> (2x2)
    Tensor c = a / b;
    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);

    // grad_a = sum(1/b) over broadcast dim = 0.5+0.5 = 1.0
    // grad_b = -a/b^2 = -8/4 = -2
    for (size_t i = 0; i < a.grad()->numel(); ++i) {
        EXPECT_FLOAT_EQ(a.grad()->at<float>(i), 1.0f);  // sum over broadcast dim
    }
    for (size_t i = 0; i < b.grad()->numel(); ++i) {
        EXPECT_FLOAT_EQ(b.grad()->at<float>(i), -2.0f);
    }
}

// Test broadcasting with matmul (batch broadcasting)
TEST_F(AutogradTest, BackwardMethodMatMulBroadcasting) {
    Tensor a({1, 2, 3}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({1, 3, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);

    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 1.0f;
    }
    for (size_t i = 0; i < b.numel(); ++i) {
        b.at<float>(i) = 1.0f;
    }

    // Broadcasting matmul: a (1,2,3) @ b (1,3,2) -> (1,2,2)
    Tensor c = a.matmul(b);
    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);

    // Shapes should match original inputs (after reduction)
    EXPECT_EQ(a.grad()->shape(), std::vector<int64_t>({1, 2, 3}));
    EXPECT_EQ(b.grad()->shape(), std::vector<int64_t>({1, 3, 2}));
}

// Test ReLU backward
TEST_F(AutogradTest, BackwardMethodRelu) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);

    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = static_cast<float>(i) - 1.5f;  // Mix of positive and negative
    }

    Tensor c = a.relu();
    c.backward();

    ASSERT_NE(a.grad(), nullptr);

    // ReLU gradient: 1 if input > 0, else 0
    for (size_t i = 0; i < a.numel(); ++i) {
        float expected_grad = (a.at<float>(i) > 0.0f) ? 1.0f : 0.0f;
        EXPECT_FLOAT_EQ(a.grad()->at<float>(i), expected_grad);
    }
}

// Test GELU backward
TEST_F(AutogradTest, BackwardMethodGelu) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);

    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = static_cast<float>(i) * 0.1f;
    }

    Tensor c = a.gelu();
    c.backward();

    ASSERT_NE(a.grad(), nullptr);

    // GELU gradient should be computed correctly
    // For small positive values, gradient should be close to 1
    // For zero, gradient should be 0.5
    EXPECT_NEAR(a.grad()->at<float>(0), 0.5f, 0.01f);  // x=0, grad≈0.5
    EXPECT_GT(a.grad()->at<float>(1), 0.5f);  // x=0.1, grad>0.5
    EXPECT_LT(a.grad()->at<float>(1), 1.0f);  // x=0.1, grad<1.0
}

// Test Sigmoid backward
TEST_F(AutogradTest, BackwardMethodSigmoid) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);

    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = static_cast<float>(i) * 0.1f;
    }

    Tensor c = a.sigmoid();
    c.backward();

    ASSERT_NE(a.grad(), nullptr);

    // Sigmoid gradient: sigmoid(x) * (1 - sigmoid(x))
    // For x=0, sigmoid=0.5, gradient=0.25
    EXPECT_NEAR(a.grad()->at<float>(0), 0.25f, 0.01f);  // x=0
    // For small positive x, gradient < 0.25
    EXPECT_LT(a.grad()->at<float>(1), 0.25f);  // x=0.1
}

// Test activation in chain
TEST_F(AutogradTest, BackwardMethodActivationChain) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);

    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = static_cast<float>(i) * 0.1f;
        b.at<float>(i) = 1.0f;
    }

    Tensor c = (a * b).relu();
    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);

    // Gradient should flow through ReLU
    // For a*b > 0, gradient is b for a and a for b
    // For a*b <= 0, gradient is 0
    EXPECT_GT(a.grad()->at<float>(1), 0.0f);  // a*b > 0
    EXPECT_FLOAT_EQ(b.grad()->at<float>(0), 0.0f);  // a*b = 0
}

// Test edge case: 1D tensor broadcasting with autograd
TEST_F(AutogradTest, EdgeCase1DBroadcasting) {
    Tensor a({2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);

    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 2.0f;
    }
    for (size_t i = 0; i < b.numel(); ++i) {
        b.at<float>(i) = 3.0f;
    }

    Tensor c = a + b;
    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);

    // grad_a should be summed over broadcast dimension (2 positions per element)
    for (size_t i = 0; i < a.grad()->numel(); ++i) {
        EXPECT_FLOAT_EQ(a.grad()->at<float>(i), 2.0f);  // sum over dim 0
    }
    for (size_t i = 0; i < b.grad()->numel(); ++i) {
        EXPECT_FLOAT_EQ(b.grad()->at<float>(i), 1.0f);
    }
}

// Test edge case: zero-sized tensor gradient
TEST_F(AutogradTest, EdgeCaseZeroGradient) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({2, 2}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);

    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 0.0f;
        b.at<float>(i) = 1.0f;
    }

    Tensor c = a * b;
    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);

    // When a is zero, grad_a should be 0 (b), grad_b should be 0 (a)
    for (size_t i = 0; i < a.grad()->numel(); ++i) {
        EXPECT_FLOAT_EQ(a.grad()->at<float>(i), 1.0f);
        EXPECT_FLOAT_EQ(b.grad()->at<float>(i), 0.0f);
    }
}

// Test edge case: large tensor with autograd
TEST_F(AutogradTest, EdgeCaseLargeTensor) {
    Tensor a({100, 100}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({100, 100}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);

    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = 1.0f;
        b.at<float>(i) = 2.0f;
    }

    Tensor c = a * b;
    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);

    // Check a few random positions
    EXPECT_FLOAT_EQ(a.grad()->at<float>(0), 2.0f);
    EXPECT_FLOAT_EQ(b.grad()->at<float>(0), 1.0f);
    EXPECT_FLOAT_EQ(a.grad()->at<float>(5000), 2.0f);
    EXPECT_FLOAT_EQ(b.grad()->at<float>(5000), 1.0f);
}

// Test edge case: single element tensor
TEST_F(AutogradTest, EdgeCaseSingleElement) {
    Tensor a({1}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);
    Tensor b({1}, DType::Float32, Device::CPU);
    b.setRequiresGrad(true);

    a.at<float>(0) = 3.0f;
    b.at<float>(0) = 4.0f;

    Tensor c = a * b;
    c.backward();

    ASSERT_NE(a.grad(), nullptr);
    ASSERT_NE(b.grad(), nullptr);

    EXPECT_FLOAT_EQ(a.grad()->at<float>(0), 4.0f);
    EXPECT_FLOAT_EQ(b.grad()->at<float>(0), 3.0f);
}

// Test edge case: negative values with ReLU
TEST_F(AutogradTest, EdgeCaseReluNegative) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);

    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = -1.0f;  // All negative
    }

    Tensor c = a.relu();
    c.backward();

    ASSERT_NE(a.grad(), nullptr);

    // All gradients should be 0 (all inputs were negative)
    for (size_t i = 0; i < a.grad()->numel(); ++i) {
        EXPECT_FLOAT_EQ(a.grad()->at<float>(i), 0.0f);
    }
}

// Test edge case: very small values with sigmoid
TEST_F(AutogradTest, EdgeCaseSigmoidSmallValues) {
    Tensor a({2, 2}, DType::Float32, Device::CPU);
    a.setRequiresGrad(true);

    for (size_t i = 0; i < a.numel(); ++i) {
        a.at<float>(i) = -10.0f;  // Very small
    }

    Tensor c = a.sigmoid();
    c.backward();

    ASSERT_NE(a.grad(), nullptr);

    // For very small x, sigmoid(x) ≈ 0, gradient ≈ 0
    for (size_t i = 0; i < a.grad()->numel(); ++i) {
        EXPECT_LT(a.grad()->at<float>(i), 0.001f);
    }
}
