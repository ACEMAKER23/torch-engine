#include "grad_fn.h"
#include "../tensor/tensor.h"
#include <vector>
#include <cmath>

// Helper: reduce gradient by summing over broadcast dimensions
// If input_shape differs from grad_shape, sum over dimensions where input had size 1
static Tensor reduce_gradient(const Tensor& grad, const std::vector<int64_t>& input_shape) {
    if (grad.shape() == input_shape) {
        return grad;  // No broadcasting, no reduction needed
    }

    Tensor result = grad;
    int64_t grad_rank = grad.shape().size();
    int64_t input_rank = input_shape.size();

    // Work from right to left (right-aligned shapes)
    // We need to track the current dimension index in result since sum reduces rank
    int64_t current_dim = grad_rank - 1;

    for (int64_t i = 0; i < grad_rank; ++i) {
        int64_t grad_dim_idx = grad_rank - 1 - i;
        int64_t input_dim_idx = input_rank - 1 - i;

        if (input_dim_idx < 0) {
            // Input has fewer dimensions - this entire dim was added by broadcasting
            // Sum over this dimension
            result = result.sum(current_dim);
            current_dim--;  // Since we reduced rank, decrement current_dim
        } else if (input_shape[input_dim_idx] == 1 && grad.shape()[grad_dim_idx] > 1) {
            // This dimension was broadcast (size 1 -> larger)
            // Sum over this dimension
            result = result.sum(current_dim);
            current_dim--;  // Since we reduced rank, decrement current_dim
        }
    }

    return result;
}

// Template helper for unary element-wise backward operations
// Eliminates unnecessary cloning by computing gradient directly
template<typename T>
static Tensor unary_backward_elementwise(const Tensor& input, const Tensor& upstream_grad, 
                                          std::function<T(T, T)> grad_fn) {
    Tensor result(input.shape(), input.dtype(), input.device());
    
    for (size_t i = 0; i < input.numel(); ++i) {
        T input_val = input.at<T>(i);
        T upstream_val = upstream_grad.at<T>(i);
        result.at<T>(i) = grad_fn(input_val, upstream_val);
    }
    
    return result;
}

// Template dispatch for unary backward operations
template<typename T>
static Tensor unary_backward_dispatch(const Tensor& input, const Tensor& upstream_grad, 
                                      std::function<T(T, T)> grad_fn) {
    return unary_backward_elementwise<T>(input, upstream_grad, grad_fn);
}

static Tensor unary_backward(const Tensor& input, const Tensor& upstream_grad,
                            std::function<float(float, float)> float_grad_fn) {
    switch (input.dtype()) {
        case DType::Float32:
            return unary_backward_dispatch<float>(input, upstream_grad, float_grad_fn);
        default:
            throw std::runtime_error("Unsupported dtype for unary backward");
    }
}

std::vector<Tensor> AddBackward::backward(const Tensor& pathDownGrad) {
    auto grad_a = reduce_gradient(pathDownGrad, inputs[0].shape());
    auto grad_b = reduce_gradient(pathDownGrad, inputs[1].shape());
    grad_a.setRequiresGrad(false);
    grad_b.setRequiresGrad(false);
    std::vector<Tensor> result;
    result.push_back(grad_a);
    result.push_back(grad_b);
    return result;
}

std::vector<Tensor> SubBackward::backward(const Tensor& pathDownGrad) {
    auto grad_a = reduce_gradient(pathDownGrad, inputs[0].shape());
    auto grad_b = reduce_gradient(-pathDownGrad, inputs[1].shape());
    grad_a.setRequiresGrad(false);
    grad_b.setRequiresGrad(false);
    std::vector<Tensor> result;
    result.push_back(grad_a);
    result.push_back(grad_b);
    return result;
}

std::vector<Tensor> MulBackward::backward(const Tensor& pathDownGrad) {
    auto grad_a = reduce_gradient(pathDownGrad * inputs[1], inputs[0].shape());
    auto grad_b = reduce_gradient(pathDownGrad * inputs[0], inputs[1].shape());
    grad_a.setRequiresGrad(false);
    grad_b.setRequiresGrad(false);
    std::vector<Tensor> result;
    result.push_back(grad_a);
    result.push_back(grad_b);
    return result;
}

std::vector<Tensor> MatMulBackward::backward(const Tensor& pathDownGrad) {
    auto grad_a = reduce_gradient(pathDownGrad.matmul(inputs[1].transpose_view(inputs[1].shape().size()-2,inputs[1].shape().size()-1)), inputs[0].shape());
    auto grad_b = reduce_gradient(inputs[0].transpose_view(inputs[0].shape().size()-2,inputs[0].shape().size()-1).matmul(pathDownGrad), inputs[1].shape());
    grad_a.setRequiresGrad(false);
    grad_b.setRequiresGrad(false);
    std::vector<Tensor> result;
    result.push_back(grad_a);
    result.push_back(grad_b);
    return result;
}


std::vector<Tensor> DivBackward::backward(const Tensor& pathDownGrad) {
    // For c = a/b: dc/da = 1/b, dc/db = -a/b^2
    auto grad_a = reduce_gradient(pathDownGrad / inputs[1], inputs[0].shape());
    auto grad_b = reduce_gradient(-pathDownGrad * inputs[0] / (inputs[1] * inputs[1]), inputs[1].shape());
    grad_a.setRequiresGrad(false);
    grad_b.setRequiresGrad(false);
    std::vector<Tensor> result;
    result.push_back(grad_a);
    result.push_back(grad_b);
    return result;
}

std::vector<Tensor> ReluBackward::backward(const Tensor& pathDownGrad) {
    // ReLU gradient: 1 if input > 0, else 0
    // Use template helper to avoid cloning and use compile-time dispatch
    auto grad = unary_backward(inputs[0], pathDownGrad, 
        [](float input_val, float upstream_val) -> float {
            return input_val > 0.0f ? upstream_val : 0.0f;
        });
    grad.setRequiresGrad(false);
    return {grad};
}

std::vector<Tensor> GeluBackward::backward(const Tensor& pathDownGrad) {
    // GELU gradient: 0.5 * (1 + tanh(z)) * (1 + x * (1 - tanh(z)^2) * (sqrt(2/pi) + 0.044715 * 3 * x^2))
    // where z = sqrt(2/pi) * (x + 0.044715 * x^3)
    // Use template helper to avoid cloning and use compile-time dispatch
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    auto grad = unary_backward(inputs[0], pathDownGrad,
        [sqrt_2_over_pi, coeff](float input_val, float upstream_val) -> float {
            float x_cubed = input_val * input_val * input_val;
            float z = sqrt_2_over_pi * (input_val + coeff * x_cubed);
            float tanh_z = std::tanh(z);
            float sech_sq = 1.0f - tanh_z * tanh_z;
            float inner = sqrt_2_over_pi * (1.0f + 3.0f * coeff * input_val * input_val);
            float gelu_grad = 0.5f * (1.0f + tanh_z) * (1.0f + input_val * sech_sq * inner);
            return upstream_val * gelu_grad;
        });
    grad.setRequiresGrad(false);
    return {grad};
}

std::vector<Tensor> SigmoidBackward::backward(const Tensor& pathDownGrad) {
    // Sigmoid gradient: sigmoid(x) * (1 - sigmoid(x))
    // Use template helper to avoid cloning and use compile-time dispatch
    auto grad = unary_backward(inputs[0], pathDownGrad,
        [](float input_val, float upstream_val) -> float {
            float sig = 1.0f / (1.0f + std::exp(-input_val));
            return upstream_val * sig * (1.0f - sig);
        });
    grad.setRequiresGrad(false);
    return {grad};
}

