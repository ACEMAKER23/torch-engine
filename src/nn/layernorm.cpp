#include "layernorm.h"
#include <cmath>

LayerNorm::LayerNorm(int64_t normalized_shape, DType dtype, float eps)
    : normalized_shape_(normalized_shape), eps_(eps),
      weight_({normalized_shape}, dtype, Device::CPU),
      bias_({normalized_shape}, dtype, Device::CPU) {
    init_parameters();
    weight_.setRequiresGrad(true);
    bias_.setRequiresGrad(true);
}

Tensor LayerNorm::forward(const Tensor& input) {
    // input: [batch_size, ..., normalized_shape]
    // weight: [normalized_shape]
    // bias: [normalized_shape]
    // output: same shape as input
    
    std::vector<int64_t> input_shape = input.shape();
    int64_t last_dim = input_shape.back();
    
    if (last_dim != normalized_shape_) {
        throw std::runtime_error("Last dimension of input must match normalized_shape");
    }
    
    Tensor result(input_shape, input.dtype(), input.device());
    
    // Compute mean and variance along the last dimension
    size_t batch_size = input.numel() / last_dim;
    
    for (size_t i = 0; i < batch_size; ++i) {
        // Compute mean
        float mean = 0.0f;
        for (int64_t j = 0; j < last_dim; ++j) {
            if (input.dtype() == DType::Float32) {
                mean += input.at<float>(i * last_dim + j);
            }
        }
        mean /= last_dim;
        
        // Compute variance
        float variance = 0.0f;
        for (int64_t j = 0; j < last_dim; ++j) {
            if (input.dtype() == DType::Float32) {
                float diff = input.at<float>(i * last_dim + j) - mean;
                variance += diff * diff;
            }
        }
        variance /= last_dim;
        
        // Apply layer normalization
        float std_dev = std::sqrt(variance + eps_);
        for (int64_t j = 0; j < last_dim; ++j) {
            if (input.dtype() == DType::Float32) {
                float x = input.at<float>(i * last_dim + j);
                float normalized = (x - mean) / std_dev;
                result.at<float>(i * last_dim + j) = normalized * weight_.at<float>(j) + bias_.at<float>(j);
            }
        }
    }
    
    return result;
}

void LayerNorm::init_parameters() {
    if (normalized_shape_ == 0) {
        throw std::runtime_error("normalized_shape cannot be zero");
    }
    
    if (weight_.dtype() != DType::Float32) {
        throw std::runtime_error("Only Float32 dtype is supported for parameter initialization");
    }

    // Initialize weight to 1.0, bias to 0.0
    auto* weight_data = static_cast<float*>(weight_.data());
    auto* bias_data = static_cast<float*>(bias_.data());
    
    for (size_t i = 0; i < weight_.numel(); ++i) {
        weight_data[i] = 1.0f;
    }
    
    for (size_t i = 0; i < bias_.numel(); ++i) {
        bias_data[i] = 0.0f;
    }
}

std::vector<Tensor> LayerNorm::parameters() {
    return {weight_, bias_};
}

void LayerNorm::zero_grad() {
    // Gradients are cleared by the tensor implementation during backward
}
