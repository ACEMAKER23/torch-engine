#include "MSELoss.h"
#include "../core/grad_fn.h"

Tensor MSELoss::forward(const Tensor& predictions, const Tensor& targets) {
    // Mean Squared Error: loss = mean((predictions - targets)^2)
    
    Tensor diff = predictions - targets;
    Tensor squared = diff * diff;
    float loss_value = squared.mean<float>();
    
    // Create loss tensor with gradient tracking
    Tensor loss({1}, DType::Float32, predictions.device());
    loss.at<float>(0) = loss_value;
    
    // Attach gradient function for backpropagation
    auto grad_fn = std::make_shared<MSEBackward>();
    grad_fn->inputs.push_back(predictions);
    grad_fn->inputs.push_back(targets);
    loss.setGradFn(grad_fn);
    
    return loss;
}
