#include "L1Loss.h"
#include "../core/grad_fn.h"
#include <cmath>

Tensor L1Loss::forward(const Tensor& predictions, const Tensor& targets) {
    // Mean Absolute Error: loss = mean(|predictions - targets|)

    Tensor diff = predictions - targets;
    float loss_value = 0.0f;
    for (size_t i = 0; i < diff.numel(); ++i) {
        loss_value += std::abs(diff.at<float>(i));
    }
    loss_value /= static_cast<float>(diff.numel());

    // Create loss tensor with gradient tracking
    Tensor loss({1}, DType::Float32, predictions.device());
    loss.at<float>(0) = loss_value;

    // Attach gradient function for backpropagation
    auto grad_fn = std::make_shared<L1Backward>();
    grad_fn->inputs.push_back(predictions);
    grad_fn->inputs.push_back(targets);
    loss.setGradFn(grad_fn);

    return loss;
}
