#include "BCELoss.h"
#include "../core/grad_fn.h"
#include <cmath>

Tensor BCELoss::forward(const Tensor& predictions, const Tensor& targets) {
    // Binary Cross Entropy: loss = -[y*log(p) + (1-y)*log(1-p)]
    // predictions are probabilities (0-1), targets are 0 or 1

    float loss_value = 0.0f;
    for (size_t i = 0; i < predictions.numel(); ++i) {
        float p = predictions.at<float>(i);
        float t = targets.at<float>(i);
        // Clamp to avoid log(0)
        p = std::max(1e-7f, std::min(1.0f - 1e-7f, p));
        loss_value -= t * std::log(p) + (1.0f - t) * std::log(1.0f - p);
    }
    loss_value /= static_cast<float>(predictions.numel());

    // Create loss tensor with gradient tracking
    Tensor loss({1}, DType::Float32, predictions.device());
    loss.at<float>(0) = loss_value;

    // Attach gradient function for backpropagation
    auto grad_fn = std::make_shared<BCEBackward>();
    grad_fn->inputs.push_back(predictions);
    grad_fn->inputs.push_back(targets);
    loss.setGradFn(grad_fn);

    return loss;
}
