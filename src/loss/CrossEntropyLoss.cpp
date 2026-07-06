#include "CrossEntropyLoss.h"
#include "../core/grad_fn.h"
#include <cmath>

Tensor crossEntropyLoss::forward(const Tensor& predictions, const Tensor& targets) {
    // PyTorch-style: predictions are raw logits, targets are class indices (int64)
    // Loss = -log(softmax(logits)[target]) = -logits[target] + log(sum(exp(logits)))
    // This is numerically stable

    int64_t target_class = targets.at<int64_t>(0);
    float logit_target = predictions.at<float>(target_class);

    // Compute log-sum-exp for numerical stability
    // log(sum(exp(logits))) = max_logit + log(sum(exp(logits - max_logit)))
    float max_logit = predictions.at<float>(0);
    for (size_t i = 1; i < predictions.numel(); ++i) {
        max_logit = std::max(max_logit, predictions.at<float>(i));
    }

    float log_sum_exp = 0.0f;
    for (size_t i = 0; i < predictions.numel(); ++i) {
        log_sum_exp += std::exp(predictions.at<float>(i) - max_logit);
    }
    log_sum_exp = max_logit + std::log(log_sum_exp);

    // Cross-entropy loss
    float loss_value = -logit_target + log_sum_exp;

    // Create loss tensor with gradient tracking
    Tensor loss({1}, DType::Float32, predictions.device());
    loss.at<float>(0) = loss_value;

    // Attach gradient function for backpropagation
    auto grad_fn = std::make_shared<CrossEntropyBackward>();
    grad_fn->inputs.push_back(predictions);
    grad_fn->inputs.push_back(targets);
    loss.setGradFn(grad_fn);

    return loss;
}

Tensor crossEntropyLoss::forward_with_probs(const Tensor& predictions, const Tensor& targets) {
    // Probability-based: predictions are softmax probabilities, targets are class indices (int64)
    // Loss = -log(probabilities[target])                                                                                                                                                                   

    int64_t target_class = targets.at<int64_t>(0);
    float probability = predictions.at<float>(target_class);
    float loss_value = -std::log(probability);

    // Create loss tensor with gradient tracking
    Tensor loss({1}, DType::Float32, predictions.device());
    loss.at<float>(0) = loss_value;

    // Attach gradient function for backpropagation
    auto grad_fn = std::make_shared<CrossEntropyWithProbsBackward>();
    grad_fn->inputs.push_back(predictions);
    grad_fn->inputs.push_back(targets);
    loss.setGradFn(grad_fn);

    return loss;
}
