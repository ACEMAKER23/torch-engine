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

Tensor crossEntropyLoss::forward_batched(const Tensor& predictions, const Tensor& targets) {
    auto logShape = predictions.shape();
    if (logShape.size() != 3) {
        throw std::runtime_error("forward_batched expects 3D logits [B, T, V]");
    }
    auto tgtShape = targets.shape();
    if (tgtShape.size() != 2) {
        throw std::runtime_error("forward_batched expects 2D targets [B, T]");
    }

    int64_t B = logShape[0];
    int64_t T = logShape[1];
    int64_t V = logShape[2];
    int64_t N = B * T;

    const float* logits_ptr = static_cast<const float*>(predictions.data());
    const int64_t* targets_ptr = static_cast<const int64_t*>(targets.data());
    const std::vector<int64_t>& ls = predictions.strides();
    const std::vector<int64_t>& ts = targets.strides();

    double total_loss = 0.0;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            const float* row = logits_ptr + b * ls[0] + t * ls[1];
            int64_t target_class = targets_ptr[b * ts[0] + t * ts[1]];

            float max_logit = row[0];
            for (int64_t v = 1; v < V; ++v) {
                max_logit = std::max(max_logit, row[v * ls[2]]);
            }

            float sum_exp = 0.0f;
            for (int64_t v = 0; v < V; ++v) {
                sum_exp += std::exp(row[v * ls[2]] - max_logit);
            }
            float log_sum_exp = max_logit + std::log(sum_exp);
            float logit_target = row[target_class * ls[2]];

            total_loss += (-logit_target + log_sum_exp);
        }
    }

    Tensor loss({1}, DType::Float32, predictions.device());
    loss.at<float>(0) = static_cast<float>(total_loss / static_cast<double>(N));

    auto grad_fn = std::make_shared<CrossEntropyBatchedBackward>();
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
