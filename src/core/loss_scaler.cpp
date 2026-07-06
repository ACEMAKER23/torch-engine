#include "loss_scaler.h"
#include <cmath>

LossScaler::LossScaler(float scale_factor, float max_scale, float growth_factor, float backoff_factor, size_t growth_interval)
    : initial_scale_(scale_factor)
    , current_scale_(scale_factor)
    , max_scale_(max_scale)
    , growth_factor_(growth_factor)
    , backoff_factor_(backoff_factor)
    , growth_interval_(growth_interval)
    , steps_since_last_adjustment_(0) {
}

Tensor LossScaler::scale(const Tensor& loss) {
    // Scale the loss by current scale factor
    Tensor scaled_loss = loss.clone();
    for (size_t i = 0; i < scaled_loss.numel(); ++i) {
        scaled_loss.at<float>(i) *= current_scale_;
    }
    return scaled_loss;
}

Tensor LossScaler::unscale(const Tensor& scaled_grad) {
    // Unscale the gradient by dividing by current scale factor
    Tensor grad = scaled_grad.clone();
    for (size_t i = 0; i < grad.numel(); ++i) {
        grad.at<float>(i) /= current_scale_;
    }
    return grad;
}

bool LossScaler::has_inf_or_nan(const Tensor& tensor) {
    for (size_t i = 0; i < tensor.numel(); ++i) {
        float val = tensor.at<float>(i);
        if (std::isnan(val) || std::isinf(val)) {
            return true;
        }
    }
    return false;
}

bool LossScaler::check_and_adjust_scale(const Tensor& grad) {
    if (has_inf_or_nan(grad)) {
        // Gradient has inf/nan, back off scale
        current_scale_ *= backoff_factor_;
        steps_since_last_adjustment_ = 0;
        return false; // Indicate that gradient was invalid
    }
    
    // Gradient is valid, consider growing scale
    steps_since_last_adjustment_++;
    if (steps_since_last_adjustment_ >= growth_interval_) {
        current_scale_ = std::min(current_scale_ * growth_factor_, max_scale_);
        steps_since_last_adjustment_ = 0;
    }
    
    return true; // Indicate that gradient was valid
}

void LossScaler::reset() {
    current_scale_ = initial_scale_;
    steps_since_last_adjustment_ = 0;
}
