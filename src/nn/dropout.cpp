#include "dropout.h"

Dropout::Dropout(float p) : p_(p), training_(true), gen_(std::random_device{}()) {
    if (p < 0.0f || p >= 1.0f) {
        throw std::runtime_error("Dropout probability p must be in [0, 1)");
    }
}

Tensor Dropout::forward(const Tensor& input) {
    Tensor result = input.clone();

    if (!training_) {
        return result; // no scaling in inference
    }

    float keep_prob = 1.0f - p_;
    float scale = 1.0f / keep_prob;

    // During training, randomly drop units
    std::bernoulli_distribution dist(1.0f - p_);
    
    if (input.dtype() == DType::Float32) {
        auto* data = static_cast<float*>(result.data());
        for (size_t i = 0; i < result.numel(); ++i) {
            if (dist(gen_)) {
                data[i] /= (1.0f - p_);
            } else {
                data[i] = 0.0f;
            }
        }
    }

    return result;
}

void Dropout::set_training(bool training) {
    training_ = training;
}
