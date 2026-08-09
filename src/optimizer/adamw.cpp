#include "adamw.h"
#include "../tensor/tensor.h"

adamw::adamw(const std::vector<Tensor>& parameters, float beta1, float beta2, float epsilon, float weight_decay)
    : optimizer(parameters), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), weight_decay_(weight_decay), step_(0) {
    
    // Initialize m and v to zeros for each parameter
    for (const Tensor& param : processing_) {
        Tensor m_zeros(param.shape(), param.dtype(), param.device());
        m_zeros.fill_<float>(0.0f);
        m_.push_back(m_zeros);
        
        Tensor v_zeros(param.shape(), param.dtype(), param.device());
        v_zeros.fill_<float>(0.0f);
        v_.push_back(v_zeros);
    }
}

void adamw::step(float learningRate) {
    step_++;

    float bc1 = 1.0f - std::pow(beta1_, step_);
    float bc2 = 1.0f - std::pow(beta2_, step_);

    for (size_t i = 0; i < processing_.size(); ++i) {
        Tensor& param = processing_[i];
        Tensor& m     = m_[i];
        Tensor& v     = v_[i];

        if (!param.requiredGrad() || !param.grad()) continue;

        size_t N = param.numel();
        auto*       p_data = static_cast<float*>(param.data());
        auto*       m_data = static_cast<float*>(m.data());
        auto*       v_data = static_cast<float*>(v.data());
        const auto* g_data = static_cast<const float*>(param.grad()->data());

        for (size_t j = 0; j < N; ++j) {
            float g = g_data[j];

            // Decoupled weight decay
            p_data[j] *= (1.0f - weight_decay_);

            // First and second moment estimates
            m_data[j] = beta1_ * m_data[j] + (1.0f - beta1_) * g;
            v_data[j] = beta2_ * v_data[j] + (1.0f - beta2_) * g * g;

            // Bias-corrected moments
            float m_hat = m_data[j] / bc1;
            float v_hat = v_data[j] / bc2;

            // In-place parameter update (updates model weights directly)
            p_data[j] -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
}
