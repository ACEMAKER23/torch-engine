#include "adam.h"
#include "../tensor/tensor.h"

adam::adam(const std::vector<Tensor>& parameters, float beta1, float beta2, float epsilon)
    : optimizer(parameters), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), step_(0) {
    
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

void adam::step(float learningRate) {
    step_++;
    
    // Create scalar tensors for hyperparameters
    Tensor beta1_tensor({1}, DType::Float32, Device::CPU);
    beta1_tensor.at<float>(0) = beta1_;
    
    Tensor beta2_tensor({1}, DType::Float32, Device::CPU);
    beta2_tensor.at<float>(0) = beta2_;
    
    Tensor one_tensor({1}, DType::Float32, Device::CPU);
    one_tensor.at<float>(0) = 1.0f;
    
    Tensor lr_tensor({1}, DType::Float32, Device::CPU);
    lr_tensor.at<float>(0) = learningRate;
    
    Tensor epsilon_tensor({1}, DType::Float32, Device::CPU);
    epsilon_tensor.at<float>(0) = epsilon_;
    
    // Bias correction factors
    float bias_correction1 = 1.0f - std::pow(beta1_, step_);
    float bias_correction2 = 1.0f - std::pow(beta2_, step_);
    
    Tensor bc1_tensor({1}, DType::Float32, Device::CPU);
    bc1_tensor.at<float>(0) = bias_correction1;
    
    Tensor bc2_tensor({1}, DType::Float32, Device::CPU);
    bc2_tensor.at<float>(0) = bias_correction2;
    
    for (size_t i = 0; i < processing_.size(); ++i) {
        Tensor& param = processing_[i];
        Tensor& m = m_[i];
        Tensor& v = v_[i];
        
        if (param.requiredGrad() && param.grad()) {
            Tensor grad = *param.grad();
            
            // Update m: m = beta1 * m + (1 - beta1) * grad
            m = (m * beta1_tensor) + (grad * (one_tensor - beta1_tensor));
            
            // Update v: v = beta2 * v + (1 - beta2) * grad^2
            Tensor grad_squared = grad * grad;
            v = (v * beta2_tensor) + (grad_squared * (one_tensor - beta2_tensor));
            
            // Bias correction
            Tensor m_hat = m / bc1_tensor;
            Tensor v_hat = v / bc2_tensor;
            
            // Update parameter: param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
            Tensor sqrt_v_hat = v_hat.sqrt();
            Tensor denominator = sqrt_v_hat + epsilon_tensor;
            Tensor update = m_hat / denominator;
            param = param - (update * lr_tensor);
        }
    }
}
