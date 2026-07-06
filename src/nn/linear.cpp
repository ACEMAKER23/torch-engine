#include "linear.h"
#include <random>
#include <cmath>

Linear::Linear(int64_t inFeatures, int64_t outFeatures, DType dtype)
: inFeatures_(inFeatures), outFeatures_(outFeatures),
weight_({outFeatures,inFeatures},dtype,Device::CPU),
bias_({outFeatures}, dtype, Device::CPU)
{init_parameters();
weight_.setRequiresGrad(true);
bias_.setRequiresGrad(true);}


Tensor Linear::forward(const Tensor& input) {
    // input: [batch_size, in_features]
    // weight: [out_features, in_features]
    // bias: [out_features]
    // output: [batch_size, out_features]
    
    Tensor result = input.matmul(weight_.transpose_view(1, 0));  // [batch_size, out_features]
    result = result + bias_;  // broadcasting
    return result;
}

void Linear::init_parameters(){
    if (inFeatures_ == 0) {
        throw std::runtime_error("inFeatures cannot be zero");
    }
    
    if (weight_.dtype() != DType::Float32) {
        throw std::runtime_error("Only Float32 dtype is supported for parameter initialization");
    }

    std::mt19937 gen(std::random_device{}());

    float stddev = std::sqrt(2.0f / inFeatures_);
    std::normal_distribution<float> dist(0.0f, stddev);

    auto* data = static_cast<float*>(weight_.data());
    for (size_t i=0; i<weight_.numel(); ++i){
        data[i] = dist(gen);
    }

    auto* biases = static_cast<float*>(bias_.data());
    for (size_t i=0; i<bias_.numel(); ++i){
        biases[i] = 0;
    }
}

std::vector<Tensor> Linear::parameters() {
    return {weight_, bias_};
}

void Linear::zero_grad() {
    if (weight_.grad() && weight_.dtype() == DType::Float32) {
        weight_.grad()->fill_<float>(0.0f);
    }
    if (bias_.grad() && bias_.dtype() == DType::Float32) {
        bias_.grad()->fill_<float>(0.0f);
    }
}