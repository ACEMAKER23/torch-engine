#include "sequential.h"

void Sequential::add(std::shared_ptr<Module> layer) {
    layers_.push_back(layer);
}

Tensor Sequential::forward(const Tensor& input) {
    Tensor output = input;
    for (const auto& layer : layers_) {
        output = layer->forward(output);
    }
    return output;
}

std::vector<Tensor> Sequential::parameters() {
    std::vector<Tensor> params;
    for (const auto& layer : layers_) {
        auto layer_params = layer->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

void Sequential::zero_grad() {
    for (const auto& layer : layers_) {
        layer->zero_grad();
    }
}