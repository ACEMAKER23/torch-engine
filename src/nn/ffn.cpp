#include "ffn.h"
#include "gelu.h"
#include "linear.h"
#include <memory>

FeedForward::FeedForward(int64_t embedDim, int64_t ffDim, DType dtype):
linear1_(std::make_unique<Linear>(embedDim,ffDim,dtype)),
gelu_(std::make_unique<GeLU>()),
linear2_(std::make_unique<Linear>(ffDim,embedDim,dtype))
{}

Tensor FeedForward::forward(const Tensor& input){
    Tensor output = linear1_->forward(input);
    output = gelu_->forward(output);
    return (linear2_->forward(output));
}

std::vector<Tensor> FeedForward::parameters(){
    std::vector<Tensor> parameter;
    auto linear1_params = linear1_->parameters();
    parameter.insert(parameter.end(), linear1_params.begin(), linear1_params.end());
    
    auto gelu_params = gelu_->parameters();
    if (!gelu_params.empty()) {
        parameter.insert(parameter.end(), gelu_params.begin(), gelu_params.end());
    }
    
    auto linear2_params = linear2_->parameters();
    parameter.insert(parameter.end(), linear2_params.begin(), linear2_params.end());
    
    return parameter;
}

void FeedForward::zero_grad(){
    linear1_->zero_grad();
    gelu_->zero_grad();
    linear2_->zero_grad();
}

void FeedForward::to_cuda() {
    linear1_->to_cuda();
    linear2_->to_cuda();
}