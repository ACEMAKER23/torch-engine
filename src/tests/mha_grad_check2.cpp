#include <iostream>
#include <cmath>
#include "../nn/multi_head_attention.h"
#include "../tensor/tensor.h"
#include "../core/dtype.h"

int main(){
    const int64_t B=2, T=4, D=8, H=2;
    MultiHeadAttention mha(D, H, DType::Float32);

    Tensor input({B,T,D}, DType::Float32, Device::CPU);
    for(size_t i=0;i<input.numel();++i) input.at<float>(i)=i*0.01f;
    input.setRequiresGrad(true);

    Tensor out = mha.forward(input);
    std::cout << "out shape: ";
    for(auto s: out.shape()) std::cout << s << " ";
    std::cout << " requires_grad=" << out.requiredGrad() << "\n";

    out.backward();

    auto params = mha.parameters();
    std::cout << "num params: " << params.size() << "\n";
    for(size_t i=0;i<params.size();++i){
        auto g = params[i].grad();
        std::cout << "param " << i << " shape ";
        for(auto s: params[i].shape()) std::cout << s << " ";
        std::cout << " grad=" << (g ? g->numel() : 0);
        if(g){
            float s=0;
            for(size_t j=0;j<g->numel();++j) s+=(*g).at<float>(j);
            std::cout << " grad_sum=" << s;
        }
        std::cout << "\n";
    }
    auto gi = input.grad();
    std::cout << "input grad=" << (gi ? gi->numel() : 0);
    if(gi){ float s=0; for(size_t j=0;j<gi->numel();++j) s+=gi->at<float>(j); std::cout << " sum="<<s; }
    std::cout << "\n";
}
