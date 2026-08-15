#ifndef LAYERNORM
#define LAYERNORM

#include "../tensor/tensor.h"
#include "module.h"

class LayerNorm : public Module {
public:
    LayerNorm(int64_t normalized_shape, DType dtype, float eps = 1e-5);
    Tensor forward(const Tensor& input) override;

    const Tensor& weight() const { return weight_; };
    const Tensor& bias() const { return bias_; };
    
    std::vector<Tensor> parameters() override;
    void zero_grad() override;
    void to_cuda() override;

private:
    int64_t normalized_shape_;
    float eps_;
    Tensor weight_;
    Tensor bias_;

    void init_parameters();
};

#endif
