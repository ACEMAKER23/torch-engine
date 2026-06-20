#ifndef LAYERNORM
#define LAYERNORM

#include "../tensor/tensor.h"

class LayerNorm {
public:
    LayerNorm(int64_t normalized_shape, DType dtype, float eps = 1e-5);
    Tensor forward(const Tensor& input);

    const Tensor& weight() const { return weight_; };
    const Tensor& bias() const { return bias_; };
    
    std::vector<Tensor> parameters();
    void zero_grad();

private:
    int64_t normalized_shape_;
    float eps_;
    Tensor weight_;
    Tensor bias_;

    void init_parameters();
};

#endif
