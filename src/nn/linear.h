#ifndef LINEAR
#define LINEAR

#include "../tensor/tensor.h"
#include "module.h"

class Linear : public Module {
public:
    Linear(int64_t inFeatures, int64_t outFeatures, DType dtype);
    Tensor forward(const Tensor& input) override;

    const Tensor& bias() const {return bias_;};
    const Tensor& weight() const {return weight_;};
    
    std::vector<Tensor> parameters() override;
    void zero_grad() override;
    void to_cuda() override;

private: 
    int64_t inFeatures_;
    int64_t outFeatures_;
    Tensor weight_;
    Tensor bias_;

    void init_parameters();
};



#endif 