#ifndef LINEAR
#define LINEAR

#include "../tensor/tensor.h"

class Linear{
public:
    Linear(int64_t inFeatures, int64_t outFeatures, DType dtype);
    Tensor forward(const Tensor& input);

    const Tensor& bias() const {return bias_;};
    const Tensor& weight() const {return weight_;};
    
    std::vector<Tensor> parameters();
    void zero_grad();

private: 
    int64_t inFeatures_;
    int64_t outFeatures_;
    Tensor weight_;
    Tensor bias_;

    void init_parameters();
};



#endif 