#ifndef ADAM
#define ADAM
#include "optimizer.h"
#include <vector>

class adam : public optimizer{
public:
    adam(const std::vector<Tensor>& parameters, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8);
    void step(float learningRate) override;

private:
    float beta1_;
    float beta2_;
    float epsilon_;
    int64_t step_;
    std::vector<Tensor> m_;
    std::vector<Tensor> v_;
};



#endif