#ifndef ADAMW
#define ADAMW
#include "optimizer.h"
#include <vector>

class adamw : public optimizer{
public:
    adamw(const std::vector<Tensor>& parameters, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8, float weight_decay = 0.01);
    void step(float learningRate) override;

private:
    float beta1_;
    float beta2_;
    float epsilon_;
    float weight_decay_;
    int64_t step_;
    std::vector<Tensor> m_;
    std::vector<Tensor> v_;
};



#endif
