#ifndef GELULAYER
#define GELULAYER
#include "../tensor/tensor.h"
#include "module.h"

class GeLU : public Module {
public:
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override { return {}; };
    void zero_grad() override {}
};

#endif