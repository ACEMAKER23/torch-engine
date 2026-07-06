#ifndef RELULAYER
#define RELULAYER
#include "../tensor/tensor.h"
#include "module.h"


class ReLU : public Module {
public:
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override { return {}; };
    void zero_grad() override {}
};




#endif