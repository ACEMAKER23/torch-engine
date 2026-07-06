#ifndef DROPOUT
#define DROPOUT

#include "../tensor/tensor.h"
#include <random>
#include "module.h"

class Dropout : public Module {
public:
    Dropout(float p = 0.5);
    Tensor forward(const Tensor& input) override;
    void set_training(bool training);

    std::vector<Tensor> parameters() override { return {}; };
    void zero_grad() override {}

private:
    float p_;
    bool training_;
    std::mt19937 gen_;
};

#endif
