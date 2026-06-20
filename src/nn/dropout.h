#ifndef DROPOUT
#define DROPOUT

#include "../tensor/tensor.h"
#include <random>

class Dropout {
public:
    Dropout(float p = 0.5);
    Tensor forward(const Tensor& input);
    void set_training(bool training);

    std::vector<Tensor> parameters() { return {}; };

private:
    float p_;
    bool training_;
    std::mt19937 gen_;
};

#endif
