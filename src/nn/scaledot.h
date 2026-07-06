#ifndef SCALEDOT
#define SCALEDOT
#include "../tensor/tensor.h"

class ScaledDot {
public:
    Tensor forward(const Tensor Q, const Tensor K, const Tensor V);
    std::vector<Tensor> parameters();
    void zero_grad();
};

#endif