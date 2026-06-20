#ifndef GELULAYER
#define GELULAYER
#include "../tensor/tensor.h"

class GeLU{
public:
    Tensor forward(const Tensor& input) const;
    std::vector<Tensor> parameters() { return {}; };
};

#endif