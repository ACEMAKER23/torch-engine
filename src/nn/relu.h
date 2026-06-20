#ifndef RELULAYER
#define RELULAYER
#include "../tensor/tensor.h"


class ReLU{
public:
    Tensor forward(const Tensor& input) const;
    std::vector<Tensor> parameters() { return {}; };
};




#endif