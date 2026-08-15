#ifndef MODULE
#define MODULE

#include "../tensor/tensor.h"

class Module {
public:
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual std::vector<Tensor> parameters() = 0;
    virtual void zero_grad() = 0;
    virtual void to_cuda() {}
};

#endif
