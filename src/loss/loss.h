#ifndef LOSS    
#define LOSS
#include "../tensor/tensor.h"

class loss{
public:
    virtual Tensor forward(const Tensor& predictions, const Tensor& targets)=0;

protected:
    string reduction_ = "mean";


};
#endif