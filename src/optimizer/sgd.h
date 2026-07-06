#ifndef SGD
#define SGD
#include "optimizer.h"

class sgd : public optimizer{
public:
    sgd(const std::vector<Tensor>& paramters) : optimizer(paramters){};
    void step(float learningRate) override;
};



#endif