#ifndef MSELOSS
#define MSELOSS
#include "loss.h"

class MSELoss : public loss {
public:
    // Mean Squared Error: loss = mean((predictions - targets)^2)
    Tensor forward(const Tensor& predictions, const Tensor& targets) override;
};

#endif
