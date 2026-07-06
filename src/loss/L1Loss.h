#ifndef L1LOSS
#define L1LOSS
#include "loss.h"

class L1Loss : public loss {
public:
    // Mean Absolute Error: loss = mean(|predictions - targets|)
    Tensor forward(const Tensor& predictions, const Tensor& targets) override;
};

#endif
