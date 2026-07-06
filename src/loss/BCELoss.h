#ifndef BCELOSS
#define BCELOSS
#include "loss.h"

class BCELoss : public loss {
public:
    // Binary Cross Entropy: loss = -[y*log(p) + (1-y)*log(1-p)]
    // predictions are probabilities (0-1), targets are 0 or 1
    Tensor forward(const Tensor& predictions, const Tensor& targets) override;
};

#endif
