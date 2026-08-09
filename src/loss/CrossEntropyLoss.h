#ifndef CROSS
#define CROSS
#include "loss.h"

class crossEntropyLoss : public loss{
public:
    // PyTorch-style: predictions are raw logits, targets are class indices (int64)
    Tensor forward(const Tensor& predictions, const Tensor& targets) override;

    // Probability-based: predictions are softmax probabilities, targets are class indices (int64)
    Tensor forward_with_probs(const Tensor& predictions, const Tensor& targets);

    // Batched PyTorch-style: predictions [B, T, V], targets [B, T] int64
    Tensor forward_batched(const Tensor& predictions, const Tensor& targets);

};

#endif 