#include "relu.h"
Tensor ReLU::forward(const Tensor& input) {
    return (input.relu());
}