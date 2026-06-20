#include "relu.h"
Tensor ReLU::forward(const Tensor& input) const{
    return (input.relu());
}