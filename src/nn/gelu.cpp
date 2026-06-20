#include "gelu.h"


Tensor GeLU::forward(const Tensor& input) const {
    return (input.gelu());
}