#include "gelu.h"


Tensor GeLU::forward(const Tensor& input) {
    return (input.gelu());
}