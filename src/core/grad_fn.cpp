#include "grad_fn.h"
#include "../tensor/tensor.h"

std::vector<Tensor> AddBackward::backward(const Tensor& pathDownGrad) {
    auto grad_a = pathDownGrad;
    auto grad_b = pathDownGrad;
    grad_a.setRequiresGrad(false);
    grad_b.setRequiresGrad(false);
    std::vector<Tensor> result;
    result.push_back(grad_a);
    result.push_back(grad_b);
    return result;
}

std::vector<Tensor> SubBackward::backward(const Tensor& pathDownGrad) {
    auto grad_a = pathDownGrad;
    auto grad_b = -pathDownGrad;
    grad_a.setRequiresGrad(false);
    grad_b.setRequiresGrad(false);
    std::vector<Tensor> result;
    result.push_back(grad_a);
    result.push_back(grad_b);
    return result;
}

std::vector<Tensor> MulBackward::backward(const Tensor& pathDownGrad) {
    auto grad_a = pathDownGrad * *inputs[1];
    auto grad_b = pathDownGrad * *inputs[0];
    grad_a.setRequiresGrad(false);
    grad_b.setRequiresGrad(false);
    std::vector<Tensor> result;
    result.push_back(grad_a);
    result.push_back(grad_b);
    return result;
}

std::vector<Tensor> MatMulBackward::backward(const Tensor& pathDownGrad) {
    auto grad_a = pathDownGrad.matmul(inputs[1]->transpose_view(inputs[1]->shape().size()-2,inputs[1]->shape().size()-1));
    auto grad_b = inputs[0]->transpose_view(inputs[0]->shape().size()-2,inputs[0]->shape().size()-1).matmul(pathDownGrad);
    grad_a.setRequiresGrad(false);
    grad_b.setRequiresGrad(false);
    std::vector<Tensor> result;
    result.push_back(grad_a);
    result.push_back(grad_b);
    return result;
}


std::vector<Tensor> DivBackward::backward(const Tensor& pathDownGrad) {
    // For c = a/b: dc/da = 1/b, dc/db = -a/b^2
    auto grad_a = pathDownGrad / *inputs[1];
    auto grad_b = -pathDownGrad * *inputs[0] / (*inputs[1] * *inputs[1]);
    grad_a.setRequiresGrad(false);
    grad_b.setRequiresGrad(false);
    std::vector<Tensor> result;
    result.push_back(grad_a);
    result.push_back(grad_b);
    return result;
}

