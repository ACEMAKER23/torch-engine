#ifndef GRADFN
#define GRADFN
#include <vector>
#include <memory>

class Tensor;
class GradFn{
public:
    // return a vector of tensor that represent the gradent if the loss
    //in terms of all the input to the function
    virtual std::vector<Tensor> backward(const Tensor& pastDownGrad)=0;

    //vector that hold all the tensor that are input to the function.
    //Stored by value: each Tensor shares its TensorImpl (data + autograd state)
    //with the original, and keeps the autograd graph alive via shared_ptr.
    std::vector<Tensor> inputs;
};

class AddBackward : public GradFn{
public:
    std::vector<Tensor> backward(const Tensor& pathDownGrad) override;
};

class SubBackward : public GradFn{
public:
    std::vector<Tensor> backward(const Tensor& pathDownGrad) override;
};

class MulBackward : public GradFn{
public:
    std::vector<Tensor> backward(const Tensor& pathDownGrad) override;
};
class MatMulBackward  : public GradFn {
public:
    std::vector<Tensor> backward(const Tensor& PathDownGrad) override;
};
class DivBackward : public GradFn {
public:
    std::vector<Tensor> backward(const Tensor& PathDownGrad) override;
};

class ReluBackward : public GradFn {
public:
    std::vector<Tensor> backward(const Tensor& pathDownGrad) override;
};

class GeluBackward : public GradFn {
public:
    std::vector<Tensor> backward(const Tensor& pathDownGrad) override;
};

class SigmoidBackward : public GradFn {
public:
    std::vector<Tensor> backward(const Tensor& pathDownGrad) override;
};
#endif