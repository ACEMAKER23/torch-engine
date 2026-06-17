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

    //vector that hold all the tensor that are input to the function
    std::vector<Tensor*> inputs;
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
#endif