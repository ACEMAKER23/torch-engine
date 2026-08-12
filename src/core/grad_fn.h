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

class CrossEntropyBackward : public GradFn {
public:
    std::vector<Tensor> backward(const Tensor& pathDownGrad) override;
};

class CrossEntropyWithProbsBackward : public GradFn {
public:
    std::vector<Tensor> backward(const Tensor& pathDownGrad) override;
};

class CrossEntropyBatchedBackward : public GradFn {
public:
    std::vector<Tensor> backward(const Tensor& pathDownGrad) override;
};

class MSEBackward : public GradFn {
public:
    std::vector<Tensor> backward(const Tensor& pathDownGrad) override;
};

class BCEBackward : public GradFn {
public:
    std::vector<Tensor> backward(const Tensor& pathDownGrad) override;
};

// Gradient for transpose_view: inputs[0] = original tensor before transpose.
// Backward simply transposes the upstream gradient back with the same d1/d2.
class TransposeBackward : public GradFn {
public:
    size_t d1 = 0, d2 = 1;
    std::vector<Tensor> backward(const Tensor& pathDownGrad) override;
};

// Gradient for LayerNorm.
// inputs[0] = original input x, inputs[1] = weight gamma, inputs[2] = bias beta.
// eps is stored separately because it is not a Tensor.
class LayerNormBackward : public GradFn {
public:
    float eps = 1e-5f;
    std::vector<Tensor> backward(const Tensor& pathDownGrad) override;
};

// Gradient for view(): inputs[0] = original tensor.
// View is a zero-copy metadata change; backward just reshapes the upstream
// gradient back to the original input shape.
class ViewBackward : public GradFn {
public:
    std::vector<int64_t> original_shape;
    std::vector<Tensor> backward(const Tensor& pathDownGrad) override;
};

// Gradient for contiguous() / clone(): inputs[0] = original tensor.
// A copy is the identity w.r.t. its input; backward passes the upstream
// gradient through unchanged.
class CopyBackward : public GradFn {
public:
    std::vector<Tensor> backward(const Tensor& pathDownGrad) override;
};

// Gradient for reshape(): inputs[0] = original (possibly non-contiguous) tensor.
// When the input is already contiguous, reshape() degrades to view() and this
// node is never created.  When a copy was required (non-contiguous input),
// backward reshapes the upstream gradient back to the original shape.
// Gradients flowing back through the network are always contiguous (CUDA
// kernels write row-major output), so view() inside backward is free.
class ReshapeBackward : public GradFn {
public:
    std::vector<int64_t> original_shape;
    std::vector<Tensor> backward(const Tensor& pathDownGrad) override;
};

// Gradient for softmax(x, dim): inputs[0] = original pre-softmax logits.
// Backward recomputes s = softmax(x) then dx_i = s_i * (dy_i - sum_j(dy_j * s_j)).
class SoftmaxBackward : public GradFn {
public:
    int64_t dim = -1;
    std::vector<Tensor> backward(const Tensor& pathDownGrad) override;
};

class L1Backward : public GradFn {
public:
    std::vector<Tensor> backward(const Tensor& pathDownGrad) override;
};

// Gradient for embedding lookup: inputs[0] = indices, inputs[1] = weight matrix.
// Backward accumulates d_weight[index] += pathDownGrad[i] for each occurrence.
class EmbeddingBackward : public GradFn {
public:
    int64_t num_embeddings = 0;
    int64_t embedding_dim = 0;
    std::vector<Tensor> backward(const Tensor& pathDownGrad) override;
};
#endif