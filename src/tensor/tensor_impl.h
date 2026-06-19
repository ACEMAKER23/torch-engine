#ifndef TENSORIMPL
#define TENSORIMPL

#include <memory>
#include <vector>
#include "../core/dtype.h"
#include "tensor_storage.h"

// Forward declarations for autograd metadata held by the shared TensorImpl.
// Storing these here (PyTorch-style) means every Tensor handle that shares this
// impl also shares the same gradient and grad_fn, which keeps the autograd graph
// alive and lets backward() write gradients back into the original leaf tensors.
class Tensor;
class GradFn;

class TensorImpl {
public:
    std::shared_ptr<Storage> storage() const { return storage_ ;};
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    DType dtype() const { return dtype_; }
    size_t offset() const { return offset_; }

    size_t numel() const;

    // ----- Autograd metadata (shared across all handles to this tensor) -----
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool value) { requires_grad_ = value; }

    std::shared_ptr<Tensor> grad() const { return grad_; }
    void set_grad(std::shared_ptr<Tensor> g) { grad_ = std::move(g); }

    std::shared_ptr<GradFn> grad_fn() const { return grad_fn_; }
    void set_grad_fn(std::shared_ptr<GradFn> fn) { grad_fn_ = std::move(fn); }
    // ------------------------------------------------------------------------

    TensorImpl(std::shared_ptr<Storage> storage,
               const std::vector<int64_t>& shape,
               DType dtype);

    TensorImpl(std::shared_ptr<Storage> storage,
               const std::vector<int64_t>& shape,
               DType dtype, int64_t off);

    TensorImpl(std::shared_ptr<Storage> storage,
               const std::vector<int64_t>& shape,
               const std::vector<int64_t>& strides,
               DType dtype, int64_t off);

private:
    std::shared_ptr<Storage> storage_;

    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;

    DType dtype_;
    size_t offset_ = 0;

    // Autograd state
    bool requires_grad_ = false;
    std::shared_ptr<Tensor> grad_;
    std::shared_ptr<GradFn> grad_fn_;

private:
    static std::vector<int64_t> compute_strides(
        const std::vector<int64_t>& shape);
};

#endif