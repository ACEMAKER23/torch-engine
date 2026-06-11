#ifndef TENSOR
#define TENSOR

#include "tensor_impl.h"

class Tensor {
public:
    Tensor(std::vector<int64_t> shape, DType dtype, Device deviceT);

    size_t numel() const;

    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    Tensor clone();

private:
    std::shared_ptr<TensorImpl> impl_;
    Tensor(shared_ptr<Storage> store, const std::vector<int64_t>& shape, DType dtype);
};


#endif