#ifndef TENSOR
#define TENSOR

#include "tensor_impl.h"
#include <memory>

class Tensor {
    friend class TensorTest_BasicConstruction_Test;
    friend class TensorTest_CopySemantics_Test;
    friend class TensorTest_MoveSemantics_Test;
    friend class TensorTest_CloneDeepCopy_Test;
    friend class TensorTest_ViewReshape_Test;
    friend class TensorTest_SliceBasic_Test;
    friend class TensorTest_NestedSlice_Test;
public:
    Tensor(std::vector<int64_t> shape, DType dtype, Device deviceT);

    size_t numel() const;

    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    Tensor clone();
    void view(const vector<int64_t>& newShape);
    Tensor slice(int64_t start,int64_t finish,int64_t dimension);

    const std::vector<int64_t>& shape() const;
    const std::vector<int64_t>& strides() const;

    Tensor operator*(const Tensor& other);
    Tensor operator+(const Tensor& other);
    Tensor operator-(const Tensor& other);

private:
    std::shared_ptr<TensorImpl> impl_;

    Tensor(shared_ptr<Storage> store, const std::vector<int64_t>& shape, DType dtype);
    Tensor(shared_ptr<TensorImpl> tImple);
};


#endif