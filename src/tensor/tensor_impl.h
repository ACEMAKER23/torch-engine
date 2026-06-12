#ifndef TENSORIMPL
#define TENSORIMPL

#include <memory>
#include <vector>
#include "../core/dtype.h"
#include "tensor_storage.h"

class TensorImpl {
public:
    std::shared_ptr<Storage> storage() const { return storage_ ;};
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    DType dtype() const { return dtype_; }
    size_t offset() const { return offset_; }

    size_t numel() const;

    TensorImpl(std::shared_ptr<Storage> storage,
               const std::vector<int64_t>& shape,
               DType dtype);

    TensorImpl(std::shared_ptr<Storage> storage,
               const std::vector<int64_t>& shape,
               DType dtype, int64_t off);

private:
    std::shared_ptr<Storage> storage_;

    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;

    DType dtype_;
    size_t offset_ = 0;

private:
    static std::vector<int64_t> compute_strides(
        const std::vector<int64_t>& shape);
};

#endif