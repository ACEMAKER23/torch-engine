#include "tensor_impl.h"


TensorImpl::TensorImpl(std::shared_ptr<Storage> storage,
               const std::vector<int64_t>& shape,
               DType dtype)
               : storage_(storage),
                shape_(shape),
                dtype_(dtype),                          
                strides_(compute_strides(shape)),
                offset_(0)
{};


std::vector<int64_t> TensorImpl::compute_strides(
        const std::vector<int64_t>& shape){

            if (shape.size() <= 0) throw runtime_error("Invalid Shape: dimension < 0");
            vector<int64_t> strid(shape.size(),(int64_t)1);

            for(int64_t i = strid.size()-1; i>=1; --i){
                strid[i-1] = shape[i] * strid[i];
            }

            return strid;

};

size_t TensorImpl::numel() const{
    size_t s = 1;
    for (int64_t shape : shape_) s*=shape;
    return s;
};