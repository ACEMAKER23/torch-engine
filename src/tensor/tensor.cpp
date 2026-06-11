#include "tensor.h"
#include "tensor_impl.h"
#include <memory>
#include <cstring>

size_t Tensor::numel() const {return impl_->numel();}

Tensor::Tensor(std::vector<int64_t> shape, DType dtype, Device deviceT){
    size_t s = 1;
    for(int64_t sh : shape) s*=sh;
    auto allocator = (deviceT==Device::CUDA) ? nullptr : make_shared<CPUAllocator>();

    auto storage = Storage::allocate(s * dtype_size(dtype), allocator);
    
    impl_=make_shared<TensorImpl>(storage, shape, dtype);
}

Tensor::Tensor(shared_ptr<Storage> store, const std::vector<int64_t>& shape, DType dtype){
    impl_=make_shared<TensorImpl>(store,shape,dtype);
}


Tensor Tensor::clone(){
    size_t bt = impl_->storage()->bytes();
    Device devi = impl_->storage()->device();
    DType dty = impl_->dtype();

    auto allocator = impl_->storage()->allocator();
    auto storage = Storage::allocate(bt, allocator);

    std::memcpy(storage->data(), impl_->storage()->data(),bt);


    return (Tensor(storage,impl_->shape(),dty));
}