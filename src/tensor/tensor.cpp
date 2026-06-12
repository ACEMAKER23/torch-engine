#include "tensor.h"
#include "tensor_impl.h"
#include <cstdint>
#include <memory>
#include <cstring>
#include <stdexcept>

size_t Tensor::numel() const {return impl_->numel();}

Tensor::Tensor(std::vector<int64_t> shape, DType dtype, Device deviceT){
    size_t s = 1;
    for(int64_t sh : shape) s*=sh;
    std::shared_ptr<Allocator> allocator;
    if (deviceT == Device::CUDA) {
        allocator = std::make_shared<CUDAAllocatorPlaceHolder>();
    } else {
        allocator = std::make_shared<CPUAllocator>();
    }

    auto storage = Storage::allocate(s * dtype_size(dtype), allocator);
    
    impl_=make_shared<TensorImpl>(storage, shape, dtype);
}

Tensor::Tensor(shared_ptr<Storage> store, const std::vector<int64_t>& shape, DType dtype){
    impl_=make_shared<TensorImpl>(store,shape,dtype);
}

Tensor::Tensor(shared_ptr<TensorImpl> tImple):
    impl_(tImple)
{}


Tensor Tensor::clone(){
    size_t bt = impl_->storage()->bytes();
    Device devi = impl_->storage()->device();
    DType dty = impl_->dtype();

    auto allocator = impl_->storage()->allocator();
    auto storage = Storage::allocate(bt, allocator);

    std::memcpy(storage->data(), impl_->storage()->data(),bt);


    return (Tensor(storage,impl_->shape(),dty));
}

void Tensor::view(const vector<int64_t>& newShape){
    auto sameStorage= impl_->storage();
    //size_t sameOffset = impl_->offset();

    size_t newNumElement = 1;
    for(int64_t i : newShape) newNumElement *= i;
    if (newNumElement != impl_->numel()){ 
        throw runtime_error("Shape error: new shape should contian the same number of elements as before");
   }

   impl_ = make_shared<TensorImpl>(sameStorage , newShape , impl_->dtype()); 
}

Tensor Tensor::slice(int64_t start,int64_t finish,int64_t dimension){
    auto newShape = impl_->shape();
    newShape[dimension]=finish - start;

    auto newOffset = impl_->offset() + start * impl_->strides()[dimension];

    auto newImpl = make_shared<TensorImpl>(impl_->storage(), newShape, impl_->dtype(), newOffset);
    return(Tensor(newImpl));
}

const std::vector<int64_t>& Tensor::shape() const{
    return (impl_->shape());
}

const std::vector<int64_t>& Tensor::strides() const{
    return (impl_->strides());
}
