#include "tensor.h"
#include "tensor_impl.h"
#include "../core/allocators.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <optional>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <typeinfo>
#include <chrono>

#ifdef __GNUC__
#include <cxxabi.h>
#endif


#ifdef USE_CUDA
#include "../cuda/elementwise_cuda.h"
#include "../core/cuda_utils.h"
// Generic dispatcher for CUDA binary elementwise operations
template<typename Op>
void dispatch_cuda_binary_op(const Tensor& a, const Tensor& b, Tensor& result, Op op) {
    switch (a.dtype()) {
        case DType::Float32:
            op(static_cast<const float*>(a.data()), static_cast<const float*>(b.data()), static_cast<float*>(result.data()), result.numel());
            break;
        case DType::Int32:
            op(static_cast<const int32_t*>(a.data()), static_cast<const int32_t*>(b.data()), static_cast<int32_t*>(result.data()), result.numel());
            break;
        case DType::Int64:
            op(static_cast<const int64_t*>(a.data()), static_cast<const int64_t*>(b.data()), static_cast<int64_t*>(result.data()), result.numel());
            break;
        case DType::Float16:
            op(static_cast<const float16_t*>(a.data()), static_cast<const float16_t*>(b.data()), static_cast<float16_t*>(result.data()), result.numel());
            break;
        case DType::BFloat16:
            op(static_cast<const bfloat16_t*>(a.data()), static_cast<const bfloat16_t*>(b.data()), static_cast<bfloat16_t*>(result.data()), result.numel());
            break;
        default:
            throw std::runtime_error("Unsupported dtype for CUDA binary operation");
    }
}
#endif

size_t Tensor::numel() const {return impl_->numel();}

Device Tensor::device() const {
    return impl_->storage()->device();
}

Tensor::Tensor(std::vector<int64_t> shape, DType dtype, Device deviceT){
    size_t s = 1;
    for(int64_t sh : shape) s*=sh;
    std::shared_ptr<Allocator> allocator;
    if (deviceT == Device::CUDA) {
#ifdef USE_CUDA
        allocator = std::make_shared<CUDAAllocator>();
#else
        throw std::runtime_error("CUDA not supported in this build");
#endif
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
{
}


Tensor Tensor::clone() const{
    size_t bt = impl_->storage()->bytes();
    Device devi = impl_->storage()->device();
    DType dty = impl_->dtype();

    auto allocator = impl_->storage()->allocator();
    auto storage = Storage::allocate(bt, allocator);

    if (devi == Device::CUDA) {
#ifdef USE_CUDA
        cuda_check_error(cudaMemcpy(storage->data(), impl_->storage()->data(), bt, cudaMemcpyDeviceToDevice),
                         "cudaMemcpy in clone");
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        std::memcpy(storage->data(), impl_->storage()->data(), bt);
    }

    return (Tensor(storage,impl_->shape(),dty));
}

bool Tensor::is_contiguous() const {
    const auto& sh = impl_->shape();
    const auto& st = impl_->strides();
    if (sh.empty()) return true;
    int64_t expected = 1;
    for (int64_t i = (int64_t)sh.size() - 1; i >= 0; --i) {
        if (st[i] != expected) return false;
        expected *= sh[i];
    }
    return true;
}

Tensor Tensor::contiguous() const {
    // Already contiguous: return a shallow copy of this tensor. No copy node
    // is needed in the autograd graph because it is an identity.
    if (is_contiguous()) {
        return *this;
    }

    // Not contiguous: allocate fresh row-major storage and copy the logical
    // elements following the original (possibly strided) layout.
    size_t numElements = impl_->numel();
    size_t elementSize = 0;

    switch (impl_->dtype()) {
        case DType::Float32:
            elementSize = sizeof(float);
            break;
        case DType::Int32:
            elementSize = sizeof(int32_t);
            break;
        case DType::Int64:
            elementSize = sizeof(int64_t);
            break;
        default:
            throw runtime_error("Unsupported dtype for contiguous");
    }

    size_t totalBytes = numElements * elementSize;
    auto allocator = impl_->storage()->allocator();
    auto storage = Storage::allocate(totalBytes, allocator);

    const void* srcData = impl_->storage()->data();
    void* dstData = storage->data();
    const auto& shape = impl_->shape();
    const auto& strides = impl_->strides();
    size_t offset = impl_->offset();
    size_t ndim = shape.size();

#ifdef USE_CUDA
    if (device() == Device::CUDA) {
        // Use a CUDA gather kernel to rearrange strided → row-major on device.
        int64_t ndim_i = static_cast<int64_t>(ndim);
        int64_t numel_i = static_cast<int64_t>(numElements);
        int64_t off_i   = static_cast<int64_t>(offset);
        if (impl_->dtype() == DType::Float32) {
            cuda_gather_strided_f32(static_cast<const float*>(srcData),
                                    static_cast<float*>(dstData),
                                    shape.data(), strides.data(),
                                    ndim_i, off_i, numel_i);
        } else if (impl_->dtype() == DType::Int32) {
            cuda_gather_strided_i32(static_cast<const int32_t*>(srcData),
                                    static_cast<int32_t*>(dstData),
                                    shape.data(), strides.data(),
                                    ndim_i, off_i, numel_i);
        } else if (impl_->dtype() == DType::Int64) {
            cuda_gather_strided_i64(static_cast<const int64_t*>(srcData),
                                    static_cast<int64_t*>(dstData),
                                    shape.data(), strides.data(),
                                    ndim_i, off_i, numel_i);
        }
        cuda_check_error(cudaGetLastError(), "cuda_gather_strided failed");
        // No cudaDeviceSynchronize: result stays on GPU; next kernel on the
        // same default stream is automatically serialized.
    } else {
#endif

    auto copy_loop = [&](auto* dst, const auto* src) {
        std::vector<int64_t> idx(ndim, 0);
        for (size_t flat = 0; flat < numElements; ++flat) {
            size_t srcOffset = offset;
            for (size_t d = 0; d < ndim; ++d) {
                srcOffset += idx[d] * strides[d];
            }
            dst[flat] = src[srcOffset];

            // Increment multi-index
            for (int d = (int)ndim - 1; d >= 0; --d) {
                if (++idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
    };

    if (impl_->dtype() == DType::Float32) {
        copy_loop(static_cast<float*>(dstData),
                  static_cast<const float*>(srcData));
    } else if (impl_->dtype() == DType::Int32) {
        copy_loop(static_cast<int32_t*>(dstData),
                  static_cast<const int32_t*>(srcData));
    } else if (impl_->dtype() == DType::Int64) {
        copy_loop(static_cast<int64_t*>(dstData),
                  static_cast<const int64_t*>(srcData));
    }

#ifdef USE_CUDA
    } // end else (CPU path)
#endif

    Tensor result(storage, shape, impl_->dtype());
    if (requiredGrad()) {
        auto fn = std::make_shared<CopyBackward>();
        fn->inputs = {*this};
        result.setGradFn(fn);
        result.setRequiresGrad(true);
    }
    return result;
}

Tensor Tensor::view(const vector<int64_t>& newShape) const {
    size_t newNumElement = 1;
    for (int64_t i : newShape) newNumElement *= i;
    if (newNumElement != impl_->numel()) {
        throw runtime_error("Shape error: new shape should contain the same number of elements as before");
    }

    // Preserve the original offset but recompute row-major strides for the new
    // shape so the view is valid for contiguous storage.
    auto newImpl = std::make_shared<TensorImpl>(
        impl_->storage(), newShape, impl_->dtype(), static_cast<int64_t>(impl_->offset()));
    Tensor result(newImpl);

    if (requiredGrad()) {
        auto fn = std::make_shared<ViewBackward>();
        fn->inputs = {*this};
        fn->original_shape = impl_->shape();
        result.setGradFn(fn);
        result.setRequiresGrad(true);
    }

    return result;
}

Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
    size_t new_numel = 1;
    for (int64_t d : new_shape) new_numel *= static_cast<size_t>(d);
    if (new_numel != impl_->numel())
        throw std::runtime_error("reshape: new shape must have the same number of elements");

    // Fast path: if already contiguous, a zero-copy view suffices.  This
    // creates a single ViewBackward node, same as calling view() directly.
    if (is_contiguous()) return view(new_shape);

    // Slow path: the tensor is non-contiguous (e.g. result of transpose_view),
    // so a physical copy is required before the memory can be reinterpreted.
    // We build the contiguous storage here and attach a single ReshapeBackward
    // node — cheaper than the CopyBackward + ViewBackward pair that a naive
    // contiguous().view() would create.
    Tensor cont = contiguous();

    auto new_impl = std::make_shared<TensorImpl>(
        cont.impl_->storage(), new_shape, cont.impl_->dtype(),
        static_cast<int64_t>(cont.impl_->offset()));
    Tensor result(new_impl);

    if (requiredGrad()) {
        auto fn           = std::make_shared<ReshapeBackward>();
        fn->inputs        = {*this};
        fn->original_shape = impl_->shape();
        result.setGradFn(fn);
        result.setRequiresGrad(true);
    }
    return result;
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

// PyTorch-style data access
void* Tensor::data() {
    return static_cast<void*>(
        static_cast<char*>(impl_->storage()->data()) + 
        impl_->offset() * dtype_size(impl_->dtype())
    );
}

const void* Tensor::data() const {
    return static_cast<const void*>(
        static_cast<const char*>(impl_->storage()->data()) + 
        impl_->offset() * dtype_size(impl_->dtype())
    );
}


Tensor Tensor::operator*(const Tensor& other) const{
    if (impl_->dtype() != other.impl_->dtype()) {
        throw std::runtime_error("Dtypes must match for elementwise multiplication");
    }
    if (device() != other.device()) {
        throw std::runtime_error("Device Mismatch");
    }
    if (device() == Device::CUDA) {
#ifdef USE_CUDA
        {
            std::vector<int64_t> bc = broadcast_to_shape<float>(shape(), other.shape());
            Tensor a_c = broadcast(bc).contiguous();
            Tensor b_c = other.broadcast(bc).contiguous();
            Tensor result(bc, dtype(), Device::CUDA);
            dispatch_cuda_binary_op(a_c, b_c, result, [](auto a, auto b, auto out, int64_t size) {
                cuda_mul(a, b, out, size);
            });
            cuda_check_error(cudaGetLastError(), "CUDA mul failed");
            if (requiredGrad() || other.requiredGrad()) {
                auto fn = make_shared<MulBackward>();
                fn->inputs = {*this, other};
                result.impl_->set_grad_fn(fn);
                result.impl_->set_requires_grad(true);
            }
            return result;
        }
#else
        throw std::runtime_error("CUDA not supported in this build");
#endif
    }
    std::optional<Tensor> result;
    switch (impl_->dtype()) {
        case DType::Float32:
            result = elementwise_operation<float>(*this, other, [](float a, float b) { return a * b; });
            break;
        case DType::Int32:
            result = elementwise_operation<int32_t>(*this, other, [](int32_t a, int32_t b) { return a * b; });
            break;
        case DType::Int64:
            result = elementwise_operation<int64_t>(*this, other, [](int64_t a, int64_t b) { return a * b; });
            break;
        default:
            throw std::runtime_error("Unsupported dtype");
    }
    if (requiredGrad() || other.requiredGrad()){
        auto fn = make_shared<MulBackward>();
        fn->inputs = {*this, other};
        result->impl_->set_grad_fn(fn);
        result->impl_->set_requires_grad(true);
    }
    return *result;
}
Tensor Tensor::operator+(const Tensor& other) const{
    if (impl_->dtype() != other.impl_->dtype()) {
        throw std::runtime_error("Dtypes must match for elementwise addition");
    }
    if (device() != other.device()) {
        throw std::runtime_error("Device Mismatch");
    }
    if (device() == Device::CUDA) {
#ifdef USE_CUDA
        {
            std::vector<int64_t> bc = broadcast_to_shape<float>(shape(), other.shape());
            Tensor a_c = broadcast(bc).contiguous();
            Tensor b_c = other.broadcast(bc).contiguous();
            Tensor result(bc, dtype(), Device::CUDA);
            dispatch_cuda_binary_op(a_c, b_c, result, [](auto a, auto b, auto out, int64_t size) {
                cuda_add(a, b, out, size);
            });
            cuda_check_error(cudaGetLastError(), "CUDA add failed");
            if (requiredGrad() || other.requiredGrad()) {
                auto fn = make_shared<AddBackward>();
                fn->inputs = {*this, other};
                result.impl_->set_grad_fn(fn);
                result.impl_->set_requires_grad(true);
            }
            return result;
        }
#else
        throw std::runtime_error("CUDA not supported in this build");
#endif
    }
    std::optional<Tensor> result;
    switch (impl_->dtype()) {
        case DType::Float32:
            result = elementwise_operation<float>(*this, other, [](float a, float b) { return a + b; });
            break;
        case DType::Int32:
            result = elementwise_operation<int32_t>(*this, other, [](int32_t a, int32_t b) { return a + b; });
            break;
        case DType::Int64:
            result = elementwise_operation<int64_t>(*this, other, [](int64_t a, int64_t b) { return a + b; });
            break;
        default:
            throw std::runtime_error("Unsupported dtype");
    }
    if (requiredGrad() || other.requiredGrad()){
        auto fn = make_shared<AddBackward>();
        fn->inputs = {*this, other};
        result->impl_->set_grad_fn(fn);
        result->impl_->set_requires_grad(true);
    }
    return *result;
}
Tensor Tensor::operator-(const Tensor& other) const{
    if (impl_->dtype() != other.impl_->dtype()) {
        throw std::runtime_error("Dtypes must match for elementwise subtraction");
    }
    if (device() != other.device()) {
        throw std::runtime_error("Device Mismatch");
    }
    if (device() == Device::CUDA) {
#ifdef USE_CUDA
        {
            std::vector<int64_t> bc = broadcast_to_shape<float>(shape(), other.shape());
            Tensor a_c = broadcast(bc).contiguous();
            Tensor b_c = other.broadcast(bc).contiguous();
            Tensor result(bc, dtype(), Device::CUDA);
            dispatch_cuda_binary_op(a_c, b_c, result, [](auto a, auto b, auto out, int64_t size) {
                cuda_sub(a, b, out, size);
            });
            cuda_check_error(cudaGetLastError(), "CUDA sub failed");
            if (requiredGrad() || other.requiredGrad()) {
                auto fn = make_shared<SubBackward>();
                fn->inputs = {*this, other};
                result.impl_->set_grad_fn(fn);
                result.impl_->set_requires_grad(true);
            }
            return result;
        }
#else
        throw std::runtime_error("CUDA not supported in this build");
#endif
    }
    std::optional<Tensor> result;
    switch (impl_->dtype()) {
        case DType::Float32:
            result = elementwise_operation<float>(*this, other, [](float a, float b) { return a - b; });
            break;
        case DType::Int32:
            result = elementwise_operation<int32_t>(*this, other, [](int32_t a, int32_t b) { return a - b; });
            break;
        case DType::Int64:
            result = elementwise_operation<int64_t>(*this, other, [](int64_t a, int64_t b) { return a - b; });
            break;
        default:
            throw std::runtime_error("Unsupported dtype");
    }
    if (requiredGrad() || other.requiredGrad()){
        auto fn = make_shared<SubBackward>();
        fn->inputs = {*this, other};
        result->impl_->set_grad_fn(fn);
        result->impl_->set_requires_grad(true);
    }
    return *result;
}
Tensor Tensor::operator/(const Tensor& other) const{
    if (impl_->dtype() != other.impl_->dtype()) {
        throw std::runtime_error("Dtypes must match for elementwise division");
    }
    if (device() != other.device()) {
        throw std::runtime_error("Device Mismatch");
    }
    if (device() == Device::CUDA) {
#ifdef USE_CUDA
        {
            std::vector<int64_t> bc = broadcast_to_shape<float>(shape(), other.shape());
            Tensor a_c = broadcast(bc).contiguous();
            Tensor b_c = other.broadcast(bc).contiguous();
            Tensor result(bc, dtype(), Device::CUDA);
            dispatch_cuda_binary_op(a_c, b_c, result, [](auto a, auto b, auto out, int64_t size) {
                cuda_div(a, b, out, size);
            });
            cuda_check_error(cudaGetLastError(), "CUDA div failed");
            if (requiredGrad() || other.requiredGrad()) {
                auto fn = make_shared<DivBackward>();
                fn->inputs = {*this, other};
                result.impl_->set_grad_fn(fn);
                result.impl_->set_requires_grad(true);
            }
            return result;
        }
#else
        throw std::runtime_error("CUDA not supported in this build");
#endif
    }
    std::optional<Tensor> result;
    switch (impl_->dtype()) {
        case DType::Float32:
            result = elementwise_operation<float>(*this, other, [](float a, float b) { return a / b; });
            break;
        case DType::Int32:
            result = elementwise_operation<int32_t>(*this, other, [](int32_t a, int32_t b) { return a / b; });
            break;
        case DType::Int64:
            result = elementwise_operation<int64_t>(*this, other, [](int64_t a, int64_t b) { return a / b; });
            break;
        default:
            throw std::runtime_error("Unsupported dtype");
    }
    if (requiredGrad() || other.requiredGrad()){
        auto fn = make_shared<DivBackward>();
        fn->inputs = {*this, other};
        result->impl_->set_grad_fn(fn);
        result->impl_->set_requires_grad(true);
    }
    return *result;
}

Tensor Tensor::operator-() const {
    Tensor result(shape(), dtype(), device());
#ifdef USE_CUDA
    if (device() == Device::CUDA) {
        if (dtype() == DType::Float32) {
            cuda_negate_f32(static_cast<const float*>(data()),
                            static_cast<float*>(result.data()),
                            static_cast<int64_t>(numel()));
            cuda_check_error(cudaGetLastError(), "cuda_negate_f32 failed");
        } else {
            throw std::runtime_error("CUDA unary negate only supports Float32");
        }
        return result;
    }
#endif
    switch (impl_->dtype()) {
        case DType::Float32:
            for (size_t i = 0; i < numel(); ++i) result.at<float>(i)   = -at<float>(i);
            break;
        case DType::Int32:
            for (size_t i = 0; i < numel(); ++i) result.at<int32_t>(i) = -at<int32_t>(i);
            break;
        case DType::Int64:
            for (size_t i = 0; i < numel(); ++i) result.at<int64_t>(i) = -at<int64_t>(i);
            break;
        default:
            throw std::runtime_error("Unsupported dtype");
    }
    return result;
}


void Tensor::transpose() {
    size_t dimension = shape().size();
    if (dimension != 2) {
        throw runtime_error("transpose() without args only works for 2D tensors");
    }
    transpose(0, 1);
}

void Tensor::transpose(size_t d1, size_t d2) {
    size_t dimension = shape().size();
    if (d1 >= dimension || d2 >= dimension || d1 == d2) {
        throw runtime_error("Invalid transpose dimensions");
    }
    
    // Swap shape dimensions
    std::vector<int64_t> newShape = shape();
    std::swap(newShape[d1], newShape[d2]);
    
    // Swap stride dimensions  
    std::vector<int64_t> newStrides = strides();
    std::swap(newStrides[d1], newStrides[d2]);
    
    // Zero-copy transpose using custom strides
    impl_ = make_shared<TensorImpl>(impl_->storage(), newShape, newStrides, impl_->dtype(), impl_->offset());
}

Tensor Tensor::transpose_view(size_t d1, size_t d2) const {
    size_t dimension = shape().size();
    if (d1 >= dimension || d2 >= dimension || d1 == d2) {
        throw runtime_error("Invalid transpose dimensions");
    }
    
    // Swap shape dimensions
    std::vector<int64_t> newShape = shape();
    std::swap(newShape[d1], newShape[d2]);
    
    // Swap stride dimensions  
    std::vector<int64_t> newStrides = strides();
    std::swap(newStrides[d1], newStrides[d2]);
    
    // Zero-copy transpose using custom strides
    auto newImpl = make_shared<TensorImpl>(impl_->storage(), newShape, newStrides, impl_->dtype(), impl_->offset());
    Tensor result(newImpl);

    // Propagate autograd: if the source requires a gradient, attach a
    // TransposeBackward node so the gradient flows back through this view.
    if (impl_->requires_grad()) {
        result.impl_->set_requires_grad(true);
        auto fn = make_shared<TransposeBackward>();
        fn->inputs = {*this};
        fn->d1 = d1;
        fn->d2 = d2;
        result.impl_->set_grad_fn(fn);
    }

    return result;
}

// Non-inplace versions (create new tensors and return them)
Tensor Tensor::relu() const {
    Tensor result = clone();
    switch (impl_->dtype()) {
        case DType::Float32:
            result.relu_<float>();
            break;
        case DType::Int32:
            result.relu_<int32_t>();
            break;
        case DType::Int64:
            result.relu_<int64_t>();
            break;
        default:
            throw std::runtime_error("Unsupported dtype");
    }
    if (requiredGrad()) {
        auto fn = make_shared<ReluBackward>();
        fn->inputs = {*this};
        result.impl_->set_grad_fn(fn);
        result.impl_->set_requires_grad(true);
    }
    return result;
}

Tensor Tensor::gelu() const {
    Tensor result = clone();
    switch (impl_->dtype()) {
        case DType::Float32:
#ifdef USE_CUDA
            if (device() == Device::CUDA) {
                cuda_gelu(static_cast<const float*>(data()),
                          static_cast<float*>(result.data()),
                          static_cast<int64_t>(numel()));
                cuda_check_error(cudaGetLastError(), "cuda_gelu failed");
                // No sync: result stays on GPU; next kernel serialized on default stream.
                break;
            }
#endif
            result.gelu_<float>();
            break;
        case DType::Int32:
            result.gelu_<int32_t>();
            break;
        case DType::Int64:
            result.gelu_<int64_t>();
            break;
        default:
            throw std::runtime_error("Unsupported dtype");
    }
    if (requiredGrad()) {
        auto fn = make_shared<GeluBackward>();
        fn->inputs = {*this};
        result.impl_->set_grad_fn(fn);
        result.impl_->set_requires_grad(true);
    }
    return result;
}

// Stub implementations for remaining activations

Tensor Tensor::sigmoid() const {
    Tensor result = clone();
    switch (impl_->dtype()) {
        case DType::Float32:
            result.sigmoid_<float>();
            break;
        case DType::Int32:
            result.sigmoid_<int32_t>();
            break;
        case DType::Int64:
            result.sigmoid_<int64_t>();
            break;
        default:
            throw std::runtime_error("Unsupported dtype");
    }
    if (requiredGrad()) {
        auto fn = make_shared<SigmoidBackward>();
        fn->inputs = {*this};
        result.impl_->set_grad_fn(fn);
        result.impl_->set_requires_grad(true);
    }
    return result;
}

Tensor Tensor::sqrt() const {
    Tensor result = clone();
    switch (impl_->dtype()) {
        case DType::Float32:
            result.sqrt_<float>();
            break;
        case DType::Int32:
            result.sqrt_<int32_t>();
            break;
        case DType::Int64:
            result.sqrt_<int64_t>();
            break;
        default:
            throw std::runtime_error("Unsupported dtype");
    }
    return result;
}

Tensor Tensor::exp() const {
    Tensor result = clone();
    switch (impl_->dtype()) {
        case DType::Float32:
            result.exp_<float>();
            break;
        case DType::Int32:
            result.exp_<int32_t>();
            break;
        case DType::Int64:
            result.exp_<int64_t>();
            break;
        default:
            throw std::runtime_error("Unsupported dtype");
    }
    return result;
}

Tensor Tensor::log() const {
    Tensor result = clone();
    switch (impl_->dtype()) {
        case DType::Float32:
            result.log_<float>();
            break;
        case DType::Int32:
            result.log_<int32_t>();
            break;
        case DType::Int64:
            result.log_<int64_t>();
            break;
        default:
            throw std::runtime_error("Unsupported dtype");
    }
    return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
    // tiled_matmul_2d now reads actual strides from each tensor so no
    // contiguous copy is required before dispatching.
    std::optional<Tensor> result;
    switch (impl_->dtype()) {
        case DType::Float32:
            result = matmul_impl<float>(*this, other);
            break;
        case DType::Int32:
            result = matmul_impl<int32_t>(*this, other);
            break;
        case DType::Int64:
            result = matmul_impl<int64_t>(*this, other);
            break;
        default:
            throw runtime_error("Unsupported dtype");
    }
    if (requiredGrad() || other.requiredGrad()){
        auto fn = make_shared<MatMulBackward>();
        fn->inputs = {*this, other};
        result->impl_->set_grad_fn(fn);
        result->impl_->set_requires_grad(true);
    }
    return *result;
}

Tensor Tensor::sum_to_scalar() const {
    Tensor result({1}, dtype(), device());
    result.fill_<float>(0.0f);

#ifdef USE_CUDA
    if (device() == Device::CUDA) {
        cuda_reduce_sum(static_cast<const float*>(data()),
                        static_cast<float*>(result.data()),
                        static_cast<int64_t>(numel()));
        cuda_check_error(cudaGetLastError(), "cuda_reduce_sum failed");
        // No cudaDeviceSynchronize here: the result is a GPU tensor and the
        // following GPU operations on the same stream are naturally ordered.
        return result;
    }
#endif

    float total = 0.0f;
    for (size_t i = 0; i < numel(); ++i) {
        total += at<float>(i);
    }
    result.at<float>(0) = total;
    return result;
}

Tensor Tensor::sum(int64_t dim) const {
    if (dim < 0 || dim >= static_cast<int64_t>(shape().size())) {
        throw std::runtime_error("Dimension out of range for sum");
    }

    // Calculate output shape (remove the summed dimension)
    std::vector<int64_t> output_shape = shape();
    output_shape.erase(output_shape.begin() + dim);

    // Create result tensor
    Tensor result(output_shape, impl_->dtype(), impl_->storage()->device());

    // Sum over the specified dimension
    switch (impl_->dtype()) {
        case DType::Float32:
            sum_impl<float>(result, dim);
            break;
        default:
            throw std::runtime_error("Unsupported dtype for sum over dimension");
    }

    return result;
}

template<typename T>
void Tensor::sum_impl(Tensor& result, int64_t dim) const {
    if constexpr (std::is_same_v<T, float>) {
        if (device() == Device::CUDA) {
#ifdef USE_CUDA
            Tensor input_c = contiguous();
            const std::vector<int64_t>& in_shape = input_c.shape();
            const std::vector<int64_t>& out_shape = result.shape();
            int64_t ndim = static_cast<int64_t>(in_shape.size());
            if (ndim > 8) {
                throw std::runtime_error("CUDA sum supports up to 8 dimensions");
            }

            int64_t in_shape_h[8] = {};
            int64_t in_strides_h[8] = {};
            int64_t out_shape_h[8] = {};
            int64_t out_strides_h[8] = {};
            int64_t stride = 1;
            for (int64_t i = ndim - 1; i >= 0; --i) {
                in_shape_h[i] = in_shape[i];
                in_strides_h[i] = stride;
                stride *= in_shape[i];
            }
            int64_t ndim_out = ndim - 1;
            stride = 1;
            for (int64_t i = ndim_out - 1; i >= 0; --i) {
                out_shape_h[i] = out_shape[i];
                out_strides_h[i] = stride;
                stride *= out_shape[i];
            }

            size_t bytes_i = static_cast<size_t>(ndim) * sizeof(int64_t);
            size_t bytes_o = static_cast<size_t>(ndim_out) * sizeof(int64_t);
            int64_t *d_in_shape = nullptr, *d_in_strides = nullptr;
            int64_t *d_out_shape = nullptr, *d_out_strides = nullptr;
            cuda_check_error(cudaMalloc(&d_in_shape, bytes_i), "cudaMalloc d_in_shape");
            cuda_check_error(cudaMalloc(&d_in_strides, bytes_i), "cudaMalloc d_in_strides");
            cuda_check_error(cudaMalloc(&d_out_shape, bytes_o), "cudaMalloc d_out_shape");
            cuda_check_error(cudaMalloc(&d_out_strides, bytes_o), "cudaMalloc d_out_strides");
            cuda_check_error(cudaMemcpy(d_in_shape, in_shape_h, bytes_i, cudaMemcpyHostToDevice), "H2D d_in_shape");
            cuda_check_error(cudaMemcpy(d_in_strides, in_strides_h, bytes_i, cudaMemcpyHostToDevice), "H2D d_in_strides");
            cuda_check_error(cudaMemcpy(d_out_shape, out_shape_h, bytes_o, cudaMemcpyHostToDevice), "H2D d_out_shape");
            cuda_check_error(cudaMemcpy(d_out_strides, out_strides_h, bytes_o, cudaMemcpyHostToDevice), "H2D d_out_strides");

            cuda_fill(static_cast<float*>(result.data()), 0.0f, static_cast<int64_t>(result.numel()));
            cuda_sum_dim(static_cast<const float*>(input_c.data()),
                         static_cast<float*>(result.data()),
                         d_in_shape, d_in_strides,
                         d_out_shape, d_out_strides,
                         dim, ndim, static_cast<int64_t>(result.numel()), in_shape[dim]);
            cuda_check_error(cudaGetLastError(), "cuda_sum_dim failed");
            // No explicit sync: cudaFree (non-async) is itself a blocking call
            // that waits for all preceding GPU work before releasing memory.

            cuda_check_error(cudaFree(d_in_shape), "cudaFree d_in_shape");
            cuda_check_error(cudaFree(d_in_strides), "cudaFree d_in_strides");
            cuda_check_error(cudaFree(d_out_shape), "cudaFree d_out_shape");
            cuda_check_error(cudaFree(d_out_strides), "cudaFree d_out_strides");
            return;
#else
            throw std::runtime_error("CUDA not available");
#endif
        }
    }

    const std::vector<int64_t>& in_shape = shape();
    const std::vector<int64_t>& out_shape = result.shape();

    // Iterate through output indices and sum over the specified dimension
    std::vector<int64_t> out_indices(out_shape.size(), 0);

    for (size_t out_flat = 0; out_flat < result.numel(); ++out_flat) {
        // Convert flat output index to multi-dimensional indices
        size_t temp = out_flat;
        for (int64_t i = out_shape.size() - 1; i >= 0; --i) {
            out_indices[i] = temp % out_shape[i];
            temp /= out_shape[i];
        }

        // Reconstruct input indices by inserting the summed dimension
        std::vector<int64_t> in_indices = out_indices;
        in_indices.insert(in_indices.begin() + dim, 0);

        // Sum over the specified dimension
        T sum_val = 0;
        for (int64_t i = 0; i < in_shape[dim]; ++i) {
            in_indices[dim] = i;
            sum_val += at<T>(in_indices);
        }

        result.at<T>(out_indices) = sum_val;
    }
}

Tensor Tensor::broadcast(const std::vector<int64_t>& target_shape) const {
    // Compute broadcasted shape and verify compatibility
    std::vector<int64_t> current_shape = impl_->shape();
    size_t ndim = current_shape.size();
    size_t target_ndim = target_shape.size();
    
    // Verify compatibility (right-aligned comparison)
    for (size_t i = 0; i < target_ndim; ++i) {
        int64_t dim = (i < ndim) ? current_shape[ndim - 1 - i] : 1;
        int64_t target_dim = target_shape[target_ndim - 1 - i];
        
        if (dim != target_dim && dim != 1 && target_dim != 1) {
            throw runtime_error("Shapes are not broadcastable");
        }
    }
    
    // Compute broadcasted strides (right-aligned)
    std::vector<int64_t> new_strides(target_ndim);
    
    for (size_t i = 0; i < target_ndim; ++i) {
        // Right-align: compare from the right
        int64_t target_idx = target_ndim - 1 - i;
        int64_t current_idx = ndim - 1 - i;
        
        if (current_idx < 0 || current_shape[current_idx] == 1) {
            // Dimension is missing or size 1 - set stride to 0 (broadcasting)
            new_strides[target_idx] = 0;
        } else {
            // Keep original stride
            new_strides[target_idx] = impl_->strides()[current_idx];
        }
    }
    
    // Create a view with new shape and strides (zero-copy)
    auto new_impl = std::make_shared<TensorImpl>(
        impl_->storage(),
        target_shape,
        new_strides,
        impl_->dtype(),
        impl_->offset()
    );
    
    return Tensor(new_impl);
}

// ---------------------------------------------------------------------------
// Per-GradFn backward profiling (development/debugging only).
// ---------------------------------------------------------------------------
static bool g_grad_profile_enabled = false;
static std::map<std::string, double> g_grad_profile_times;

#ifdef __GNUC__
static std::string demangle_grad_name(const char* mangled) {
    int status = 0;
    char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
    std::string result = (status == 0 && demangled) ? demangled : mangled;
    if (demangled) free(demangled);
    // Strip the namespace/class prefix for brevity.
    size_t pos = result.rfind("::");
    if (pos != std::string::npos) result = result.substr(pos + 2);
    return result;
}
#else
static std::string demangle_grad_name(const char* mangled) { return mangled; }
#endif

void Tensor::enable_grad_profile(bool enable) {
    g_grad_profile_enabled = enable;
}

void Tensor::reset_grad_profile() {
    g_grad_profile_times.clear();
}

void Tensor::print_grad_profile() {
    if (g_grad_profile_times.empty()) return;
    std::vector<std::pair<std::string, double>> v(g_grad_profile_times.begin(),
                                                  g_grad_profile_times.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    double total = 0.0;
    for (const auto& kv : v) total += kv.second;
    std::cerr << "\n=== GradFn backward profile (synced GPU time) ===\n";
    std::cerr << std::left << std::setw(32) << "Op" << std::right
              << std::setw(12) << "ms" << std::setw(10) << "%" << "\n";
    for (const auto& kv : v) {
        double pct = (total > 0.0) ? (kv.second / total) * 100.0 : 0.0;
        std::cerr << std::left << std::setw(32) << kv.first << std::right
                  << std::fixed << std::setprecision(4) << std::setw(12) << kv.second
                  << std::setw(10) << std::setprecision(2) << pct << "\n";
    }
    std::cerr << std::left << std::setw(32) << "TOTAL" << std::right
              << std::setw(12) << std::fixed << std::setprecision(4) << total << "\n";
    std::cerr << "==================================================\n";
}

void Tensor::backward() {
    // Seed gradient is ones with the same shape as this tensor (d(loss)/d(loss) = 1).
    Tensor grad_output(impl_->shape(), impl_->dtype(), impl_->storage()->device());
    if (grad_output.device() == Device::CUDA) {
#ifdef USE_CUDA
        cuda_fill(static_cast<float*>(grad_output.data()), 1.0f,
                  static_cast<int64_t>(grad_output.numel()));
        cuda_check_error(cudaGetLastError(), "cuda_fill failed");
#else
        throw std::runtime_error("CUDA not supported in this build");
#endif
    } else {
        for (size_t i = 0; i < grad_output.numel(); ++i) {
            grad_output.at<float>(i) = 1.0f;
        }
    }

    // Call private implementation
    backward_impl(grad_output);
}

void Tensor::backward_impl(const Tensor& passedDownGrad){
    // ── Phase 1: BFS graph discovery + consumer (in-degree) counting ────────
    //
    // Each GradFn is the unique node that produced one tensor.  We use the raw
    // GradFn pointer as the node identity — it is stable for the lifetime of
    // the computation graph (kept alive by shared_ptr chains in the TensorImpls).
    //
    // in_degree[fn] = number of downstream nodes (consumers) that list fn as an
    // input and that also require grad.  A node is "ready" when this count
    // reaches 0, meaning every consumer has already accumulated its gradient.
    using GradFnPtr = GradFn*;

    std::unordered_map<GradFnPtr, int>    in_degree;
    std::unordered_set<GradFnPtr>         discovered;

    GradFnPtr root_fn = impl_->grad_fn().get();
    if (!root_fn) return;

    std::queue<GradFnPtr> bfs;
    bfs.push(root_fn);
    discovered.insert(root_fn);
    // Root has no consumers; default int value is 0 so the [] access is enough,
    // but be explicit for clarity.
    in_degree[root_fn] = 0;

    while (!bfs.empty()) {
        GradFnPtr fn = bfs.front();
        bfs.pop();

        for (const Tensor& inp : fn->inputs) {
            if (!inp.requiredGrad()) continue;
            GradFnPtr inp_fn = inp.gradFn().get();
            if (!inp_fn) continue;   // leaf — has no GradFn node to track

            // fn is a consumer of inp_fn, so increment inp_fn's in-degree.
            in_degree[inp_fn]++;

            // Only enqueue inp_fn for BFS traversal the first time we see it.
            if (discovered.find(inp_fn) == discovered.end()) {
                discovered.insert(inp_fn);
                bfs.push(inp_fn);
            }
        }
    }

    // ── Phase 2: Topological processing (Kahn's algorithm) ──────────────────
    //
    // node_grad[fn] accumulates all upstream gradients flowing into fn before
    // fn->backward() is called.  A node is enqueued in `ready` only when its
    // in_degree has been decremented to 0 by all of its consumers.
    std::unordered_map<GradFnPtr, Tensor> node_grad;
    node_grad.emplace(root_fn, passedDownGrad);

    std::queue<GradFnPtr> ready;
    ready.push(root_fn);   // root has in_degree == 0 already

    while (!ready.empty()) {
        GradFnPtr fn = ready.front();
        ready.pop();

        const Tensor& upstream = node_grad.at(fn);

        // ── Call this node's backward exactly once ─────────────────────────
        std::vector<Tensor> input_grads;
        if (g_grad_profile_enabled) {
            auto t0 = std::chrono::high_resolution_clock::now();
            input_grads = fn->backward(upstream);
#ifdef USE_CUDA
            cudaDeviceSynchronize();
#endif
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            g_grad_profile_times[demangle_grad_name(typeid(*fn).name())] += ms;
        } else {
            input_grads = fn->backward(upstream);
        }

        // Free the upstream gradient buffer — we no longer need it.
        node_grad.erase(fn);

        // ── Distribute gradients to each input ────────────────────────────
        for (size_t i = 0; i < input_grads.size() && i < fn->inputs.size(); ++i) {
            Tensor& inp = fn->inputs[i];
            if (!inp.requiredGrad()) continue;

            GradFnPtr inp_fn = inp.gradFn().get();

            if (!inp_fn) {
                // ── Leaf: accumulate directly into inp.grad ───────────────
                // input shares its TensorImpl with the original tensor, so
                // writing here propagates back to the leaf the user holds.
                if (!inp.impl_->grad()) {
                    inp.impl_->set_grad(std::make_shared<Tensor>(input_grads[i]));
                } else {
                    *inp.impl_->grad() = *inp.impl_->grad() + input_grads[i];
                }
            } else {
                // ── Non-leaf: accumulate into node_grad[inp_fn] ──────────
                // We do NOT call inp_fn->backward yet; we wait until all of
                // inp_fn's consumers have contributed (in_degree reaches 0).
                auto it = node_grad.find(inp_fn);
                if (it == node_grad.end()) {
                    node_grad.emplace(inp_fn, input_grads[i]);
                } else {
                    it->second = it->second + input_grads[i];
                }

                // Decrement consumer count; enqueue when all consumers done.
                if (--in_degree[inp_fn] == 0) {
                    ready.push(inp_fn);
                }
            }
        }
    }
}


template <typename T>
Tensor Tensor::softmax(int64_t dim) {
    // We MUST allocate 'out' with its own TensorImpl (independent of *this).
    // contiguous() returns *this when already contiguous (sharing impl_), so
    // calling out.setGradFn() would also overwrite *this's gradFn and store
    // *this in fn->inputs[0] creating a self-referential cycle that causes
    // backward_impl to loop forever.
    //
    // Instead: copy the logical elements from *this into a fresh row-major tensor,
    // run softmax in-place on that tensor, and attach SoftmaxBackward(inputs={*this})
    // (which holds the unmodified pre-softmax logits).
    const auto& sh = impl_->shape();
    int64_t ndim = (int64_t)sh.size();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim)
        throw std::runtime_error("Invalid softmax dim");

    // Allocate fresh row-major output.
    Tensor out(sh, impl_->dtype(), impl_->storage()->device());
    T* data = static_cast<T*>(out.data());
    // out.strides() are the standard row-major strides for sh.
    const auto& out_st = out.strides();

#ifdef USE_CUDA
    if (impl_->storage()->device() == Device::CUDA) {
        if (is_contiguous() && dim == ndim - 1) {
            // Fast GPU path: copy contiguous input to contiguous output and
            // launch a per-row softmax kernel.
            cuda_check_error(
                cudaMemcpy(out.data(), this->data(),
                           impl_->numel() * sizeof(T), cudaMemcpyDeviceToDevice),
                "cudaMemcpy in softmax");

            int64_t dim_size = sh[dim];
            int64_t outer_size = static_cast<int64_t>(out.numel()) / dim_size;
            cuda_softmax_forward(static_cast<const float*>(out.data()),
                                 static_cast<float*>(out.data()),
                                 outer_size, dim_size);
            cuda_check_error(cudaGetLastError(), "cuda_softmax_forward failed");
            // No sync: result stays on GPU; next kernel serialized on default stream.

            if (requiredGrad()) {
                auto fn = std::make_shared<SoftmaxBackward>();
                fn->inputs = {*this};
                fn->dim = dim;
                out.impl_->set_grad_fn(fn);
                out.impl_->set_requires_grad(true);
            }
            return out;
        }
        throw std::runtime_error("CUDA softmax currently requires contiguous input and dim == last dimension");
    }
#endif

    // Copy *this into out, element by element, using *this's strides for reading
    // and out's (row-major) strides for writing.
    {
        const T*   src      = static_cast<const T*>(impl_->storage()->data());
        const auto& src_st  = impl_->strides();
        size_t     base_off = impl_->offset();
        std::vector<int64_t> cidx(ndim, 0);
        for (size_t flat = 0; flat < impl_->numel(); ++flat) {
            // source element
            size_t src_off = base_off;
            for (int d = 0; d < ndim; ++d) src_off += cidx[d] * src_st[d];
            // destination: row-major flat index (same multi-index, contiguous layout)
            size_t dst_off = 0;
            for (int d = 0; d < ndim; ++d) dst_off += cidx[d] * out_st[d];
            data[dst_off] = src[src_off];
            for (int d = ndim - 1; d >= 0; --d) {
                if (++cidx[d] < sh[d]) break;
                cidx[d] = 0;
            }
        }
    }

    int64_t dim_size   = sh[dim];
    int64_t outer_size = (int64_t)out.numel() / dim_size;

    // We iterate over all "outer" indices (all dims except `dim`)
    std::vector<int64_t> idx(ndim, 0);

    for (int64_t outer = 0; outer < outer_size; ++outer) {

        // decode outer index into multidimensional index (except dim)
        int64_t tmp = outer;
        for (int i = ndim - 1; i >= 0; --i) {
            if (i == dim) continue;
            idx[i] = tmp % sh[i];
            tmp /= sh[i];
        }

        // Step 1: find max (numerical stability)
        T max_val = -std::numeric_limits<T>::infinity();
        for (int64_t i = 0; i < dim_size; ++i) {
            idx[dim] = i;
            int64_t offset = 0;
            for (int d = 0; d < ndim; ++d) offset += idx[d] * out_st[d];
            max_val = std::max(max_val, data[offset]);
        }

        // Step 2: exp + sum
        T sum = 0;
        for (int64_t i = 0; i < dim_size; ++i) {
            idx[dim] = i;
            int64_t offset = 0;
            for (int d = 0; d < ndim; ++d) offset += idx[d] * out_st[d];
            T val = std::exp(data[offset] - max_val);
            data[offset] = val;
            sum += val;
        }

        // Step 3: normalize
        for (int64_t i = 0; i < dim_size; ++i) {
            idx[dim] = i;
            int64_t offset = 0;
            for (int d = 0; d < ndim; ++d) offset += idx[d] * out_st[d];
            data[offset] /= sum;
        }
    }

    // Attach SoftmaxBackward.  inputs[0] = *this (pre-softmax logits, unchanged).
    // out.impl_ is distinct from this->impl_, so this does NOT touch *this's gradFn.
    if (requiredGrad()) {
        auto fn = std::make_shared<SoftmaxBackward>();
        fn->inputs = {*this};
        fn->dim = dim;
        out.impl_->set_grad_fn(fn);
        out.impl_->set_requires_grad(true);
    }

    return out;
}

Tensor Tensor::softmax(int64_t dim) {
    switch (impl_->dtype()) {
        case DType::Float32:
            return softmax<float>(dim);
        default:
            throw std::runtime_error("Softmax only supports float types");
    }
}


Tensor Tensor::toDevice(Device targetDevice) const {
    if (device() == targetDevice) {
        return *this;
    }

    Tensor result(shape(), dtype(), targetDevice);

    if (device() == Device::CPU && targetDevice == Device::CUDA) {
#ifdef USE_CUDA
        cuda_check_error(cudaMemcpy(result.data(), data(), numel() * dtype_size(dtype()), cudaMemcpyHostToDevice), "cudaMemcpy H2D failed");
#else
        throw std::runtime_error("CUDA not supported in this build");
#endif
    } else if (device() == Device::CUDA && targetDevice == Device::CPU) {
#ifdef USE_CUDA
        cuda_check_error(cudaMemcpy(result.data(), data(), numel() * dtype_size(dtype()), cudaMemcpyDeviceToHost), "cudaMemcpy D2H failed");
#else
        throw std::runtime_error("CUDA not supported in this build");
#endif
    } else {
        throw std::runtime_error("Unsupported device transfer");
    }

    return result;
}