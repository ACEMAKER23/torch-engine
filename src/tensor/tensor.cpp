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
#include <stack>

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

    std::memcpy(storage->data(), impl_->storage()->data(),bt);


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
        std::vector<int64_t> broadcasted_shape = broadcast_to_shape<float>(shape(), other.shape());
        Tensor result(broadcasted_shape, dtype(), Device::CUDA);
        dispatch_cuda_binary_op(*this, other, result, [](auto a, auto b, auto out, int64_t size) {
            cuda_mul(a, b, out, size);
        });
        cuda_check_error(cudaGetLastError(), "CUDA mul failed");
        return result;
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
        std::vector<int64_t> broadcasted_shape = broadcast_to_shape<float>(shape(), other.shape());
        Tensor result(broadcasted_shape, dtype(), Device::CUDA);
        dispatch_cuda_binary_op(*this, other, result, [](auto a, auto b, auto out, int64_t size) {
            cuda_add(a, b, out, size);
        });
        cuda_check_error(cudaGetLastError(), "CUDA add failed");
        return result;
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
        std::vector<int64_t> broadcasted_shape = broadcast_to_shape<float>(shape(), other.shape());
        Tensor result(broadcasted_shape, dtype(), Device::CUDA);
        dispatch_cuda_binary_op(*this, other, result, [](auto a, auto b, auto out, int64_t size) {
            cuda_sub(a, b, out, size);
        });
        cuda_check_error(cudaGetLastError(), "CUDA sub failed");
        return result;
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
        std::vector<int64_t> broadcasted_shape = broadcast_to_shape<float>(shape(), other.shape());
        Tensor result(broadcasted_shape, dtype(), Device::CUDA);
        dispatch_cuda_binary_op(*this, other, result, [](auto a, auto b, auto out, int64_t size) {
            cuda_div(a, b, out, size);
        });
        cuda_check_error(cudaGetLastError(), "CUDA div failed");
        return result;
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
    Tensor result = clone();
    switch (impl_->dtype()) {
        case DType::Float32:
            for (size_t i = 0; i < result.numel(); ++i) {
                result.at<float>(i) = -result.at<float>(i);
            }
            break;
        case DType::Int32:
            for (size_t i = 0; i < result.numel(); ++i) {
                result.at<int32_t>(i) = -result.at<int32_t>(i);
            }
            break;
        case DType::Int64:
            for (size_t i = 0; i < result.numel(); ++i) {
                result.at<int64_t>(i) = -result.at<int64_t>(i);
            }
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

void Tensor::backward() {
    // Seed gradient is ones with the same shape as this tensor (d(loss)/d(loss) = 1).
    Tensor grad_output(impl_->shape(), impl_->dtype(), impl_->storage()->device());
    for (size_t i = 0; i < grad_output.numel(); ++i) {
        grad_output.at<float>(i) = 1.0f;
    }

    // Call private implementation
    backward_impl(grad_output);
}

void Tensor::backward_impl(const Tensor& passedDownGrad){
    stack<pair<Tensor,Tensor>> currentPre;
    currentPre.push({*this,passedDownGrad});

    while (!currentPre.empty()){
        auto current = currentPre.top();
        //gradient of all the input to the current tensor
        currentPre.pop();

        if (!current.first.gradFn()) continue;
        auto inputGradent=current.first.gradFn()->backward(current.second);

        for(size_t i=0; i<inputGradent.size(); ++i){
            // input shares its TensorImpl (data + autograd state) with the
            // original tensor, so writing the gradient here propagates back to
            // the leaf the user holds.
            Tensor& input = current.first.gradFn()->inputs[i];

            if (input.requiredGrad() && !input.gradFn()){  // Leaf node requiring grad
                if (!input.impl_->grad()) {
                    input.impl_->set_grad(make_shared<Tensor>(inputGradent[i]));
                } else {
                    // Accumulate if gradient already exists
                    *input.impl_->grad() = *input.impl_->grad() + inputGradent[i];
                }
            }
            else if (input.requiredGrad()){  // Non-leaf, continue traversal
                currentPre.push({input,inputGradent[i]});
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


Tensor Tensor::toDevice(Device targetDevice){
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