#include "tensor.h"
#include "tensor_impl.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <optional>
#include <stack>


size_t Tensor::numel() const {return impl_->numel();}

Device Tensor::device() const {
    return impl_->storage()->device();
}

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
    return Tensor(newImpl);
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