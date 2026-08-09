#ifndef TENSOR
#define TENSOR

#include "tensor_impl.h"
#include <memory>
#include <functional>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include "../core/grad_fn.h"

#ifdef USE_CUDA
#include "../core/cuda_utils.h"
#include <cuda_runtime.h>
#include <type_traits>

// Forward declarations of CUDA elementwise helpers used by Tensor methods.
// (Full declarations live in ../cuda/elementwise_cuda.h to avoid a circular
// include chain through dtype_utils.h.)
void cuda_fill(float* data, float value, int64_t size);
void cuda_reduce_sum(const float* input, float* output, int64_t size);
void cuda_sum_dim(const float* input, float* output,
                  const int64_t* in_shape, const int64_t* in_strides,
                  const int64_t* out_shape, const int64_t* out_strides,
                  int64_t dim, int64_t ndim, int64_t out_numel, int64_t dim_size);
#endif

// Forward declarations for low-precision types (defined in dtype_utils.h)
struct float16_t;
struct bfloat16_t;

class Tensor {
    friend class TensorTest_BasicConstruction_Test;
    friend class TensorTest_CopySemantics_Test;
    friend class TensorTest_MoveSemantics_Test;
    friend class TensorTest_CloneDeepCopy_Test;
    friend class TensorTest_ViewReshape_Test;
    friend class TensorTest_SliceBasic_Test;
    friend class TensorTest_NestedSlice_Test;
    friend class TensorTest_ElementwiseAddFloat32_Test;
    friend class TensorTest_ElementwiseAddInt32_Test;
    friend class TensorTest_ElementwiseAddInt64_Test;
    friend class TensorTest_ElementwiseSubFloat32_Test;
    friend class TensorTest_ElementwiseMulFloat32_Test;
    friend class TensorTest_ElementwiseDivFloat32_Test;
    friend class TensorTest_ElementwiseShapeMismatch_Test;
    friend class TensorTest_ElementwiseDtypeMismatch_Test;
    friend class TensorTest_ReductionSumFloat32_Test;
    friend class TensorTest_ReductionMeanFloat32_Test;
    friend class TensorTest_ReductionMaxFloat32_Test;
    friend class TensorTest_ReductionMinFloat32_Test;
    friend class TensorTest_AtFlatIndexFloat32_Test;
    friend class TensorTest_AtMultiIndexFloat32_Test;
    friend class TensorTest_AtTypeMismatch_Test;
    friend class TensorTest_Transpose2D_Test;
    friend class TensorTest_TransposeZeroCopy_Test;
    friend class TensorTest_TransposeInvalid_Test;
    friend class TensorTest_ActivationRelu_Test;
    friend class TensorTest_ActivationReluInplace_Test;
    friend class TensorTest_ActivationGelu_Test;
    friend class TensorTest_ActivationSigmoid_Test;
    friend class TensorTest_ActivationSqrt_Test;
    friend class TensorTest_ActivationExp_Test;
    friend class TensorTest_ActivationLog_Test;
    friend class TensorTest_ActivationPow_Test;
    friend class TensorTest_MatmulFloat32_Test;
    friend class TensorTest_MatmulInt32_Test;
    friend class TensorTest_MatmulDimensionMismatch_Test;
    friend class TensorTest_Matmul3D_Test;
    friend class TensorTest_Matmul4D_Test;
    friend class TensorTest_MatmulBroadcast_Test;
    friend class TensorTest_MatmulBroadcastIncompatible_Test;
    friend class TensorTest_ElementwiseBroadcast_Test;
    friend class TensorTest_ElementwiseBroadcastIncompatible_Test;
public:
    Tensor(std::vector<int64_t> shape, DType dtype, Device deviceT);

    size_t numel() const;

    Device device() const;

    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;


    Tensor clone() const;
    bool is_contiguous() const;
    Tensor contiguous() const;

    Tensor view(const vector<int64_t>& newShape) const;
    void transpose();
    void transpose(size_t d1, size_t d2);
    Tensor transpose_view(size_t d1, size_t d2) const;
    Tensor broadcast(const std::vector<int64_t>& target_shape) const;

    Tensor slice(int64_t start,int64_t finish,int64_t dimension);

    const std::vector<int64_t>& shape() const;
    const std::vector<int64_t>& strides() const;
    DType dtype() const {return impl_->dtype();}
    
    // PyTorch-style data access
    void* data();
    const void* data() const;

    //Operator Overloads//
    Tensor operator*(const Tensor& other) const;
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    //Negation
    Tensor operator-() const;
    //--------------//

    bool requiredGrad() const {return impl_->requires_grad();};
    std::shared_ptr<Tensor> grad() const {return impl_->grad();};
    std::shared_ptr<GradFn> gradFn() const {return impl_->grad_fn();};
    void setRequiresGrad(bool value) {impl_->set_requires_grad(value);};
    void setGradFn(std::shared_ptr<GradFn> fn) {impl_->set_grad_fn(fn);};

    Tensor relu() const;
    Tensor gelu() const;
    Tensor sigmoid() const;
    Tensor sqrt() const;
    Tensor exp() const;
    Tensor log() const;
    
    template<typename T>
    Tensor softmax(int64_t dim);
    Tensor softmax(int64_t dim);

    Tensor toDevice(Device targetDevice) const;

    // Template implementations (must be in header)
    template<typename T>
    static std::vector<int64_t> broadcast_to_shape(const std::vector<int64_t>& shape,
                                                     const std::vector<int64_t>& target_shape) {
        // Compute broadcasted shape from shape to target_shape
        // Returns the broadcasted shape (should equal target_shape if compatible)
        size_t ndim = shape.size();
        size_t target_ndim = target_shape.size();
        size_t max_ndim = std::max(ndim, target_ndim);
        
        std::vector<int64_t> result(max_ndim);
        
        for (size_t i = 0; i < max_ndim; ++i) {
            int64_t dim = (i < ndim) ? shape[ndim - 1 - i] : 1;
            int64_t target_dim = (i < target_ndim) ? target_shape[target_ndim - 1 - i] : 1;
            
            if (dim != target_dim && dim != 1 && target_dim != 1) {
                throw runtime_error("Shapes are not broadcastable");
            };
            
            result[max_ndim - 1 - i] = std::max(dim, target_dim);
        }
        
        return result;
    }
    
    template<typename T>
    static Tensor elementwise_operation(const Tensor& a, const Tensor& b,
                                         std::function<T(T, T)> op) {
        // Compute broadcasted shape
        std::vector<int64_t> shape_a = a.shape();
        std::vector<int64_t> shape_b = b.shape();
        std::vector<int64_t> broadcasted_shape = broadcast_to_shape<T>(shape_a, shape_b);
        
        // Broadcast both tensors to the same shape (zero-copy)
        Tensor a_broadcasted = a.broadcast(broadcasted_shape);
        Tensor b_broadcasted = b.broadcast(broadcasted_shape);
        
        // Create result tensor with broadcasted shape
        Tensor result(broadcasted_shape, a.dtype(), a.impl_->storage()->device());
        
        // Iterate using multi-dimensional indices to handle strides correctly
        size_t num_elements = 1;
        for (int64_t dim : broadcasted_shape) num_elements *= dim;
        
        for (size_t flat_idx = 0; flat_idx < num_elements; ++flat_idx) {
            // Convert flat index to multi-dimensional indices
            std::vector<int64_t> indices;
            size_t temp = flat_idx;
            for (int i = broadcasted_shape.size() - 1; i >= 0; --i) {
                indices.insert(indices.begin(), temp % broadcasted_shape[i]);
                temp /= broadcasted_shape[i];
            }
            
            // Access using stride-aware indexing (handles stride 0)
            result.at<T>(indices) = op(a_broadcasted.at<T>(indices), b_broadcasted.at<T>(indices));
        }
        
        return result;
    }
    
    template<typename T>
    T& at(size_t flatIndex){
        if constexpr (std::is_same_v<T, float>) {
            if (impl_->dtype() != DType::Float32) {
                throw std::runtime_error("Cannot access Float32 data from non-Float32 tensor");
            }
        } else if constexpr (std::is_same_v<T, int32_t>) {
            if (impl_->dtype() != DType::Int32) {
                throw std::runtime_error("Cannot access Int32 data from non-Int32 tensor");
            }
        } else if constexpr (std::is_same_v<T, int64_t>) {
            if (impl_->dtype() != DType::Int64) {
                throw std::runtime_error("Cannot access Int64 data from non-Int64 tensor");
            }
        } else if constexpr (std::is_same_v<T, float16_t>) {
            if (impl_->dtype() != DType::Float16) {
                throw std::runtime_error("Cannot access Float16 data from non-Float16 tensor");
            }
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            if (impl_->dtype() != DType::BFloat16) {
                throw std::runtime_error("Cannot access BFloat16 data from non-BFloat16 tensor");
            }
        } else {
            static_assert(!std::is_same_v<T, T>, "Unsupported type for at()");
        }

        auto* data = static_cast<T*>(impl_->storage()->data());
        return data[flatIndex + impl_->offset()];
    }

    template<typename T>
    T at(size_t flatIndex) const {
        if constexpr (std::is_same_v<T, float>) {
            if (impl_->dtype() != DType::Float32) {
                throw std::runtime_error("Cannot access Float32 data from non-Float32 tensor");
            }
        } else if constexpr (std::is_same_v<T, int32_t>) {
            if (impl_->dtype() != DType::Int32) {
                throw std::runtime_error("Cannot access Int32 data from non-Int32 tensor");
            }
        } else if constexpr (std::is_same_v<T, int64_t>) {
            if (impl_->dtype() != DType::Int64) {
                throw std::runtime_error("Cannot access Int64 data from non-Int64 tensor");
            }
        } else if constexpr (std::is_same_v<T, float16_t>) {
            if (impl_->dtype() != DType::Float16) {
                throw std::runtime_error("Cannot access Float16 data from non-Float16 tensor");
            }
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            if (impl_->dtype() != DType::BFloat16) {
                throw std::runtime_error("Cannot access BFloat16 data from non-BFloat16 tensor");
            }
        } else {
            static_assert(!std::is_same_v<T, T>, "Unsupported type for at()");
        }

        auto* data = static_cast<const T*>(impl_->storage()->data());
        return data[flatIndex + impl_->offset()];
    }

    template<typename T>
    T& at(const std::vector<int64_t>& indices){
        if constexpr (std::is_same_v<T, float>) {
            if (impl_->dtype() != DType::Float32) {
                throw std::runtime_error("Cannot access Float32 data from non-Float32 tensor");
            }
        } else if constexpr (std::is_same_v<T, int32_t>) {
            if (impl_->dtype() != DType::Int32) {
                throw std::runtime_error("Cannot access Int32 data from non-Int32 tensor");
            }
        } else if constexpr (std::is_same_v<T, int64_t>) {
            if (impl_->dtype() != DType::Int64) {
                throw std::runtime_error("Cannot access Int64 data from non-Int64 tensor");
            }
        } else if constexpr (std::is_same_v<T, float16_t>) {
            if (impl_->dtype() != DType::Float16) {
                throw std::runtime_error("Cannot access Float16 data from non-Float16 tensor");
            }
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            if (impl_->dtype() != DType::BFloat16) {
                throw std::runtime_error("Cannot access BFloat16 data from non-BFloat16 tensor");
            }
        } else {
            static_assert(!std::is_same_v<T, T>, "Unsupported type for at()");
        }

        auto* data = static_cast<T*>(impl_->storage()->data());
        size_t flatIndex = 0;
        for(size_t i = 0; i < indices.size(); ++i) {
            flatIndex += indices[i] * impl_->strides()[i];
        }
        return data[flatIndex + impl_->offset()];
    }

    template<typename T>
    T at(const std::vector<int64_t>& indices) const {
        if constexpr (std::is_same_v<T, float>) {
            if (impl_->dtype() != DType::Float32) {
                throw std::runtime_error("Cannot access Float32 data from non-Float32 tensor");
            }
        } else if constexpr (std::is_same_v<T, int32_t>) {
            if (impl_->dtype() != DType::Int32) {
                throw std::runtime_error("Cannot access Int32 data from non-Int32 tensor");
            }
        } else if constexpr (std::is_same_v<T, int64_t>) {
            if (impl_->dtype() != DType::Int64) {
                throw std::runtime_error("Cannot access Int64 data from non-Int64 tensor");
            }
        } else if constexpr (std::is_same_v<T, float16_t>) {
            if (impl_->dtype() != DType::Float16) {
                throw std::runtime_error("Cannot access Float16 data from non-Float16 tensor");
            }
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            if (impl_->dtype() != DType::BFloat16) {
                throw std::runtime_error("Cannot access BFloat16 data from non-BFloat16 tensor");
            }
        } else {
            static_assert(!std::is_same_v<T, T>, "Unsupported type for at()");
        }

        auto* data = static_cast<const T*>(impl_->storage()->data());
        size_t flatIndex = 0;
        for(size_t i = 0; i < indices.size(); ++i) {
            flatIndex += indices[i] * impl_->strides()[i];
        }
        return data[flatIndex + impl_->offset()];
    }

    template<typename T>
    T sum() {
        if constexpr (std::is_same_v<T, float>) {
            if (device() == Device::CUDA) {
#ifdef USE_CUDA
                Tensor d_scalar({1}, dtype(), Device::CUDA);
                cuda_fill(static_cast<float*>(d_scalar.data()), 0.0f, 1);
                cuda_reduce_sum(static_cast<const float*>(data()),
                                static_cast<float*>(d_scalar.data()),
                                static_cast<int64_t>(numel()));
                cuda_check_error(cudaGetLastError(), "cuda_reduce_sum failed");
                cuda_check_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize after reduce_sum");
                float h_scalar = 0.0f;
                cuda_check_error(
                    cudaMemcpy(&h_scalar, d_scalar.data(), sizeof(float), cudaMemcpyDeviceToHost),
                    "cudaMemcpy reduce_sum D2H");
                return h_scalar;
#else
                throw std::runtime_error("CUDA not available");
#endif
            }
        }
        T result = 0;
        for (size_t i = 0; i < numel(); ++i) {
            result += at<T>(i);
        }
        return result;
    }

    // Sum over a specific dimension, reducing the rank by 1
    Tensor sum(int64_t dim) const;

    template<typename T>
    T mean() {
        return (sum<T>() / static_cast<T>(numel()));
    }

    template<typename T>
    T max() {
        T largest = at<T>(0);
        for (size_t i = 0; i < numel(); ++i) {
            largest = std::max(largest, at<T>(i));
        }
        return largest;
    }

    template<typename T>
    T min() {
        T smallest = at<T>(0);
        for (size_t i = 0; i < numel(); ++i) {
            smallest = std::min(smallest, at<T>(i));
        }
        return smallest;
    }

    template<typename T>
    void relu_() {
        if constexpr (std::is_same_v<T, float>) {
            if (impl_->dtype() != DType::Float32) {
                throw std::runtime_error("Cannot access Float32 data from non-Float32 tensor");
            }
        } else if constexpr (std::is_same_v<T, int32_t>) {
            if (impl_->dtype() != DType::Int32) {
                throw std::runtime_error("Cannot access Int32 data from non-Int32 tensor");
            }
        } else if constexpr (std::is_same_v<T, int64_t>) {
            if (impl_->dtype() != DType::Int64) {
                throw std::runtime_error("Cannot access Int64 data from non-Int64 tensor");
            }
        } else {
            static_assert(!std::is_same_v<T, T>, "Unsupported type for relu_()");
        }
        
        auto* data = static_cast<T*>(impl_->storage()->data());
        for (size_t i = 0; i < numel(); ++i) {
            size_t idx = impl_->offset() + i;
            data[idx] = (data[idx] > 0) ? data[idx] : 0;
        }
    }

    template<typename T>
    void gelu_() {
        if constexpr (std::is_same_v<T, float>) {
            if (impl_->dtype() != DType::Float32) {
                throw std::runtime_error("Cannot access Float32 data from non-Float32 tensor");
            }
        } else if constexpr (std::is_same_v<T, int32_t>) {
            if (impl_->dtype() != DType::Int32) {
                throw std::runtime_error("Cannot access Int32 data from non-Int32 tensor");
            }
        } else if constexpr (std::is_same_v<T, int64_t>) {
            if (impl_->dtype() != DType::Int64) {
                throw std::runtime_error("Cannot access Int64 data from non-Int64 tensor");
            }
        } else {
            static_assert(!std::is_same_v<T, T>, "Unsupported type for gelu_()");
        }
        
        auto* data = static_cast<T*>(impl_->storage()->data());
        for (size_t i = 0; i < numel(); i++) {
            size_t idx = impl_->offset() + i;
            double x = static_cast<double>(data[idx]);
            double x3 = x * x * x;
            data[idx] = static_cast<T>(
                0.5 * x *
                (1.0 + std::tanh(
                    0.7978845608 * (x + 0.044715 * x3)
                ))
            );
        }
    }

    template<typename T>
    void sigmoid_() {
        auto* data = static_cast<T*>(impl_->storage()->data());
        for (size_t i = 0; i < numel(); i++) {
            size_t idx = impl_->offset() + i;
            double x = static_cast<double>(data[idx]);
            data[idx] = static_cast<T>(1.0 / (1.0 + std::exp(-x)));
        }
    }

    template<typename T>
    void sqrt_() {
        auto* data = static_cast<T*>(impl_->storage()->data());
        for (size_t i = 0; i < numel(); i++) {
            size_t idx = impl_->offset() + i;
            data[idx] = static_cast<T>(std::sqrt(static_cast<double>(data[idx])));
        }
    }

    template<typename T>
    void exp_() {
        auto* data = static_cast<T*>(impl_->storage()->data());
        for (size_t i = 0; i < numel(); i++) {
            size_t idx = impl_->offset() + i;
            data[idx] = static_cast<T>(std::exp(static_cast<double>(data[idx])));
        }
    }

    template<typename T>
    void log_() {
        auto* data = static_cast<T*>(impl_->storage()->data());
        for (size_t i = 0; i < numel(); i++) {
            size_t idx = impl_->offset() + i;
            data[idx] = static_cast<T>(std::log(static_cast<double>(data[idx])));
        }
    }

    template<typename T>
    void fill_(T value) {
        auto* data = static_cast<T*>(impl_->storage()->data());
        for (size_t i = 0; i < numel(); i++) {
            size_t idx = impl_->offset() + i;
            data[idx] = value;
        }
    }

    template<typename T>
    void pow_(T exponent) {
        auto* data = static_cast<T*>(impl_->storage()->data());
        for (size_t i = 0; i < numel(); i++) {
            size_t idx = impl_->offset() + i;
            data[idx] = static_cast<T>(std::pow(static_cast<double>(data[idx]), static_cast<double>(exponent)));
        }
    }

    template<typename T>
    Tensor pow(T exponent) {
        Tensor result = clone();
        result.pow_<T>(exponent);
        return result;
    }

    template<typename T>
    static Tensor matmul_impl(const Tensor& a, const Tensor& b) {
        // Handle N-dimensional tensors (last 2 dims are matrix dimensions)
        size_t ndim_a = a.shape().size();
        size_t ndim_b = b.shape().size();
        
        // Validate dimensions
        if (ndim_a < 2 || ndim_b < 2) {
            throw runtime_error("Matmul requires tensors with at least 2 dimensions");
        }
        
        // Get matrix dimensions (last 2 dims)
        int64_t M = a.shape()[ndim_a - 2];
        int64_t K_a = a.shape()[ndim_a - 1];
        int64_t K_b = b.shape()[ndim_b - 2];
        int64_t N = b.shape()[ndim_b - 1];
        
        if (K_a != K_b) {
            throw runtime_error("Inner dimensions must match for matmul");
        }
        
        // Compute batch dimensions (all dims except last 2)
        std::vector<int64_t> batch_shape_a(a.shape().begin(), a.shape().end() - 2);
        std::vector<int64_t> batch_shape_b(b.shape().begin(), b.shape().end() - 2);
        
        // Broadcast batch dimensions
        std::vector<int64_t> batch_shape = broadcast_shapes(batch_shape_a, batch_shape_b);
        
        // Compute output shape
        std::vector<int64_t> output_shape = batch_shape;
        output_shape.push_back(M);
        output_shape.push_back(N);
        
        Tensor result(output_shape, a.dtype(), a.impl_->storage()->device());

#ifdef USE_CUDA
        if (a.device() == Device::CUDA) {
            if constexpr (std::is_same_v<T, float>) {
                cuda_matmul_impl_f32(a, b, result, M, K_a, N, batch_shape);
                return result;
            }
            throw runtime_error("CUDA matmul currently only supports Float32");
        }
#endif

        // Zero-initialise via raw pointer -- at<T>() chases three shared_ptr levels
        // on every element access and costs ~45 µs for a 2048-element tensor vs
        // <1 µs for a plain memset.
        std::memset(result.data(), 0, result.numel() * sizeof(T));
        
        // Block size for tiled algorithm (cache-friendly)
        const int64_t BLOCK_SIZE = 64;
        
        // Iterate over batch dimensions
        if (batch_shape.empty()) {
            // 2D case
            tiled_matmul_2d<T>(a, b, result, 0, 0, 0, M, K_a, N, BLOCK_SIZE);
        } else {
            // N-dimensional case with batches
            size_t num_batches = 1;
            for (int64_t dim : batch_shape) num_batches *= dim;
            
            // Pre-allocate once, reused every iteration to avoid per-batch heap allocs.
            std::vector<int64_t> batch_indices_multi(batch_shape.size());
            for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
                // Convert flat batch index to multi-dimensional indices (right-to-left)
                size_t temp_idx = batch_idx;
                for (int i = (int)batch_shape.size() - 1; i >= 0; --i) {
                    batch_indices_multi[i] = temp_idx % batch_shape[i];
                    temp_idx /= batch_shape[i];
                }
                
                // Compute flat offsets for batch slices (broadcasting handled in compute_batch_offset)
                size_t offset_a = compute_batch_offset(a, batch_indices_multi);
                size_t offset_b = compute_batch_offset(b, batch_indices_multi);
                size_t offset_result = compute_batch_offset(result, batch_indices_multi);
                
                // Perform tiled matmul on this batch
                tiled_matmul_2d<T>(a, b, result, offset_a, offset_b, offset_result, M, K_a, N, BLOCK_SIZE);
            }
        }
        
        return result;
    }
    
    static void cuda_matmul_impl_f32(const Tensor& a, const Tensor& b, Tensor& result,
                                     int64_t M, int64_t K, int64_t N,
                                     const std::vector<int64_t>& batch_shape) {
#ifdef USE_CUDA
        // cuBLAS is column-major.  We compute row-major C = A * B via the identity
        // C^T = B^T * A^T, swapping the operands to cuBLAS.  Each 2D slice is
        // either contiguous row-major (last-dim stride 1) or contiguous column-major
        // (first-dim stride 1).  For row-major [R, C] we pass op = N with ld = R;
        // for column-major we pass op = T with ld = C.
        auto cublas_params = [](int64_t s_row, int64_t s_col) -> std::pair<cublasOperation_t, int> {
            if (s_col == 1) return {CUBLAS_OP_N, static_cast<int>(s_row)};   // row-major [R, C]
            if (s_row == 1) return {CUBLAS_OP_T, static_cast<int>(s_col)};   // col-major [R, C]
            throw std::runtime_error("CUDA matmul only supports row- or column-major 2D slices");
        };

        size_t ndim_a = a.shape().size();
        size_t ndim_b = b.shape().size();

        int64_t s_row_a = a.strides()[ndim_a - 2];
        int64_t s_col_a = a.strides()[ndim_a - 1];
        int64_t s_row_b = b.strides()[ndim_b - 2];
        int64_t s_col_b = b.strides()[ndim_b - 1];

        // first cuBLAS arg is our B, interpreted as [N, K]
        auto paramsA = cublas_params(s_row_b, s_col_b);
        cublasOperation_t transA = paramsA.first;
        int lda = paramsA.second;
        // second cuBLAS arg is our A, interpreted as [K, M]
        auto paramsB = cublas_params(s_row_a, s_col_a);
        cublasOperation_t transB = paramsB.first;
        int ldb = paramsB.second;

        const float* A_ptr = static_cast<const float*>(a.data()); // our A
        const float* B_ptr = static_cast<const float*>(b.data()); // our B
        float* C_ptr = static_cast<float*>(result.data());

        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasHandle_t handle = cuda_cublas_handle();

        auto do_gemm = [&](const float* A_batch, const float* B_batch, float* C_batch) {
            // C^T = B^T * A^T  ->  C is N x M in column-major (ld = N), which is
            // exactly the storage layout of the row-major [M, N] result tensor.
            cublas_check_error(
                cublasSgemm(handle,
                            transA, transB,
                            static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                            &alpha,
                            B_batch, lda,
                            A_batch, ldb,
                            &beta,
                            C_batch, static_cast<int>(N)),
                "cublasSgemm");
        };

        if (batch_shape.empty()) {
            do_gemm(A_ptr, B_ptr, C_ptr);
        } else {
            size_t num_batches = 1;
            for (int64_t dim : batch_shape) num_batches *= dim;

            std::vector<int64_t> batch_indices_multi(batch_shape.size());
            for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
                size_t temp_idx = batch_idx;
                for (int i = (int)batch_shape.size() - 1; i >= 0; --i) {
                    batch_indices_multi[i] = temp_idx % batch_shape[i];
                    temp_idx /= batch_shape[i];
                }

                size_t offset_a = compute_batch_offset(a, batch_indices_multi);
                size_t offset_b = compute_batch_offset(b, batch_indices_multi);
                size_t offset_result = compute_batch_offset(result, batch_indices_multi);

                do_gemm(A_ptr + offset_a, B_ptr + offset_b, C_ptr + offset_result);
            }
        }
        cuda_check_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize after cublasSgemm");
#else
        (void)a; (void)b; (void)result; (void)M; (void)K; (void)N; (void)batch_shape;
        throw std::runtime_error("CUDA not built into this binary");
#endif
    }

    template<typename T>
    static void tiled_matmul_2d(const Tensor& a, const Tensor& b, Tensor& c,
                                size_t offset_a, size_t offset_b, size_t offset_c,
                                int64_t M, int64_t K, int64_t N, int64_t block_size) {
        const T* A = static_cast<const T*>(a.data()) + offset_a;
        const T* B = static_cast<const T*>(b.data()) + offset_b;
        T* C = static_cast<T*>(c.data()) + offset_c;
        
        // Use actual strides from the last 2 dimensions so that transposed
        // and other non-contiguous views are handled correctly without copying.
        size_t ndim_a = a.shape().size();
        size_t ndim_b = b.shape().size();

        int64_t lda     = a.strides()[ndim_a - 2]; // row stride of A
        int64_t inner_a = a.strides()[ndim_a - 1]; // col stride of A (1 when row-major)
        int64_t ldb     = b.strides()[ndim_b - 2]; // row stride of B
        int64_t inner_b = b.strides()[ndim_b - 1]; // col stride of B (1 when row-major)
        // Result c is always freshly allocated with contiguous (row-major) strides.
        int64_t ldc = N;
        
        // Tiled/blocking algorithm for cache optimization.
        // i-k-j loop order: a_ik is hoisted as a scalar register; the
        // innermost j loop keeps C and B sequential when inner strides are 1.
        for (int64_t i = 0; i < M; i += block_size) {
            for (int64_t j = 0; j < N; j += block_size) {
                for (int64_t k = 0; k < K; k += block_size) {
                    int64_t i_end = std::min(i + block_size, M);
                    int64_t j_end = std::min(j + block_size, N);
                    int64_t k_end = std::min(k + block_size, K);
                    
                    for (int64_t ii = i; ii < i_end; ++ii) {
                        for (int64_t kk = k; kk < k_end; ++kk) {
                            T a_ik = A[ii * lda + kk * inner_a];
                            for (int64_t jj = j; jj < j_end; ++jj) {
                                C[ii * ldc + jj] += a_ik * B[kk * ldb + jj * inner_b];
                            }
                        }
                    }
                }
            }
        }
    }
    
    static std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& shape1, 
                                                   const std::vector<int64_t>& shape2) {
        // NumPy-style broadcasting
        // 1. Right-align shapes
        // 2. Dimensions of size 1 broadcast to match the other
        // 3. Missing dimensions are treated as size 1
        // 4. Output shape is element-wise maximum
        
        size_t ndim1 = shape1.size();
        size_t ndim2 = shape2.size();
        size_t max_ndim = std::max(ndim1, ndim2);
        
        std::vector<int64_t> result(max_ndim);
        
        for (size_t i = 0; i < max_ndim; ++i) {
            int64_t dim1 = (i < ndim1) ? shape1[ndim1 - 1 - i] : 1;
            int64_t dim2 = (i < ndim2) ? shape2[ndim2 - 1 - i] : 1;
            
            // Check compatibility
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                throw runtime_error("Batch dimensions are not broadcastable");
            }
            
            result[max_ndim - 1 - i] = std::max(dim1, dim2);
        }
        
        return result;
    }
    
    static size_t compute_batch_offset(const Tensor& tensor, const std::vector<int64_t>& batch_indices) {
        // Compute flat offset for batch slice
        // Handles broadcasting: if tensor has fewer batch dims than batch_indices,
        // the extra dims in batch_indices are ignored (treated as size 1)
        
        size_t offset = 0;
        const auto& shape = tensor.shape();
        const auto& strides = tensor.strides();
        
        // Tensor has (batch_dims, M, K) or (batch_dims, K, N)
        // batch_indices corresponds to the broadcasted batch shape
        // We need to map batch_indices to tensor's actual batch dimensions
        
        size_t tensor_ndim = tensor.shape().size();
        size_t tensor_batch_ndim = tensor_ndim - 2;  // Last 2 dims are matrix dims
        size_t batch_indices_ndim = batch_indices.size();
        
        // Right-align: compare from the right
        for (size_t i = 0; i < tensor_batch_ndim; ++i) {
            // Map batch_indices to tensor's batch dimensions
            // batch_indices is right-aligned with broadcasted shape
            // tensor's batch dims are right-aligned within batch_indices
            
            size_t batch_idx_pos = batch_indices_ndim - tensor_batch_ndim + i;
            int64_t idx = 0;
            
            if (batch_idx_pos < batch_indices_ndim) {
                // If tensor dim is 1, always use index 0 (broadcasting)
                if (shape[i] == 1) {
                    idx = 0;
                } else {
                    idx = batch_indices[batch_idx_pos];
                }
            }
            
            offset += idx * strides[i];
        }
        
        return offset;
    }
    
    Tensor matmul(const Tensor& other) const;

    //user faced backward function used for loss.backward()
    void backward();


private:
    std::shared_ptr<TensorImpl> impl_;

    Tensor(shared_ptr<Storage> store, const std::vector<int64_t>& shape, DType dtype);
    Tensor(shared_ptr<TensorImpl> tImple);

    //server face function that actually compute the gradient
    void backward_impl(const Tensor& passedDownGrad);

    template<typename T>
    void sum_impl(Tensor& result, int64_t dim) const;
};


#endif