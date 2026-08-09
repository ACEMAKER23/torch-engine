#include "grad_fn.h"
#include "../tensor/tensor.h"
#include <vector>
#include <cmath>

#ifdef USE_CUDA
#include "../cuda/elementwise_cuda.h"
#include "../core/cuda_utils.h"
#endif

// Helper: reduce gradient by summing over broadcast dimensions
// If input_shape differs from grad_shape, sum over dimensions where input had size 1
static Tensor reduce_gradient(const Tensor& grad, const std::vector<int64_t>& input_shape) {
    if (grad.shape() == input_shape) {
        return grad;  // No broadcasting, no reduction needed
    }

    Tensor result = grad;
    int64_t grad_rank = (int64_t)grad.shape().size();
    int64_t input_rank = (int64_t)input_shape.size();

    // Process dimensions from left (dim 0) to right.
    // current_dim tracks the current position in the evolving result tensor.
    // Summing at current_dim removes that dimension (rank shrinks by 1), so
    // the next original dim maps to the same current_dim.
    // Keeping a dimension advances current_dim by 1.
    int64_t current_dim = 0;

    for (int64_t i = 0; i < grad_rank; ++i) {
        // Right-aligned: what is the corresponding input dimension?
        int64_t input_dim_idx = i - (grad_rank - input_rank);

        if (input_dim_idx < 0) {
            // This grad dim does not exist in input (extra leading dim) → sum away
            result = result.sum(current_dim);
            // Don't advance current_dim; next original dim maps to the same slot
        } else if (input_shape[input_dim_idx] == 1 && result.shape()[current_dim] > 1) {
            // Input had size 1 at this dim (was broadcast to a larger size) → sum away
            if (result.shape().size() == 1) {
                // Total reduction that would otherwise become a 0-dim (unsupported)
                // scalar tensor; keep it as a 1-element 1-D tensor instead.
                Tensor result_1({1}, result.dtype(), result.device());
                float total = 0.0f;
                for (size_t j = 0; j < result.numel(); ++j)
                    total += result.at<float>(j);
                result_1.at<float>(0) = total;
                result = result_1;
            } else {
                result = result.sum(current_dim);
                // Re-insert the size-1 dimension so the shape matches input_shape
                auto sh = result.shape();
                sh.insert(sh.begin() + current_dim, 1LL);
                result = result.view(sh);
            }
            current_dim++;
        } else {
            // Dimensions match → advance to the next slot
            current_dim++;
        }
    }

    return result;
}

// Template helper for unary element-wise backward operations
// Eliminates unnecessary cloning by computing gradient directly
template<typename T>
static Tensor unary_backward_elementwise(const Tensor& input, const Tensor& upstream_grad, 
                                          std::function<T(T, T)> grad_fn) {
    Tensor result(input.shape(), input.dtype(), input.device());
    
    for (size_t i = 0; i < input.numel(); ++i) {
        T input_val = input.at<T>(i);
        T upstream_val = upstream_grad.at<T>(i);
        result.at<T>(i) = grad_fn(input_val, upstream_val);
    }
    
    return result;
}

// Template dispatch for unary backward operations
template<typename T>
static Tensor unary_backward_dispatch(const Tensor& input, const Tensor& upstream_grad, 
                                      std::function<T(T, T)> grad_fn) {
    return unary_backward_elementwise<T>(input, upstream_grad, grad_fn);
}

static Tensor unary_backward(const Tensor& input, const Tensor& upstream_grad,
                            std::function<float(float, float)> float_grad_fn) {
    switch (input.dtype()) {
        case DType::Float32:
            return unary_backward_dispatch<float>(input, upstream_grad, float_grad_fn);
        default:
            throw std::runtime_error("Unsupported dtype for unary backward");
    }
}

std::vector<Tensor> AddBackward::backward(const Tensor& pathDownGrad) {
    auto grad_a = reduce_gradient(pathDownGrad, inputs[0].shape());
    auto grad_b = reduce_gradient(pathDownGrad, inputs[1].shape());
    grad_a.setRequiresGrad(false);
    grad_b.setRequiresGrad(false);
    std::vector<Tensor> result;
    result.push_back(grad_a);
    result.push_back(grad_b);
    return result;
}

std::vector<Tensor> SubBackward::backward(const Tensor& pathDownGrad) {
    auto grad_a = reduce_gradient(pathDownGrad, inputs[0].shape());
    auto grad_b = reduce_gradient(-pathDownGrad, inputs[1].shape());
    grad_a.setRequiresGrad(false);
    grad_b.setRequiresGrad(false);
    std::vector<Tensor> result;
    result.push_back(grad_a);
    result.push_back(grad_b);
    return result;
}

std::vector<Tensor> MulBackward::backward(const Tensor& pathDownGrad) {
    auto grad_a = reduce_gradient(pathDownGrad * inputs[1], inputs[0].shape());
    auto grad_b = reduce_gradient(pathDownGrad * inputs[0], inputs[1].shape());
    grad_a.setRequiresGrad(false);
    grad_b.setRequiresGrad(false);
    std::vector<Tensor> result;
    result.push_back(grad_a);
    result.push_back(grad_b);
    return result;
}

std::vector<Tensor> MatMulBackward::backward(const Tensor& pathDownGrad) {
    auto grad_a = reduce_gradient(pathDownGrad.matmul(inputs[1].transpose_view(inputs[1].shape().size()-2,inputs[1].shape().size()-1)), inputs[0].shape());
    auto grad_b = reduce_gradient(inputs[0].transpose_view(inputs[0].shape().size()-2,inputs[0].shape().size()-1).matmul(pathDownGrad), inputs[1].shape());
    grad_a.setRequiresGrad(false);
    grad_b.setRequiresGrad(false);
    std::vector<Tensor> result;
    result.push_back(grad_a);
    result.push_back(grad_b);
    return result;
}


std::vector<Tensor> DivBackward::backward(const Tensor& pathDownGrad) {
    // For c = a/b: dc/da = 1/b, dc/db = -a/b^2
    auto grad_a = reduce_gradient(pathDownGrad / inputs[1], inputs[0].shape());
    auto grad_b = reduce_gradient(-pathDownGrad * inputs[0] / (inputs[1] * inputs[1]), inputs[1].shape());
    grad_a.setRequiresGrad(false);
    grad_b.setRequiresGrad(false);
    std::vector<Tensor> result;
    result.push_back(grad_a);
    result.push_back(grad_b);
    return result;
}

std::vector<Tensor> ReluBackward::backward(const Tensor& pathDownGrad) {
    // ReLU gradient: 1 if input > 0, else 0
    // Use template helper to avoid cloning and use compile-time dispatch
    auto grad = unary_backward(inputs[0], pathDownGrad, 
        [](float input_val, float upstream_val) -> float {
            return input_val > 0.0f ? upstream_val : 0.0f;
        });
    grad.setRequiresGrad(false);
    return {grad};
}

std::vector<Tensor> GeluBackward::backward(const Tensor& pathDownGrad) {
    const Tensor& input = inputs[0];
    Tensor grad(input.shape(), input.dtype(), input.device());

    if (input.device() == Device::CUDA && input.dtype() == DType::Float32) {
#ifdef USE_CUDA
        Tensor input_c = input.contiguous();
        Tensor path_c  = pathDownGrad.contiguous();
        cuda_gelu_backward(static_cast<const float*>(input_c.data()),
                           static_cast<const float*>(path_c.data()),
                           static_cast<float*>(grad.data()),
                           static_cast<int64_t>(input.numel()));
        cuda_check_error(cudaGetLastError(), "cuda_gelu_backward failed");
        cuda_check_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize after gelu backward");
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;

        grad = unary_backward(input, pathDownGrad,
            [sqrt_2_over_pi, coeff](float input_val, float upstream_val) -> float {
                float x_cubed = input_val * input_val * input_val;
                float z = sqrt_2_over_pi * (input_val + coeff * x_cubed);
                float tanh_z = std::tanh(z);
                float sech_sq = 1.0f - tanh_z * tanh_z;
                float inner = sqrt_2_over_pi * (1.0f + 3.0f * coeff * input_val * input_val);
                float gelu_grad = 0.5f * (1.0f + tanh_z) * (1.0f + input_val * sech_sq * inner);
                return upstream_val * gelu_grad;
            });
    }

    grad.setRequiresGrad(false);
    return {grad};
}

std::vector<Tensor> SigmoidBackward::backward(const Tensor& pathDownGrad) {
    // Sigmoid gradient: sigmoid(x) * (1 - sigmoid(x))
    // Use template helper to avoid cloning and use compile-time dispatch
    auto grad = unary_backward(inputs[0], pathDownGrad,
        [](float input_val, float upstream_val) -> float {
            float sig = 1.0f / (1.0f + std::exp(-input_val));
            return upstream_val * sig * (1.0f - sig);
        });
    grad.setRequiresGrad(false);
    return {grad};
}

std::vector<Tensor> CrossEntropyBackward::backward(const Tensor& pathDownGrad) {
    // PyTorch-style gradient: softmax(logits) - one_hot(target)
    // inputs[0] = logits, inputs[1] = target class indices

    const Tensor& logits = inputs[0];
    const Tensor& targets = inputs[1];
    int64_t target_class = targets.at<int64_t>(0);

    // Compute softmax
    float max_logit = logits.at<float>(0);
    for (size_t i = 1; i < logits.numel(); ++i) {
        max_logit = std::max(max_logit, logits.at<float>(i));
    }

    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits.numel(); ++i) {
        sum_exp += std::exp(logits.at<float>(i) - max_logit);
    }

    Tensor grad(logits.shape(), logits.dtype(), logits.device());

    for (size_t i = 0; i < logits.numel(); ++i) {
        float softmax_val = std::exp(logits.at<float>(i) - max_logit) / sum_exp;
        float one_hot = (i == target_class) ? 1.0f : 0.0f;
        grad.at<float>(i) = softmax_val - one_hot;
    }

    grad.setRequiresGrad(false);
    return {grad};
}

std::vector<Tensor> CrossEntropyWithProbsBackward::backward(const Tensor& pathDownGrad) {
    // Gradient for probability-based loss: -1/p for target class, 0 otherwise
    // inputs[0] = probabilities, inputs[1] = target class indices

    const Tensor& probabilities = inputs[0];
    const Tensor& targets = inputs[1];
    int64_t target_class = targets.at<int64_t>(0);

    Tensor grad(probabilities.shape(), probabilities.dtype(), probabilities.device());

    for (size_t i = 0; i < probabilities.numel(); ++i) {
        if (i == target_class) {
            grad.at<float>(i) = -1.0f / probabilities.at<float>(i);
        } else {
            grad.at<float>(i) = 0.0f;
        }
    }

    grad.setRequiresGrad(false);
    return {grad};
}

std::vector<Tensor> CrossEntropyBatchedBackward::backward(const Tensor& pathDownGrad) {
    const Tensor& logits = inputs[0];
    const Tensor& targets = inputs[1];

    auto logShape = logits.shape();
    int64_t B = logShape[0];
    int64_t T = logShape[1];
    int64_t V = logShape[2];
    int64_t N = B * T;

    float upstream = 1.0f;
    if (pathDownGrad.numel() == 1) {
        if (pathDownGrad.device() == Device::CUDA) {
#ifdef USE_CUDA
            cuda_check_error(
                cudaMemcpy(&upstream, pathDownGrad.data(), sizeof(float), cudaMemcpyDeviceToHost),
                "cudaMemcpy upstream D2H");
#else
            throw std::runtime_error("CUDA not available");
#endif
        } else {
            upstream = pathDownGrad.at<float>(0);
        }
    }
    float scale = upstream / static_cast<float>(N);

    Tensor grad(logits.shape(), logits.dtype(), logits.device());

    if (logits.device() == Device::CUDA) {
#ifdef USE_CUDA
        Tensor logits_c  = logits.contiguous();
        Tensor targets_c = targets.contiguous();
        cuda_crossentropy_batched_backward(
            static_cast<const float*>(logits_c.data()),
            static_cast<const int64_t*>(targets_c.data()),
            static_cast<float*>(grad.data()),
            scale,
            static_cast<int>(B), static_cast<int>(T), static_cast<int>(V));
        cuda_check_error(cudaGetLastError(), "cuda_crossentropy_batched_backward failed");
        cuda_check_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize after crossentropy backward");
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        const float* logits_ptr = static_cast<const float*>(logits.data());
        const int64_t* targets_ptr = static_cast<const int64_t*>(targets.data());
        const std::vector<int64_t>& ls = logits.strides();
        const std::vector<int64_t>& ts = targets.strides();
        float* grad_ptr = static_cast<float*>(grad.data());
        const std::vector<int64_t>& gs = grad.strides();

        for (int64_t b = 0; b < B; ++b) {
            for (int64_t t = 0; t < T; ++t) {
                const float* row = logits_ptr + b * ls[0] + t * ls[1];
                float* grad_row = grad_ptr + b * gs[0] + t * gs[1];
                int64_t target_class = targets_ptr[b * ts[0] + t * ts[1]];

                float max_logit = row[0];
                for (int64_t v = 1; v < V; ++v) {
                    max_logit = std::max(max_logit, row[v * ls[2]]);
                }

                float sum_exp = 0.0f;
                for (int64_t v = 0; v < V; ++v) {
                    sum_exp += std::exp(row[v * ls[2]] - max_logit);
                }

                for (int64_t v = 0; v < V; ++v) {
                    float softmax_val = std::exp(row[v * ls[2]] - max_logit) / sum_exp;
                    float one_hot = (v == target_class) ? 1.0f : 0.0f;
                    grad_row[v * gs[2]] = (softmax_val - one_hot) * scale;
                }
            }
        }
    }

    grad.setRequiresGrad(false);
    return {grad};
}

std::vector<Tensor> MSEBackward::backward(const Tensor& pathDownGrad) {
    // MSE gradient: 2 * (predictions - targets) / n
    // inputs[0] = predictions, inputs[1] = targets

    const Tensor& predictions = inputs[0];
    const Tensor& targets = inputs[1];

    Tensor diff = predictions - targets;
    float n = static_cast<float>(predictions.numel());
    float scale = 2.0f / n;

    Tensor grad(diff.shape(), diff.dtype(), diff.device());

    for (size_t i = 0; i < diff.numel(); ++i) {
        grad.at<float>(i) = diff.at<float>(i) * scale;
    }

    grad.setRequiresGrad(false);
    return {grad};
}

std::vector<Tensor> BCEBackward::backward(const Tensor& pathDownGrad) {
    // BCE gradient: (predictions - targets) / (predictions * (1 - predictions))
    // inputs[0] = predictions (probabilities), inputs[1] = targets (0 or 1)

    const Tensor& predictions = inputs[0];
    const Tensor& targets = inputs[1];

    Tensor grad(predictions.shape(), predictions.dtype(), predictions.device());

    for (size_t i = 0; i < predictions.numel(); ++i) {
        float p = predictions.at<float>(i);
        float t = targets.at<float>(i);
        // Avoid division by zero
        float denom = p * (1.0f - p);
        if (std::abs(denom) < 1e-7f) {
            grad.at<float>(i) = 0.0f;
        } else {
            grad.at<float>(i) = (p - t) / denom;
        }
    }

    grad.setRequiresGrad(false);
    return {grad};
}

std::vector<Tensor> TransposeBackward::backward(const Tensor& pathDownGrad) {
    // inputs[0] = original tensor (before the transpose).
    // Swapping d1,d2 is its own inverse, so we just transpose back.
    Tensor grad = pathDownGrad.transpose_view(d1, d2).contiguous();
    grad.setRequiresGrad(false);
    return {grad};
}

std::vector<Tensor> LayerNormBackward::backward(const Tensor& pathDownGrad) {
    // inputs[0] = input x   [... , D]
    // inputs[1] = weight γ  [D]
    // inputs[2] = bias β    [D]   (not needed for math but keeps the inputs[] index consistent)
    const Tensor& x     = inputs[0];
    const Tensor& gamma = inputs[1];

    const auto& x_shape = x.shape();
    const int64_t D     = x_shape.back();
    const int64_t batch = static_cast<int64_t>(x.numel()) / D;
    const float   N     = static_cast<float>(D);

#ifdef USE_CUDA
    if (x.device() == Device::CUDA) {
        if (!x.is_contiguous() || !pathDownGrad.is_contiguous()) {
            throw std::runtime_error("CUDA LayerNormBackward requires contiguous x and pathDownGrad");
        }

        Tensor gamma_dev = gamma.toDevice(Device::CUDA);

        Tensor d_x(x_shape, x.dtype(), Device::CUDA);
        Tensor d_gamma_gpu({D}, gamma.dtype(), Device::CUDA);
        Tensor d_beta_gpu ({D}, gamma.dtype(), Device::CUDA);

        cuda_fill(static_cast<float*>(d_gamma_gpu.data()), 0.0f, D);
        cuda_fill(static_cast<float*>(d_beta_gpu.data()),  0.0f, D);

        cuda_layernorm_backward(static_cast<const float*>(x.data()),
                                static_cast<const float*>(gamma_dev.data()),
                                static_cast<const float*>(pathDownGrad.data()),
                                static_cast<float*>(d_x.data()),
                                static_cast<float*>(d_gamma_gpu.data()),
                                static_cast<float*>(d_beta_gpu.data()),
                                static_cast<int>(batch),
                                static_cast<int>(D),
                                eps);
        cuda_check_error(cudaGetLastError(), "cuda_layernorm_backward failed");
        cuda_check_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize after layernorm backward");

        Tensor d_gamma({D}, gamma.dtype(), Device::CPU);
        Tensor d_beta ({D}, gamma.dtype(), Device::CPU);
        cuda_check_error(
            cudaMemcpy(d_gamma.data(), d_gamma_gpu.data(), D * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy d_gamma D2H");
        cuda_check_error(
            cudaMemcpy(d_beta.data(), d_beta_gpu.data(), D * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy d_beta D2H");



        d_x    .setRequiresGrad(false);
        d_gamma.setRequiresGrad(false);
        d_beta .setRequiresGrad(false);
        return {d_x, d_gamma, d_beta};
    }
#endif

    Tensor d_x    (x_shape,      x.dtype(),     x.device());
    Tensor d_gamma({D},          gamma.dtype(), gamma.device());
    Tensor d_beta ({D},          gamma.dtype(), gamma.device());

    d_gamma.fill_<float>(0.0f);
    d_beta .fill_<float>(0.0f);

    const float* x_data  = static_cast<const float*>(x.data());
    const float* g_data  = static_cast<const float*>(gamma.data());
    const float* dy_data = static_cast<const float*>(pathDownGrad.data());
    float*       dx_data = static_cast<float*>(d_x.data());
    float*       dg_data = static_cast<float*>(d_gamma.data());
    float*       db_data = static_cast<float*>(d_beta.data());

    // Allocate x_hat once outside the loop to avoid per-row heap allocations.
    std::vector<float> x_hat(D);

    for (int64_t b = 0; b < batch; ++b) {
        const float* xr  = x_data  + b * D;
        const float* dyr = dy_data + b * D;
        float*       dxr = dx_data + b * D;

        // mean and variance of this row
        float mean = 0.0f;
        for (int64_t j = 0; j < D; ++j) mean += xr[j];
        mean /= N;

        float var = 0.0f;
        for (int64_t j = 0; j < D; ++j) {
            float d = xr[j] - mean;
            var += d * d;
        }
        var /= N;
        const float std_val = std::sqrt(var + eps);

        // normalised input for this row (reuses the pre-allocated buffer)
        for (int64_t j = 0; j < D; ++j)
            x_hat[j] = (xr[j] - mean) / std_val;

        // accumulate d_gamma = dy * x_hat,  d_beta = dy
        for (int64_t j = 0; j < D; ++j) {
            dg_data[j] += dyr[j] * x_hat[j];
            db_data[j] += dyr[j];
        }

        // mean(γ·dy) and mean(γ·dy·x̂) across the normalised dimension
        float mean_g_dy      = 0.0f;
        float mean_g_dy_xhat = 0.0f;
        for (int64_t j = 0; j < D; ++j) {
            float g_dy = g_data[j] * dyr[j];
            mean_g_dy      += g_dy;
            mean_g_dy_xhat += g_dy * x_hat[j];
        }
        mean_g_dy      /= N;
        mean_g_dy_xhat /= N;

        // d_x_i = (1/σ) · (γ_i·dy_i  −  mean(γ·dy)  −  x̂_i·mean(γ·dy·x̂))
        for (int64_t j = 0; j < D; ++j) {
            dxr[j] = (g_data[j] * dyr[j] - mean_g_dy - x_hat[j] * mean_g_dy_xhat)
                     / std_val;
        }
    }

    d_x   .setRequiresGrad(false);
    d_gamma.setRequiresGrad(false);
    d_beta .setRequiresGrad(false);
    return {d_x, d_gamma, d_beta};
}

std::vector<Tensor> ViewBackward::backward(const Tensor& pathDownGrad) {
    // Reshape the upstream gradient back to the original input shape.
    // View is metadata-only, so the data order is unchanged.
    Tensor grad = pathDownGrad.view(original_shape);
    grad.setRequiresGrad(false);
    return {grad};
}

std::vector<Tensor> CopyBackward::backward(const Tensor& pathDownGrad) {
    // Contiguous()/clone() is a copy in the forward; backward passes the
    // upstream gradient through unchanged (shape already matches input).
    Tensor grad = pathDownGrad;
    grad.setRequiresGrad(false);
    return {grad};
}

std::vector<Tensor> SoftmaxBackward::backward(const Tensor& pathDownGrad) {
    // inputs[0] = original pre-softmax logits x
    // dx_i = s_i * (dy_i - sum_j (dy_j * s_j))  where s = softmax(x)
    const Tensor& x = inputs[0];
    const auto shape = x.shape();
    const int64_t ndim = static_cast<int64_t>(shape.size());
    int64_t d = dim;
    if (d < 0) d += ndim;
    if (d < 0 || d >= ndim) throw std::runtime_error("Invalid softmax dim in backward");

    const int64_t dim_size = shape[d];
    const int64_t outer_size = static_cast<int64_t>(x.numel()) / dim_size;

    Tensor dx(shape, x.dtype(), x.device());
    dx.setRequiresGrad(false);

#ifdef USE_CUDA
    if (x.device() == Device::CUDA) {
        if (!x.is_contiguous() || d != ndim - 1 || !pathDownGrad.is_contiguous()) {
            throw std::runtime_error("CUDA SoftmaxBackward currently requires contiguous tensors and dim == last dimension");
        }
        // Recompute s = softmax(x) on device, then dx = s * (dy - dot(dy, s)).
        Tensor s(shape, x.dtype(), Device::CUDA);
        s.setRequiresGrad(false);
        cuda_check_error(
            cudaMemcpy(s.data(), x.data(), x.numel() * sizeof(float), cudaMemcpyDeviceToDevice),
            "cudaMemcpy in SoftmaxBackward");
        cuda_softmax_forward(static_cast<const float*>(s.data()),
                             static_cast<float*>(s.data()),
                             outer_size, dim_size);
        cuda_softmax_backward(static_cast<const float*>(s.data()),
                              static_cast<const float*>(pathDownGrad.data()),
                              static_cast<float*>(dx.data()),
                              outer_size, dim_size);
        cuda_check_error(cudaGetLastError(), "cuda_softmax backward kernels failed");
        cuda_check_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize after SoftmaxBackward");
        return {dx};
    }
#endif

    std::vector<int64_t> idx(ndim, 0);

    for (int64_t outer = 0; outer < outer_size; ++outer) {
        // decode outer index to multidimensional index (except dim)
        int64_t tmp = outer;
        for (int i = ndim - 1; i >= 0; --i) {
            if (i == d) continue;
            idx[i] = tmp % shape[i];
            tmp /= shape[i];
        }

        // Step 1: max for numerical stability
        float max_val = -std::numeric_limits<float>::infinity();
        for (int64_t i = 0; i < dim_size; ++i) {
            idx[d] = i;
            int64_t offset = 0;
            for (int dim = 0; dim < ndim; ++dim) offset += idx[dim] * x.strides()[dim];
            max_val = std::max(max_val, x.at<float>(offset));
        }

        // Step 2: compute softmax s_i and store
        float sum = 0.0f;
        std::vector<float> s(dim_size);
        for (int64_t i = 0; i < dim_size; ++i) {
            idx[d] = i;
            int64_t offset = 0;
            for (int dim = 0; dim < ndim; ++dim) offset += idx[dim] * x.strides()[dim];
            float v = std::exp(x.at<float>(offset) - max_val);
            s[i] = v;
            sum += v;
        }
        for (float& v : s) v /= sum;

        // Step 3: dot = sum_j (dy_j * s_j)
        float dot = 0.0f;
        for (int64_t i = 0; i < dim_size; ++i) {
            idx[d] = i;
            int64_t offset = 0;
            for (int dim = 0; dim < ndim; ++dim) offset += idx[dim] * pathDownGrad.strides()[dim];
            dot += pathDownGrad.at<float>(offset) * s[i];
        }

        // Step 4: dx_i = s_i * (dy_i - dot)
        for (int64_t i = 0; i < dim_size; ++i) {
            idx[d] = i;
            int64_t grad_offset = 0;
            int64_t out_offset = 0;
            for (int dim = 0; dim < ndim; ++dim) {
                grad_offset += idx[dim] * pathDownGrad.strides()[dim];
                out_offset += idx[dim] * dx.strides()[dim];
            }
            dx.at<float>(out_offset) = s[i] * (pathDownGrad.at<float>(grad_offset) - dot);
        }
    }

    return {dx};
}

std::vector<Tensor> L1Backward::backward(const Tensor& pathDownGrad) {
    // L1 gradient: sign(predictions - targets) / n
    // inputs[0] = predictions, inputs[1] = targets

    const Tensor& predictions = inputs[0];
    const Tensor& targets = inputs[1];

    Tensor diff = predictions - targets;
    float n = static_cast<float>(predictions.numel());

    Tensor grad(diff.shape(), diff.dtype(), diff.device());

    for (size_t i = 0; i < diff.numel(); ++i) {
        float d = diff.at<float>(i);
        grad.at<float>(i) = (d > 0.0f) ? 1.0f / n : ((d < 0.0f) ? -1.0f / n : 0.0f);
    }

    grad.setRequiresGrad(false);
    return {grad};
}

std::vector<Tensor> EmbeddingBackward::backward(const Tensor& pathDownGrad) {
    const Tensor& indices = inputs[0];
    const Tensor& weight  = inputs[1];

    const Tensor grad = pathDownGrad.contiguous();
    std::vector<int64_t> weight_shape = weight.shape();
    const int64_t D = weight_shape.back();
    const int64_t total_indices = indices.numel();
    const int64_t flat_grad_size = grad.numel();
    (void)flat_grad_size;  // Unused for now; validates shape consistency implicitly.

    // Placeholder gradient for indices (indices are discrete, no meaningful gradient).
    Tensor d_input(indices.shape(), indices.dtype(), indices.device());

    Tensor d_weight(weight_shape, weight.dtype(), weight.device());
    d_weight.fill_<float>(0.0f);

    if (d_input.device() == Device::CPU) {
        d_input.fill_<int64_t>(0);
    } else {
#ifdef USE_CUDA
        size_t d_input_bytes = d_input.numel() * sizeof(int64_t);
        cuda_check_error(cudaMemset(d_input.data(), 0, d_input_bytes),
                         "cudaMemset d_input");
#endif
    }

    if (grad.device() == Device::CUDA) {
#ifdef USE_CUDA
        Tensor d_weight_gpu(weight_shape, weight.dtype(), Device::CUDA);
        cuda_fill(static_cast<float*>(d_weight_gpu.data()), 0.0f,
                  static_cast<int64_t>(d_weight_gpu.numel()));

        cuda_embedding_backward(static_cast<const int64_t*>(indices.data()),
                                static_cast<const float*>(grad.data()),
                                static_cast<float*>(d_weight_gpu.data()),
                                static_cast<int>(total_indices),
                                static_cast<int>(D));
        cuda_check_error(cudaGetLastError(), "cuda_embedding_backward failed");
        cuda_check_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize after embedding backward");

        cuda_check_error(
            cudaMemcpy(d_weight.data(), d_weight_gpu.data(),
                       static_cast<size_t>(num_embeddings * D * sizeof(float)),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy d_weight D2H");
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        const int64_t* idx_data = static_cast<const int64_t*>(indices.data());
        const float* grad_data = static_cast<const float*>(grad.data());
        float* dw_data = static_cast<float*>(d_weight.data());

        for (int64_t i = 0; i < total_indices; ++i) {
            int64_t idx = idx_data[i];
            if (idx < 0 || idx >= num_embeddings) continue;
            for (int64_t j = 0; j < D; ++j) {
                dw_data[idx * D + j] += grad_data[i * D + j];
            }
        }
    }

    d_input.setRequiresGrad(false);
    d_weight.setRequiresGrad(false);
    return {d_input, d_weight};
}

