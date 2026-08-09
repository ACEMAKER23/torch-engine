#include "embedding.h"
#include "../core/grad_fn.h"
#include <random>
#include <cmath>

#ifdef USE_CUDA
#include "../cuda/elementwise_cuda.h"
#include "../core/cuda_utils.h"
#endif

Embedding::Embedding(int64_t num_embeddings, int64_t embedding_dim, DType dtype)
    : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim),
      weight_({num_embeddings, embedding_dim}, dtype, Device::CPU) {
    init_parameters();
    weight_.setRequiresGrad(true);
}

Tensor Embedding::forward(const Tensor& input) {
    // input: [batch_size, seq_len] or [seq_len] - indices
    // weight: [num_embeddings, embedding_dim]
    // output: [batch_size, seq_len, embedding_dim] or [seq_len, embedding_dim]
    
    std::vector<int64_t> input_shape = input.shape();
    std::vector<int64_t> output_shape = input_shape;
    output_shape.push_back(embedding_dim_);
    
    if (input.dtype() != DType::Int64) {
        throw std::runtime_error("Embedding input must be Int64");
    }
    
    Tensor result(output_shape, weight_.dtype(), input.device());
    size_t total_indices = input.numel();
    
#ifdef USE_CUDA
    if (input.device() == Device::CUDA) {
        if (weight_.dtype() != DType::Float32) {
            throw std::runtime_error("CUDA Embedding only supports Float32 weight");
        }
        Tensor weight_dev = weight_.toDevice(Device::CUDA);
        cuda_check_error(cudaGetLastError(), "before embedding forward");
        cuda_embedding_forward(static_cast<const int64_t*>(input.data()),
                               static_cast<const float*>(weight_dev.data()),
                               static_cast<float*>(result.data()),
                               static_cast<int>(total_indices),
                               static_cast<int>(embedding_dim_));
        cuda_check_error(cudaGetLastError(), "cuda_embedding_forward failed");
        cuda_check_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize after embedding forward");
    } else
#endif
    {
        for (size_t i = 0; i < total_indices; ++i) {
            int64_t idx = input.at<int64_t>(i);
            if (idx < 0 || idx >= num_embeddings_) {
                throw std::runtime_error("Embedding index out of range");
            }
            
            for (int64_t j = 0; j < embedding_dim_; ++j) {
                if (weight_.dtype() == DType::Float32) {
                    result.at<float>(i * embedding_dim_ + j) = weight_.at<float>({idx, j});
                } else if (weight_.dtype() == DType::Int32) {
                    result.at<int32_t>(i * embedding_dim_ + j) = weight_.at<int32_t>({idx, j});
                } else if (weight_.dtype() == DType::Int64) {
                    result.at<int64_t>(i * embedding_dim_ + j) = weight_.at<int64_t>({idx, j});
                }
            }
        }
    }
    
    if (result.requiredGrad() || weight_.requiredGrad()) {
        auto fn = std::make_shared<EmbeddingBackward>();
        fn->inputs = {input, weight_};
        fn->num_embeddings = num_embeddings_;
        fn->embedding_dim   = embedding_dim_;
        result.setGradFn(fn);
        result.setRequiresGrad(true);
    }
    
    return result;
}

void Embedding::init_parameters() {
    if (num_embeddings_ == 0 || embedding_dim_ == 0) {
        throw std::runtime_error("num_embeddings and embedding_dim cannot be zero");
    }
    
    if (weight_.dtype() != DType::Float32) {
        throw std::runtime_error("Only Float32 dtype is supported for parameter initialization");
    }

    std::mt19937 gen(std::random_device{}());
    float stddev = std::sqrt(1.0f / embedding_dim_);
    std::normal_distribution<float> dist(0.0f, stddev);

    auto* data = static_cast<float*>(weight_.data());
    for (size_t i = 0; i < weight_.numel(); ++i) {
        data[i] = dist(gen);
    }
}

std::vector<Tensor> Embedding::parameters() {
    return {weight_};
}

void Embedding::zero_grad() {
    // Gradients are cleared by the tensor implementation during backward
}
