#include "positional_embedding.h"
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

PositionalEmbedding::PositionalEmbedding(int64_t max_seq_len, int64_t d_model, DType dtype)
    : max_seq_len_(max_seq_len), d_model_(d_model), pe_({max_seq_len, d_model}, dtype, Device::CPU) {
    
    if (dtype != DType::Float32) {
        throw std::runtime_error("PositionalEmbedding only supports Float32");
    }
    
    init_positional_embeddings();
}

void PositionalEmbedding::init_positional_embeddings() {
    auto* data = static_cast<float*>(pe_.data());
    
    for (int64_t pos = 0; pos < max_seq_len_; ++pos) {
        for (int64_t i = 0; i < d_model_; ++i) {
            float div_term = std::exp(std::log(10000.0f) * (2.0f * (static_cast<float>(i) / 2.0f)) / static_cast<float>(d_model_));
            
            int64_t idx = pos * d_model_ + i;
            
            if (i % 2 == 0) {
                // Even dimension: sin
                data[idx] = std::sin(static_cast<float>(pos) / div_term);
            } else {
                // Odd dimension: cos
                data[idx] = std::cos(static_cast<float>(pos) / div_term);
            }
        }
    }
}

Tensor PositionalEmbedding::forward(const Tensor& input) {
    // input shape: [batch, seq_len] or [seq_len]
    // pe shape: [max_seq_len, d_model]
    // Return positional embeddings for the sequence length

    int64_t seq_len = input.shape()[input.shape().size() - 1];

    // Slice positional embeddings to match sequence length
    // Create a new tensor with shape [seq_len, d_model] on the same device as pe_
    Tensor sliced_pe({seq_len, d_model_}, pe_.dtype(), pe_.device());

    size_t bytes = static_cast<size_t>(seq_len * d_model_) * sizeof(float);

    if (pe_.device() == Device::CUDA) {
#ifdef USE_CUDA
        cudaMemcpy(sliced_pe.data(), pe_.data(), bytes, cudaMemcpyDeviceToDevice);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        std::memcpy(sliced_pe.data(), pe_.data(), bytes);
    }

    return sliced_pe;  // Return [seq_len, d_model]
}

void PositionalEmbedding::to_cuda() {
    pe_ = pe_.toDevice(Device::CUDA);
}
