#include "gpt.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

GPT::GPT(int64_t vocabSize, int64_t maxSeqLen, int64_t embedDim, 
         int64_t numHeads, int64_t numLayers, int64_t ffDim, 
         DType dtype, float dropoutRate)
    : vocabSize_(vocabSize), maxSeqLen_(maxSeqLen), embedDim_(embedDim),
      numHeads_(numHeads), numLayers_(numLayers), ffDim_(ffDim),
      dtype_(dtype), dropoutRate_(dropoutRate),
      tokenEmbedding_(std::make_unique<Embedding>(vocabSize, embedDim, dtype)),
      posEmbedding_(std::make_unique<PositionalEmbedding>(maxSeqLen, embedDim, dtype)),
      embedDropout_(std::make_unique<Dropout>(dropoutRate)),
      finalNorm_(std::make_unique<LayerNorm>(embedDim, dtype)),
      lmHead_(std::make_unique<Linear>(embedDim, vocabSize, dtype))
{
    // Create transformer blocks
    for (int64_t i = 0; i < numLayers; ++i) {
        blocks_.push_back(std::make_unique<TransformerBlock>(
            embedDim, numHeads, ffDim, dtype, dropoutRate));
    }
}

Tensor GPT::forward(const Tensor& input) {
    // input: [batch_size, seq_len] - token IDs
    
    // Get token embeddings: [batch_size, seq_len, embed_dim]
    Tensor tokenEmb = tokenEmbedding_->forward(input);
    
    // Get positional embeddings: [seq_len, embed_dim]
    Tensor posEmb = posEmbedding_->forward(input);
    
    // Broadcast positional embeddings to match batch size
    // posEmb: [seq_len, embed_dim] -> [batch_size, seq_len, embed_dim]
    int64_t batch_size = input.shape()[0];
    int64_t seq_len = input.shape()[1];
    int64_t embed_dim = embedDim_;
    
    Tensor posEmbBroadcast({batch_size, seq_len, embed_dim}, tokenEmb.dtype(), tokenEmb.device());

    // Copy positional embeddings for each batch
    size_t row_bytes = static_cast<size_t>(seq_len * embed_dim) * sizeof(float);
    if (posEmb.device() == Device::CUDA) {
#ifdef USE_CUDA
        auto* dst = static_cast<char*>(posEmbBroadcast.data());
        const auto* src = static_cast<const char*>(posEmb.data());
        for (int64_t b = 0; b < batch_size; ++b) {
            cudaMemcpy(dst + b * row_bytes, src, row_bytes, cudaMemcpyDeviceToDevice);
        }
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        auto* src_data = static_cast<float*>(posEmb.data());
        auto* dst_data = static_cast<float*>(posEmbBroadcast.data());
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t s = 0; s < seq_len; ++s) {
                for (int64_t e = 0; e < embed_dim; ++e) {
                    dst_data[b * seq_len * embed_dim + s * embed_dim + e] =
                        src_data[s * embed_dim + e];
                }
            }
        }
    }
    
    // Add token and positional embeddings
    Tensor x = tokenEmb + posEmbBroadcast;
    
    // Apply dropout
    x = embedDropout_->forward(x);
    
    // Pass through transformer blocks
    for (auto& block : blocks_) {
        x = block->forward(x);
    }
    
    // Apply final layer norm
    x = finalNorm_->forward(x);
    
    // Project to vocabulary logits: [batch_size, seq_len, vocab_size]
    Tensor logits = lmHead_->forward(x);
    
    return logits;
}

std::vector<Tensor> GPT::parameters() {
    std::vector<Tensor> params;
    
    // Token embedding parameters
    auto tokenEmbParams = tokenEmbedding_->parameters();
    params.insert(params.end(), tokenEmbParams.begin(), tokenEmbParams.end());
    
    // Positional embedding parameters
    auto posEmbParams = posEmbedding_->parameters();
    params.insert(params.end(), posEmbParams.begin(), posEmbParams.end());
    
    // Transformer block parameters
    for (auto& block : blocks_) {
        auto blockParams = block->parameters();
        params.insert(params.end(), blockParams.begin(), blockParams.end());
    }
    
    // Final layer norm parameters
    auto finalNormParams = finalNorm_->parameters();
    params.insert(params.end(), finalNormParams.begin(), finalNormParams.end());
    
    // Language model head parameters
    auto lmHeadParams = lmHead_->parameters();
    params.insert(params.end(), lmHeadParams.begin(), lmHeadParams.end());
    
    return params;
}

void GPT::zero_grad() {
    tokenEmbedding_->zero_grad();
    posEmbedding_->zero_grad();
    
    for (auto& block : blocks_) {
        block->zero_grad();
    }
    
    finalNorm_->zero_grad();
    lmHead_->zero_grad();
}

void GPT::to_cuda() {
    tokenEmbedding_->to_cuda();
    posEmbedding_->to_cuda();
    for (auto& block : blocks_) {
        block->to_cuda();
    }
    finalNorm_->to_cuda();
    lmHead_->to_cuda();
}
