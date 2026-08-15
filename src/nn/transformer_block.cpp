#include "transformer_block.h"

TransformerBlock::TransformerBlock(int64_t embedDim, int64_t numHeads, int64_t ffDim, 
                                   DType dtype, float dropoutRate)
    : embedDim_(embedDim), numHeads_(numHeads), ffDim_(ffDim), dtype_(dtype), dropoutRate_(dropoutRate),
      norm1_(std::make_unique<LayerNorm>(embedDim, dtype)),
      attention_(std::make_unique<MultiHeadAttention>(embedDim, numHeads, dtype)),
      dropout1_(std::make_unique<Dropout>(dropoutRate)),
      norm2_(std::make_unique<LayerNorm>(embedDim, dtype)),
      ffn_(std::make_unique<FeedForward>(embedDim, ffDim, dtype)),
      dropout2_(std::make_unique<Dropout>(dropoutRate))
{
}

Tensor TransformerBlock::forward(const Tensor& input) {
    // Pre-LN architecture: LayerNorm before attention and FFN
    
    // First residual block: Attention
    Tensor norm1_out = norm1_->forward(input);
    Tensor attn_out = attention_->forward(norm1_out);
    attn_out = dropout1_->forward(attn_out);
    
    // Residual connection: input + attention output
    Tensor residual1_out = input + attn_out;
    
    // Second residual block: FeedForward
    Tensor norm2_out = norm2_->forward(residual1_out);
    Tensor ffn_out = ffn_->forward(norm2_out);
    ffn_out = dropout2_->forward(ffn_out);
    
    // Residual connection: residual1_out + FFN output
    Tensor output = residual1_out + ffn_out;
    
    return output;
}

std::vector<Tensor> TransformerBlock::parameters() {
    std::vector<Tensor> params;
    
    // Collect parameters from all sub-modules
    auto norm1_params = norm1_->parameters();
    if (!norm1_params.empty()) {
        params.insert(params.end(), norm1_params.begin(), norm1_params.end());
    }
    
    auto attn_params = attention_->parameters();
    if (!attn_params.empty()) {
        params.insert(params.end(), attn_params.begin(), attn_params.end());
    }
    
    auto norm2_params = norm2_->parameters();
    if (!norm2_params.empty()) {
        params.insert(params.end(), norm2_params.begin(), norm2_params.end());
    }
    
    auto ffn_params = ffn_->parameters();
    if (!ffn_params.empty()) {
        params.insert(params.end(), ffn_params.begin(), ffn_params.end());
    }
    
    return params;
}

void TransformerBlock::zero_grad() {
    norm1_->zero_grad();
    attention_->zero_grad();
    norm2_->zero_grad();
    ffn_->zero_grad();
}

void TransformerBlock::to_cuda() {
    norm1_->to_cuda();
    attention_->to_cuda();
    norm2_->to_cuda();
    ffn_->to_cuda();
}
