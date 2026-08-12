#include "multi_head_attention.h"
#include <cassert>

MultiHeadAttention::MultiHeadAttention(int64_t embedDim, int64_t numHeads, DType dtype)
    : qProj_(std::make_unique<Linear>(embedDim, embedDim, dtype)),
      kProj_(std::make_unique<Linear>(embedDim, embedDim, dtype)),
      vProj_(std::make_unique<Linear>(embedDim, embedDim, dtype)),
      outProj_(std::make_unique<Linear>(embedDim, embedDim, dtype)),
      attention_(std::make_unique<ScaledDot>()),
      embedDim_(embedDim),
    numHeads_(numHeads),
     dtype_(dtype) {
    if (embedDim_ % numHeads_ != 0) {
        throw std::runtime_error("embedDim must be divisible by numHeads");
    }
}


Tensor MultiHeadAttention::forward(const Tensor& input) {
    Tensor Q = qProj_->forward(input);
    Tensor K = kProj_->forward(input);
    Tensor V = vProj_->forward(input);
    int64_t headDim = embedDim_ / numHeads_;

    // Reshape to [batch, seq, heads, headDim] and then swap heads/seq so the
    // attention acts on [batch, heads, seq, headDim].
    Q = Q.view({input.shape()[0], input.shape()[1], numHeads_, headDim})
          .transpose_view(1, 2);
    K = K.view({input.shape()[0], input.shape()[1], numHeads_, headDim})
          .transpose_view(1, 2);
    V = V.view({input.shape()[0], input.shape()[1], numHeads_, headDim})
          .transpose_view(1, 2);

    Tensor sftmax = attention_->forward(Q, K, V);
    // Swap back to [batch, seq, heads, headDim] and flatten to [batch, seq, E].
    // reshape() handles the non-contiguous transpose_view with a single
    // ReshapeBackward node (vs CopyBackward + ViewBackward from contiguous().view()).
    sftmax = sftmax.transpose_view(1, 2)
                   .reshape({input.shape()[0], input.shape()[1], embedDim_});

    return outProj_->forward(sftmax);
}

std::vector<Tensor> MultiHeadAttention::parameters() {
    std::vector<Tensor> params;
    auto qParams = qProj_->parameters();
    auto kParams = kProj_->parameters();
    auto vParams = vProj_->parameters();
    auto outParams = outProj_->parameters();
    
    params.insert(params.end(), qParams.begin(), qParams.end());
    params.insert(params.end(), kParams.begin(), kParams.end());
    params.insert(params.end(), vParams.begin(), vParams.end());
    params.insert(params.end(), outParams.begin(), outParams.end());
    
    return params;
}

void MultiHeadAttention::zero_grad() {
    qProj_->zero_grad();
    kProj_->zero_grad();
    vProj_->zero_grad();
    outProj_->zero_grad();
}

void MultiHeadAttention::to_cuda() {
    qProj_->to_cuda();
    kProj_->to_cuda();
    vProj_->to_cuda();
    outProj_->to_cuda();
}
