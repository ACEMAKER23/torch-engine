#ifndef TRANSFORMER
#define TRANSFORMER

#include "module.h"
#include "layernorm.h"
#include "multi_head_attention.h"
#include "ffn.h"
#include "dropout.h"
#include <memory>
#include <vector>

class TransformerBlock : public Module {
public:
    TransformerBlock(int64_t embedDim, int64_t numHeads, int64_t ffDim, 
                    DType dtype, float dropoutRate = 0.1);
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    void zero_grad() override;
    void to_cuda() override;

private:
    std::unique_ptr<LayerNorm> norm1_;
    std::unique_ptr<MultiHeadAttention> attention_;
    std::unique_ptr<Dropout> dropout1_;
    
    std::unique_ptr<LayerNorm> norm2_;
    std::unique_ptr<FeedForward> ffn_;
    std::unique_ptr<Dropout> dropout2_;
    
    int64_t embedDim_;
    int64_t numHeads_;
    int64_t ffDim_;
    DType dtype_;
    float dropoutRate_;
};

#endif

