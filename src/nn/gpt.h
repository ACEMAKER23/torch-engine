#ifndef GPTLAYER
#define GPTLAYER

#include "../tensor/tensor.h"
#include "module.h"
#include "embedding.h"
#include "positional_embedding.h"
#include "transformer_block.h"
#include "layernorm.h"
#include "linear.h"
#include "dropout.h"
#include <memory>
#include <vector>
#include <cstdint>

class GPT : public Module {
public:
    GPT(int64_t vocabSize, int64_t maxSeqLen, int64_t embedDim, 
        int64_t numHeads, int64_t numLayers, int64_t ffDim, 
        DType dtype, float dropoutRate = 0.1);

    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    void zero_grad() override;
    void to_cuda() override;

private:
    std::unique_ptr<Embedding> tokenEmbedding_;
    std::unique_ptr<PositionalEmbedding> posEmbedding_;
    std::unique_ptr<Dropout> embedDropout_;
    
    std::vector<std::unique_ptr<TransformerBlock>> blocks_;
    
    std::unique_ptr<LayerNorm> finalNorm_;
    std::unique_ptr<Linear> lmHead_;
    
    int64_t vocabSize_;
    int64_t maxSeqLen_;
    int64_t embedDim_;
    int64_t numHeads_;
    int64_t numLayers_;
    int64_t ffDim_;
    DType dtype_;
    float dropoutRate_;
};

#endif
