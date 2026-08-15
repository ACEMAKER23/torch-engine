#ifndef POSITIONAL_EMBEDDING
#define POSITIONAL_EMBEDDING

#include "module.h"
#include <cmath>

class PositionalEmbedding : public Module {
public:
    PositionalEmbedding(int64_t max_seq_len, int64_t d_model, DType dtype);
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override { return {}; }
    void zero_grad() override {}
    void to_cuda() override;

private:
    int64_t max_seq_len_;
    int64_t d_model_;
    Tensor pe_;
    
    void init_positional_embeddings();
};

#endif
