#ifndef EMBEDDING
#define EMBEDDING

#include "../tensor/tensor.h"
#include "module.h"

class Embedding : public Module {
public:
    Embedding(int64_t num_embeddings, int64_t embedding_dim, DType dtype);
    Tensor forward(const Tensor& input) override;

    const Tensor& weight() const { return weight_; };
    
    std::vector<Tensor> parameters() override;
    void zero_grad() override;

private:
    int64_t num_embeddings_;
    int64_t embedding_dim_;
    Tensor weight_;

    void init_parameters();
};

#endif
