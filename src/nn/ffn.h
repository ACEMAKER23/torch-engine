#ifndef FFN 
#define FFN 

#include "module.h"
#include "linear.h"
#include "gelu.h"
#include <cstdint>
#include <memory>

class FeedForward : public Module {
public:
    FeedForward(int64_t embeddingDim, int64_t forwardDim, DType dtype);
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    void zero_grad() override;
    void to_cuda() override;

private:    
    std::unique_ptr<Linear> linear1_;
    std::unique_ptr<GeLU> gelu_;
    std::unique_ptr<Linear> linear2_;
};


#endif    
