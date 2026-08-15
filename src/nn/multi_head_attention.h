#ifndef MULTIHEAD
#define MULTIHEAD
#include "module.h"
#include "linear.h"
#include "scaledot.h"
#include <memory>
class MultiHeadAttention : public Module {
public:
    MultiHeadAttention(int64_t embedDim, int64_t numHeads, DType dtype);
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    void zero_grad() override;
    void to_cuda() override;

private:
    std::unique_ptr<Linear> qProj_;
    std::unique_ptr<Linear> kProj_;
    std::unique_ptr<Linear> vProj_;
    std::unique_ptr<Linear> outProj_;
    std::unique_ptr<ScaledDot> attention_;
    int64_t embedDim_;
    int64_t numHeads_;
    DType dtype_;

};




#endif
