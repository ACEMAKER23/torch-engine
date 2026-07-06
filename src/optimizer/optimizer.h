#ifndef OPTIMIZER
#define OPTIMIZER

#include "../tensor/tensor.h"
#include "../core/dtype.h"

class optimizer {
public:
    optimizer(const std::vector<Tensor>& paramters);
    optimizer(const std::vector<Tensor>& paramters, bool use_mixed_precision);

    virtual void step(float learningRate) = 0;
    virtual void zero_grad();
    
    // Enable/disable mixed precision training
    void set_mixed_precision(bool enable);
    
    // Get master weights (FP32 copies when using mixed precision)
    const std::vector<Tensor>& get_master_weights() const { return master_weights_; }
    
protected:
    std::vector<Tensor> processing_;
    bool use_mixed_precision_;
    std::vector<Tensor> master_weights_; // FP32 master weights for mixed precision
    
    // Sync master weights with model weights
    void sync_master_weights();
    // Sync model weights with master weights
    void sync_model_weights();
};


#endif