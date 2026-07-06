#ifndef LOSS_SCALER
#define LOSS_SCALER
#include "../tensor/tensor.h"

class LossScaler {
public:
    LossScaler(float scale_factor = 2.0f, float max_scale = 65536.0f, float growth_factor = 2.0f, float backoff_factor = 0.5f, size_t growth_interval = 2000);
    
    // Scale the loss for forward pass
    Tensor scale(const Tensor& loss);
    
    // Unscaled gradients during backward pass
    Tensor unscale(const Tensor& scaled_grad);
    
    // Check if gradients are inf/nan and adjust scale
    bool check_and_adjust_scale(const Tensor& grad);
    
    // Get current scale factor
    float get_scale() const { return current_scale_; }
    
    // Reset scale to initial value
    void reset();
    
private:
    float initial_scale_;
    float current_scale_;
    float max_scale_;
    float growth_factor_;
    float backoff_factor_;
    size_t growth_interval_;
    size_t steps_since_last_adjustment_;
    
    // Check if tensor contains inf or nan
    bool has_inf_or_nan(const Tensor& tensor);
};

#endif
