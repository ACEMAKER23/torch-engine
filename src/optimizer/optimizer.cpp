#include "optimizer.h"
#include "../core/dtype_utils.h"

void optimizer::zero_grad() {
    for (Tensor& t : processing_){
        if (t.requiredGrad() && t.grad()) {
            t.grad()->fill_<float>(0.0f);
        }
    }
}

optimizer::optimizer(const std::vector<Tensor>& paramters):
processing_(paramters)
, use_mixed_precision_(false)
{}

optimizer::optimizer(const std::vector<Tensor>& paramters, bool use_mixed_precision):
processing_(paramters)
, use_mixed_precision_(use_mixed_precision)
{
    if (use_mixed_precision_) {
        // Create master weights in FP32
        for (const Tensor& param : paramters) {
            if (is_half_precision(param.dtype())) {
                Tensor master_weight = cast_dtype(param, DType::Float32);
                master_weights_.push_back(master_weight);
            } else {
                master_weights_.push_back(param.clone());
            }
        }
    }
}

void optimizer::set_mixed_precision(bool enable) {
    use_mixed_precision_ = enable;
    if (enable && master_weights_.empty()) {
        // Initialize master weights if not already done
        for (const Tensor& param : processing_) {
            if (is_half_precision(param.dtype())) {
                Tensor master_weight = cast_dtype(param, DType::Float32);
                master_weights_.push_back(master_weight);
            } else {
                master_weights_.push_back(param.clone());
            }
        }
    }
}

void optimizer::sync_master_weights() {
    if (!use_mixed_precision_) return;
    
    // Copy model weights to master weights (before forward pass)
    for (size_t i = 0; i < processing_.size(); ++i) {
        if (is_half_precision(processing_[i].dtype())) {
            Tensor converted = cast_dtype(processing_[i], DType::Float32);
            for (size_t j = 0; j < converted.numel(); ++j) {
                master_weights_[i].at<float>(j) = converted.at<float>(j);
            }
        }
    }
}

void optimizer::sync_model_weights() {
    if (!use_mixed_precision_) return;
    
    // Copy master weights back to model weights (after optimizer step)
    for (size_t i = 0; i < processing_.size(); ++i) {
        if (is_half_precision(processing_[i].dtype())) {
            Tensor converted = cast_dtype(master_weights_[i], processing_[i].dtype());
            // Access raw data for uint16_t types
            auto* model_data = static_cast<uint16_t*>(processing_[i].data());
            auto* converted_data = static_cast<uint16_t*>(converted.data());
            for (size_t j = 0; j < converted.numel(); ++j) {
                model_data[j] = converted_data[j];
            }
        }
    }
}