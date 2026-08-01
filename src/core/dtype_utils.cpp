#include "dtype_utils.h"
#include <cmath>
#include <cstring>

// Dtype conversion
Tensor cast_dtype(const Tensor& tensor, DType target_dtype) {
    if (tensor.dtype() == target_dtype) {
        return tensor.clone();
    }
    
    Tensor result(tensor.shape(), target_dtype, tensor.device());
    
    // Convert element by element
    for (size_t i = 0; i < tensor.numel(); ++i) {
        float value = tensor.at<float>(i);
        
        switch (target_dtype) {
            case DType::Float32:
                result.at<float>(i) = value;
                break;
            case DType::Float16:
                // Access raw data for Float16
                {
                    auto* data = static_cast<uint16_t*>(result.data());
                    data[i] = float16_t::from_float32(value).bits;
                }
                break;
            case DType::BFloat16:
                // Access raw data for BFloat16
                {
                    auto* data = static_cast<uint16_t*>(result.data());
                    data[i] = bfloat16_t::from_float32(value).bits;
                }
                break;
            case DType::Int32:
                result.at<int32_t>(i) = static_cast<int32_t>(value);
                break;
            case DType::Int64:
                result.at<int64_t>(i) = static_cast<int64_t>(value);
                break;
        }
    }
    
    return result;
}
