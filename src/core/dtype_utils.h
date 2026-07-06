#ifndef DTYPE_UTILS
#define DTYPE_UTILS
#include "dtype.h"
#include "../tensor/tensor.h"
#include <cstdint>

// Float16 representation
struct float16_t {
    uint16_t bits;
    
    // Convert from float32 to float16
    static float16_t from_float32(float f);
    
    // Convert from float16 to float32
    float to_float32() const;
};

// BFloat16 representation (truncated float32)
struct bfloat16_t {
    uint16_t bits;
    
    // Convert from float32 to bfloat16
    static bfloat16_t from_float32(float f);
    
    // Convert from bfloat16 to float32
    float to_float32() const;
};

// Dtype conversion utilities
Tensor cast_dtype(const Tensor& tensor, DType target_dtype);

// Check if dtype is floating point
inline bool is_dtype_floating_point(DType dtype) {
    return dtype == DType::Float32 || dtype == DType::Float16 || dtype == DType::BFloat16;
}

// Check if dtype is lower precision (float16/bfloat16)
inline bool is_half_precision(DType dtype) {
    return dtype == DType::Float16 || dtype == DType::BFloat16;
}

// Get higher precision dtype for master weights
inline DType get_master_dtype(DType dtype) {
    if (is_half_precision(dtype)) {
        return DType::Float32;
    }
    return dtype;
}

#endif
