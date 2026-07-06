#include "dtype_utils.h"
#include <cmath>
#include <cstring>

// Helper for bit casting using memcpy (C++17 compatible)
template<typename To, typename From>
To bit_cast(const From& from) {
    static_assert(sizeof(To) == sizeof(From), "Size mismatch");
    To to;
    std::memcpy(&to, &from, sizeof(To));
    return to;
}

// Float16 conversion
float16_t float16_t::from_float32(float f) {
    float16_t result;
    uint32_t f_bits = bit_cast<uint32_t>(f);
    
    // Extract sign, exponent, mantissa
    uint32_t sign = (f_bits >> 31) & 0x1;
    uint32_t exponent = (f_bits >> 23) & 0xFF;
    uint32_t mantissa = f_bits & 0x7FFFFF;
    
    if (exponent == 0xFF) {
        // Inf or NaN
        result.bits = (sign << 15) | 0x7C00 | (mantissa >> 13);
    } else if (exponent >= 103) {
        // Normal number
        int new_exp = exponent - 127 + 15;
        if (new_exp >= 31) {
            // Overflow to Inf
            result.bits = (sign << 15) | 0x7C00;
        } else if (new_exp <= 0) {
            // Underflow to zero or subnormal
            if (new_exp < -10) {
                result.bits = sign << 15;
            } else {
                mantissa |= 0x800000;
                result.bits = (sign << 15) | (mantissa >> (14 - new_exp));
            }
        } else {
            result.bits = (sign << 15) | (new_exp << 10) | (mantissa >> 13);
        }
    } else {
        // Very small number, underflow to zero
        result.bits = sign << 15;
    }
    
    return result;
}

float float16_t::to_float32() const {
    uint32_t sign = (bits >> 15) & 0x1;
    uint32_t exponent = (bits >> 10) & 0x1F;
    uint32_t mantissa = bits & 0x3FF;
    
    uint32_t f_bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            f_bits = sign << 31;
        } else {
            // Subnormal
            f_bits = sign << 31;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            exponent++;
            mantissa &= 0x3FF;
            f_bits |= ((exponent + 127) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        // Inf or NaN
        f_bits = (sign << 31) | 0x7F800000 | (mantissa << 13);
    } else {
        // Normal number
        f_bits = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    
    return bit_cast<float>(f_bits);
}

// BFloat16 conversion (truncated float32)
bfloat16_t bfloat16_t::from_float32(float f) {
    bfloat16_t result;
    uint32_t f_bits = bit_cast<uint32_t>(f);
    
    // BFloat16 is just the top 16 bits of float32
    result.bits = static_cast<uint16_t>(f_bits >> 16);
    
    return result;
}

float bfloat16_t::to_float32() const {
    // BFloat16 to float32: zero-extend the lower 16 bits
    uint32_t f_bits = static_cast<uint32_t>(bits) << 16;
    return bit_cast<float>(f_bits);
}

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
