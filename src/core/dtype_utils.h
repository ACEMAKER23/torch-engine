#ifndef DTYPE_UTILS
#define DTYPE_UTILS
#include "dtype.h"
#include "../tensor/tensor.h"
#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// Helper for bit casting using memcpy (C++17 compatible)
inline HD uint32_t float_to_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return bits;
}

inline HD float bits_to_float(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Float16 representation
struct float16_t {
    uint16_t bits;

    // Convert from float32 to float16
    static HD float16_t from_float32(float f) {
        float16_t result;
        uint32_t f_bits = float_to_bits(f);

        uint32_t sign = (f_bits >> 31) & 0x1;
        uint32_t exponent = (f_bits >> 23) & 0xFF;
        uint32_t mantissa = f_bits & 0x7FFFFF;

        if (exponent == 0xFF) {
            result.bits = static_cast<uint16_t>((sign << 15) | 0x7C00 | (mantissa >> 13));
        } else if (exponent >= 103) {
            int new_exp = static_cast<int>(exponent) - 127 + 15;
            if (new_exp >= 31) {
                result.bits = static_cast<uint16_t>((sign << 15) | 0x7C00);
            } else if (new_exp <= 0) {
                if (new_exp < -10) {
                    result.bits = static_cast<uint16_t>(sign << 15);
                } else {
                    mantissa |= 0x800000;
                    result.bits = static_cast<uint16_t>((sign << 15) | (mantissa >> (14 - new_exp)));
                }
            } else {
                result.bits = static_cast<uint16_t>((sign << 15) | (new_exp << 10) | (mantissa >> 13));
            }
        } else {
            result.bits = static_cast<uint16_t>(sign << 15);
        }

        return result;
    }

    // Convert from float16 to float32
    HD float to_float32() const {
        uint32_t sign = (bits >> 15) & 0x1;
        uint32_t exponent = (bits >> 10) & 0x1F;
        uint32_t mantissa = bits & 0x3FF;

        uint32_t f_bits;
        if (exponent == 0) {
            if (mantissa == 0) {
                f_bits = sign << 31;
            } else {
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
            f_bits = (sign << 31) | 0x7F800000 | (mantissa << 13);
        } else {
            f_bits = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }

        return bits_to_float(f_bits);
    }
};

inline HD float16_t operator+(float16_t a, float16_t b) { return float16_t::from_float32(a.to_float32() + b.to_float32()); }
inline HD float16_t operator-(float16_t a, float16_t b) { return float16_t::from_float32(a.to_float32() - b.to_float32()); }
inline HD float16_t operator*(float16_t a, float16_t b) { return float16_t::from_float32(a.to_float32() * b.to_float32()); }
inline HD float16_t operator/(float16_t a, float16_t b) { return float16_t::from_float32(a.to_float32() / b.to_float32()); }
inline HD float16_t& operator+=(float16_t& a, float16_t b) { a = a + b; return a; }
inline HD bool operator>(float16_t a, float b) { return a.to_float32() > b; }
inline HD bool operator<(float16_t a, float b) { return a.to_float32() < b; }

// BFloat16 representation (truncated float32)
struct bfloat16_t {
    uint16_t bits;

    // Convert from float32 to bfloat16
    static HD bfloat16_t from_float32(float f) {
        bfloat16_t result;
        uint32_t f_bits = float_to_bits(f);
        result.bits = static_cast<uint16_t>(f_bits >> 16);
        return result;
    }

    // Convert from bfloat16 to float32
    HD float to_float32() const {
        uint32_t f_bits = static_cast<uint32_t>(bits) << 16;
        return bits_to_float(f_bits);
    }
};

inline HD bfloat16_t operator+(bfloat16_t a, bfloat16_t b) { return bfloat16_t::from_float32(a.to_float32() + b.to_float32()); }
inline HD bfloat16_t operator-(bfloat16_t a, bfloat16_t b) { return bfloat16_t::from_float32(a.to_float32() - b.to_float32()); }
inline HD bfloat16_t operator*(bfloat16_t a, bfloat16_t b) { return bfloat16_t::from_float32(a.to_float32() * b.to_float32()); }
inline HD bfloat16_t operator/(bfloat16_t a, bfloat16_t b) { return bfloat16_t::from_float32(a.to_float32() / b.to_float32()); }
inline HD bfloat16_t& operator+=(bfloat16_t& a, bfloat16_t b) { a = a + b; return a; }
inline HD bool operator>(bfloat16_t a, float b) { return a.to_float32() > b; }
inline HD bool operator<(bfloat16_t a, float b) { return a.to_float32() < b; }

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


