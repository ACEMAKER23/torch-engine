#ifndef TENSORDTYPE
#define TENSORDTYPE
#include <cstddef>
#include <stdexcept>

enum class DType {
    Float32,
    Int32,
    Int64
};

enum class Device {
    CPU,
    CUDA
};

inline size_t dtype_size(DType t) {
    switch (t) {
        case DType::Float32: return 4;
        case DType::Int32:   return 4;
        case DType::Int64:   return 8;
        default:
            throw std::runtime_error("Unknown DType");
    }
}
#endif