#ifndef ELEMENTWISE_CUDA_H
#define ELEMENTWISE_CUDA_H

#include <cstdint>
#include "../core/dtype_utils.h"

// C++ overloaded elementwise CUDA operations
void cuda_add(const float* a, const float* b, float* out, int64_t size);
void cuda_add(const int32_t* a, const int32_t* b, int32_t* out, int64_t size);
void cuda_add(const int64_t* a, const int64_t* b, int64_t* out, int64_t size);
void cuda_add(const float16_t* a, const float16_t* b, float16_t* out, int64_t size);
void cuda_add(const bfloat16_t* a, const bfloat16_t* b, bfloat16_t* out, int64_t size);

void cuda_sub(const float* a, const float* b, float* out, int64_t size);
void cuda_sub(const int32_t* a, const int32_t* b, int32_t* out, int64_t size);
void cuda_sub(const int64_t* a, const int64_t* b, int64_t* out, int64_t size);
void cuda_sub(const float16_t* a, const float16_t* b, float16_t* out, int64_t size);
void cuda_sub(const bfloat16_t* a, const bfloat16_t* b, bfloat16_t* out, int64_t size);

void cuda_mul(const float* a, const float* b, float* out, int64_t size);
void cuda_mul(const int32_t* a, const int32_t* b, int32_t* out, int64_t size);
void cuda_mul(const int64_t* a, const int64_t* b, int64_t* out, int64_t size);
void cuda_mul(const float16_t* a, const float16_t* b, float16_t* out, int64_t size);
void cuda_mul(const bfloat16_t* a, const bfloat16_t* b, bfloat16_t* out, int64_t size);

void cuda_div(const float* a, const float* b, float* out, int64_t size);
void cuda_div(const int32_t* a, const int32_t* b, int32_t* out, int64_t size);
void cuda_div(const int64_t* a, const int64_t* b, int64_t* out, int64_t size);
void cuda_div(const float16_t* a, const float16_t* b, float16_t* out, int64_t size);
void cuda_div(const bfloat16_t* a, const bfloat16_t* b, bfloat16_t* out, int64_t size);

void cuda_relu(const float* input, float* out, int64_t size);
void cuda_relu(const float16_t* input, float16_t* out, int64_t size);
void cuda_relu(const bfloat16_t* input, bfloat16_t* out, int64_t size);

void cuda_matmul(const float* A, const float* B, float* C, int M, int K, int N);

#endif
