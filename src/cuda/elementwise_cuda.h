#ifndef ELEMENTWISE_CUDA_H
#define ELEMENTWISE_CUDA_H

#include <cstdint>
#include <cuda_fp16.h>
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

void cuda_softmax_forward(const float* x, float* out, int64_t outer_size, int64_t dim_size);
void cuda_softmax_backward(const float* s, const float* dy, float* out, int64_t outer_size, int64_t dim_size);
void cuda_fill(float* data, float value, int64_t size);

void cuda_layernorm_forward(const float* x, const float* gamma, const float* beta,
                            float* out, int batch, int D, float eps);
void cuda_layernorm_backward(const float* x, const float* gamma, const float* dy,
                             float* dx, float* dg, float* db,
                             int batch, int D, float eps);

void cuda_embedding_forward(const int64_t* indices, const float* weight,
                            float* out, int total_indices, int D);
void cuda_embedding_backward(const int64_t* indices, const float* grad_out,
                             float* grad_weight, int total_indices, int D);

void cuda_crossentropy_batched_forward(const float* logits, const int64_t* targets,
                                       float* loss, int B, int T, int V);
void cuda_crossentropy_batched_backward(const float* logits, const int64_t* targets,
                                        float* grad, float scale, int B, int T, int V);

void cuda_gelu(const float* input, float* output, int64_t size);
void cuda_gelu_backward(const float* input, const float* grad_output,
                        float* grad_input, int64_t size);

void cuda_reduce_sum(const float* input, float* output, int64_t size);
void cuda_sum_dim(const float* input, float* output,
                  const int64_t* in_shape, const int64_t* in_strides,
                  const int64_t* out_shape, const int64_t* out_strides,
                  int64_t dim, int64_t ndim, int64_t out_numel, int64_t dim_size);

void cuda_matmul(const float* A, const float* B, float* C, int M, int K, int N);
void cuda_matmul_shared_memory(const float* A, const float* B, float* C, int M, int K, int N);
void cuda_matmul_register_blocking(const float* A, const float* B, float* C, int M, int K, int N);
void cuda_matmul_vectorized_input(const float* A, const float* B, float* C, int M, int K, int N);
void cuda_matmul_warp_tiling(const float* A, const float* B, float* C, int M, int K, int N);
void cuda_matmul_double_buffered(const float* A, const float* B, float* C, int M, int K, int N);
void cuda_matmul_double_buffered_cpasync(const float* A, const float* B, float* C, int M, int K, int N);
void cuda_matmul_double_buffered_swizzled(const float* A, const float* B, float* C, int M, int K, int N);
void cuda_matmul_vector_storage(const float* A, const float* B, float* C, int M, int K, int N);
void cuda_matmul_3stage_cpasync(const float* A, const float* B, float* C, int M, int K, int N);

// FP16 input / FP16 output non-tensor-core matmul overloads
void cuda_matmul(const half* A, const half* B, half* C, int M, int K, int N);
void cuda_matmul_shared_memory(const half* A, const half* B, half* C, int M, int K, int N);
void cuda_matmul_register_blocking(const half* A, const half* B, half* C, int M, int K, int N);
void cuda_matmul_vectorized_input(const half* A, const half* B, half* C, int M, int K, int N);
void cuda_matmul_warp_tiling(const half* A, const half* B, half* C, int M, int K, int N);
void cuda_matmul_double_buffered(const half* A, const half* B, half* C, int M, int K, int N);
void cuda_matmul_double_buffered_cpasync(const half* A, const half* B, half* C, int M, int K, int N);
void cuda_matmul_double_buffered_swizzled(const half* A, const half* B, half* C, int M, int K, int N);
void cuda_matmul_vector_storage(const half* A, const half* B, half* C, int M, int K, int N);
void cuda_matmul_3stage_cpasync(const half* A, const half* B, half* C, int M, int K, int N);

// FP16 input / FP32 output tensor-core launchers
void launch_matmul_tensor_core(const half* A, const half* B, float* C, int M, int K, int N);
void launch_matmul_ampere(const half* A, const half* B, float* C, int M, int K, int N);

#endif
