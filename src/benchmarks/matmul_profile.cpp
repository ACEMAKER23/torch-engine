// Lightweight harness for `ncu` (Nsight Compute) profiling.
//
// Unlike matmul_benchmark.cpp (which sweeps many shapes/iterations with CUDA
// events for wall-clock timing), this binary launches each kernel exactly
// once per requested shape so that `ncu` only has to instrument a handful of
// kernel launches instead of the whole benchmark sweep.
//
// Usage: matmul_profile <kernel_name> <M> <K> <N>
//   kernel_name one of: naive, shared_memory, register_blocking,
//     vectorized_input, warp_tiling, double_buffered,
//     double_buffered_cpasync, double_buffered_swizzled, vector_storage,
//     3stage_cpasync, tensor_core, ampere, tensor_core_padded, ampere_padded,
//     cublas_f32, cublas_tc

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../core/cuda_utils.h"
#include "../cuda/elementwise_cuda.h"

namespace {

void run_f32(void (*kernel)(const float*, const float*, float*, int, int, int), int M, int K, int N) {
    size_t A_elems = static_cast<size_t>(M) * K, B_elems = static_cast<size_t>(K) * N,
           C_elems = static_cast<size_t>(M) * N;
    std::vector<float> h_A(A_elems), h_B(B_elems);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : h_A) x = dist(gen);
    for (auto& x : h_B) x = dist(gen);

    float *d_A, *d_B, *d_C;
    cuda_check_error(cudaMalloc(&d_A, A_elems * sizeof(float)), "malloc A");
    cuda_check_error(cudaMalloc(&d_B, B_elems * sizeof(float)), "malloc B");
    cuda_check_error(cudaMalloc(&d_C, C_elems * sizeof(float)), "malloc C");
    cudaMemcpy(d_A, h_A.data(), A_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), B_elems * sizeof(float), cudaMemcpyHostToDevice);

    kernel(d_A, d_B, d_C, M, K, N);  // warmup / JIT
    cuda_check_error(cudaDeviceSynchronize(), "warmup sync");
    kernel(d_A, d_B, d_C, M, K, N);  // profiled launch
    cuda_check_error(cudaDeviceSynchronize(), "profiled sync");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void run_cublas_f32(int M, int K, int N) {
    size_t A_elems = static_cast<size_t>(M) * K, B_elems = static_cast<size_t>(K) * N,
           C_elems = static_cast<size_t>(M) * N;
    std::vector<float> h_A(A_elems), h_B(B_elems);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : h_A) x = dist(gen);
    for (auto& x : h_B) x = dist(gen);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A_elems * sizeof(float));
    cudaMalloc(&d_B, B_elems * sizeof(float));
    cudaMalloc(&d_C, C_elems * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), A_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), B_elems * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaDeviceSynchronize();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaDeviceSynchronize();

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void run_tc(void (*kernel)(const half*, const half*, float*, int, int, int), int M, int K, int N) {
    size_t A_elems = static_cast<size_t>(M) * K, B_elems = static_cast<size_t>(K) * N,
           C_elems = static_cast<size_t>(M) * N;
    std::vector<half> h_A(A_elems), h_B(B_elems);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& x : h_A) x = __float2half(dist(gen));
    for (auto& x : h_B) x = __float2half(dist(gen));

    half *d_A, *d_B;
    float* d_C;
    cuda_check_error(cudaMalloc(&d_A, A_elems * sizeof(half)), "malloc A");
    cuda_check_error(cudaMalloc(&d_B, B_elems * sizeof(half)), "malloc B");
    cuda_check_error(cudaMalloc(&d_C, C_elems * sizeof(float)), "malloc C");
    cudaMemcpy(d_A, h_A.data(), A_elems * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), B_elems * sizeof(half), cudaMemcpyHostToDevice);

    kernel(d_A, d_B, d_C, M, K, N);
    cuda_check_error(cudaDeviceSynchronize(), "warmup sync");
    kernel(d_A, d_B, d_C, M, K, N);
    cuda_check_error(cudaDeviceSynchronize(), "profiled sync");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void run_cublas_tc(int M, int K, int N) {
    size_t A_elems = static_cast<size_t>(M) * K, B_elems = static_cast<size_t>(K) * N,
           C_elems = static_cast<size_t>(M) * N;
    std::vector<half> h_A(A_elems), h_B(B_elems);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& x : h_A) x = __float2half(dist(gen));
    for (auto& x : h_B) x = __float2half(dist(gen));

    half *d_A, *d_B;
    float* d_C;
    cudaMalloc(&d_A, A_elems * sizeof(half));
    cudaMalloc(&d_B, B_elems * sizeof(half));
    cudaMalloc(&d_C, C_elems * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), A_elems * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), B_elems * sizeof(half), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, CUDA_R_16F, N, d_A, CUDA_R_16F, K, &beta,
                 d_C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, CUDA_R_16F, N, d_A, CUDA_R_16F, K, &beta,
                 d_C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <kernel_name> <M> <K> <N>\n";
        return 1;
    }
    std::string name = argv[1];
    int M = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);
    int N = std::atoi(argv[4]);

    if (!cuda_available()) {
        std::cerr << "CUDA not available\n";
        return 1;
    }

    if (name == "naive") run_f32(cuda_matmul, M, K, N);
    else if (name == "shared_memory") run_f32(cuda_matmul_shared_memory, M, K, N);
    else if (name == "register_blocking") run_f32(cuda_matmul_register_blocking, M, K, N);
    else if (name == "vectorized_input") run_f32(cuda_matmul_vectorized_input, M, K, N);
    else if (name == "warp_tiling") run_f32(cuda_matmul_warp_tiling, M, K, N);
    else if (name == "double_buffered") run_f32(cuda_matmul_double_buffered, M, K, N);
    else if (name == "double_buffered_cpasync") run_f32(cuda_matmul_double_buffered_cpasync, M, K, N);
    else if (name == "double_buffered_swizzled") run_f32(cuda_matmul_double_buffered_swizzled, M, K, N);
    else if (name == "vector_storage") run_f32(cuda_matmul_vector_storage, M, K, N);
    else if (name == "3stage_cpasync") run_f32(cuda_matmul_3stage_cpasync, M, K, N);
    else if (name == "cublas_f32") run_cublas_f32(M, K, N);
    else if (name == "tensor_core") run_tc(launch_matmul_tensor_core, M, K, N);
    else if (name == "ampere") run_tc(launch_matmul_ampere, M, K, N);
    else if (name == "tensor_core_padded") run_tc(launch_matmul_tensor_core, M, K, N);
    else if (name == "ampere_padded") run_tc(launch_matmul_ampere, M, K, N);
    else if (name == "cublas_tc") run_cublas_tc(M, K, N);
    else {
        std::cerr << "Unknown kernel name: " << name << "\n";
        return 1;
    }

    std::cout << "Done: " << name << " " << M << "x" << K << "x" << N << "\n";
    return 0;
}
