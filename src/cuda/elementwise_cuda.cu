#include <cuda_runtime.h>
 
// Elementwise add kernel
template<typename T>
__global__ void add_kernel(const T* a, const T* b, T* out, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}
 
// Elementwise sub kernel
template<typename T>
__global__ void sub_kernel(const T* a, const T* b, T* out, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}
 
// Elementwise mul kernel
template<typename T>
__global__ void mul_kernel(const T* a, const T* b, T* out, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}
 
// Elementwise div kernel
template<typename T>
__global__ void div_kernel(const T* a, const T* b, T* out, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] / b[idx];
    }
}
 
// ReLU activation kernel
template<typename T>
__global__ void relu_kernel(const T* in, T* out, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = in[idx]>0 ? in[idx] : 0;
    }
}
 
// Naive matrix multiplication kernel
// C = A * B where A is (M x K), B is (K x N), C is (M x N)
template<typename T>
__global__ void matmul_kernel(const T* A, const T* B, T* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        T sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
 

// Wrapper functions for launching kernels
extern "C" {
    void cuda_add_float(const float* a, const float* b, float* out, int64_t size) {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        add_kernel<float><<<blocks, threads>>>(a, b, out, size);
    }
 
    void cuda_sub_float(const float* a, const float* b, float* out, int64_t size) {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        sub_kernel<float><<<blocks, threads>>>(a, b, out, size);
    }
 
    void cuda_mul_float(const float* a, const float* b, float* out, int64_t size) {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        mul_kernel<float><<<blocks, threads>>>(a, b, out, size);
    }
 
    void cuda_div_float(const float* a, const float* b, float* out, int64_t size) {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        div_kernel<float><<<blocks, threads>>>(a, b, out, size);
    }

    void cuda_relu_float(const float* input, float* out, int64_t size) {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        relu_kernel<float><<<blocks, threads>>>(input, out, size);
    }

    void cuda_matmul_float(const float* A, const float* B, float* C, int M, int K, int N) {
        dim3 threads(16, 16);
        dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
        matmul_kernel<float><<<blocks, threads>>>(A, B, C, M, K, N);
    }
}