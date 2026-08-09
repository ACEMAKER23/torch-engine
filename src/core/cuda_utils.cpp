#include "cuda_utils.h"
#include <iostream>

bool cuda_available() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

int cuda_device_count() {
    int count;
    cuda_check_error(cudaGetDeviceCount(&count), "cudaGetDeviceCount failed");
    return count;
}

int cuda_get_device() {
    int device;
    cuda_check_error(cudaGetDevice(&device), "cudaGetDevice failed");
    return device;
}

void cuda_set_device(int device) {
    cuda_check_error(cudaSetDevice(device), "cudaSetDevice failed");
}

std::string cuda_device_name(int device) {
    cudaDeviceProp prop;
    cuda_check_error(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties failed");
    return std::string(prop.name);
}

void cuda_device_capability(int device, int* major, int* minor) {
    cudaDeviceProp prop;
    cuda_check_error(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties failed");
    *major = prop.major;
    *minor = prop.minor;
}

void cuda_init() {
    if (!cuda_available()) {
        std::cerr << "CUDA is not available" << std::endl;
        return;
    }
    std::cout << "CUDA initialized. Device: " << cuda_device_name() << std::endl;
}

#ifdef USE_CUDA
cublasHandle_t cuda_cublas_handle() {
    static cublasHandle_t handle = []() {
        cublasHandle_t h;
        cublas_check_error(cublasCreate(&h), "cublasCreate");
        return h;
    }();
    return handle;
}
#endif
