#include "cuda_utils.h"
#include <iostream>
 
bool cuda_available() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}
 
int cuda_device_count() {
    int count;
    cudaGetDeviceCount(&count);
    return count;
}
 
int cuda_get_device() {
    int device;
    cudaGetDevice(&device);
    return device;
}
 
bool cuda_set_device(int device) {
    cudaError_t err = cudaSetDevice(device);
    return (err == cudaSuccess);
}
 
std::string cuda_device_name(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return std::string(prop.name);
}
 
void cuda_device_capability(int device, int* major, int* minor) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
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
