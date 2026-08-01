#ifndef CUDA_UTILS
#define CUDA_UTILS

#include <cuda_runtime.h>
#include <string>
#include <stdexcept>

// CUDA error checking helper
inline void cuda_check_error(cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(err));
    }
}

// Synchronize CUDA device
inline void cuda_synchronize() {
    cuda_check_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
}

// Check if CUDA is available
bool cuda_available();

// Get number of CUDA devices
int cuda_device_count();

// Get current device
int cuda_get_device();

// Set CUDA device
void cuda_set_device(int device);

// Get device name
std::string cuda_device_name(int device = 0);

// Get device capability (major.minor)
void cuda_device_capability(int device, int* major, int* minor);

// Initialize CUDA runtime (call at program start)
void cuda_init();

#endif