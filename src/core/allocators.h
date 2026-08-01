#ifndef ALLOCATOR
#define ALLOCATOR

#include <cstdlib>
#include "dtype.h"

#ifdef USE_CUDA
#include "cuda_utils.h"
#endif

class Allocator{
public:
    virtual ~Allocator() = default;
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual Device device() const = 0;
};

class CPUAllocator : public Allocator{
    //return a void pointer to any memory on cpu of size bytes
    void* allocate(size_t bytes) override {
        return (std::malloc(bytes));
    }

    void deallocate(void* ptr) override {
        std::free(ptr);
    }

    Device device() const override {
        return (Device::CPU);
    }
};

#ifdef USE_CUDA
class CUDAAllocator : public Allocator{
    //return a void pointer to any memory on gpu of size bytes
    void* allocate(size_t bytes) override {
        void* ptr;
        cuda_check_error(cudaMalloc(&ptr, bytes), "cudaMalloc failed");
        return ptr;
    }

    void deallocate(void* ptr) override {
        if (ptr != nullptr) {
            cuda_check_error(cudaFree(ptr), "cudaFree failed");
        }
    }

    Device device() const override {
        return (Device::CUDA);
    }
};
#endif
#endif