#ifndef ALLOCATOR
#define ALLOCATOR

#include <cstdlib>
#include "dtype.h"
#include <cuda_runtime.h>

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

class CUDAAllocatorPlaceHolder : public Allocator{
    //return a void pointer to any memory on gpu of size bytes
    void* allocate(size_t bytes) override {
        void* ptr;
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess) {
            return nullptr;
        }
        return ptr;
    }

    void deallocate(void* ptr) override {
        cudaFree(ptr);
    }

    Device device() const override {
        return (Device::CUDA);
    }
};
#endif