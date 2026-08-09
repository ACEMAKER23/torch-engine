#include <iostream>
#include <cmath>
#include <cstring>
#include "../tensor/tensor.h"
#include "../core/cuda_utils.h"

int main() {
    if (!cuda_available()) {
        std::cerr << "CUDA not available\n";
        return 1;
    }

    // Full reduction sum/mean.
    const int size = 6;
    float x_h[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor x({size}, DType::Float32, Device::CPU);
    std::memcpy(x.data(), x_h, size * sizeof(float));
    Tensor x_cuda = x.toDevice(Device::CUDA);

    float expected_sum = 21.0f;
    float expected_mean = 3.5f;

    float cuda_sum = x_cuda.sum<float>();
    float cuda_mean = x_cuda.mean<float>();

    bool ok = true;
    if (std::fabs(cuda_sum - expected_sum) > 1e-4f) {
        std::cout << "full sum mismatch: got " << cuda_sum << " expected " << expected_sum << "\n";
        ok = false;
    }
    if (std::fabs(cuda_mean - expected_mean) > 1e-4f) {
        std::cout << "full mean mismatch: got " << cuda_mean << " expected " << expected_mean << "\n";
        ok = false;
    }

    // Sum over dimension 0: [3, 4] -> [4]
    float x2_h[12] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    };
    Tensor x2({3, 4}, DType::Float32, Device::CPU);
    std::memcpy(x2.data(), x2_h, 12 * sizeof(float));
    Tensor x2_cuda = x2.toDevice(Device::CUDA);

    Tensor s0 = x2_cuda.sum(0);
    float s0_h[4];
    cudaMemcpy(s0_h, s0.data(), 4 * sizeof(float), cudaMemcpyDeviceToHost);
    float expected_s0[4] = {15.0f, 18.0f, 21.0f, 24.0f};
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(s0_h[i] - expected_s0[i]) > 1e-4f) {
            std::cout << "sum(0) mismatch at " << i << ": got " << s0_h[i]
                      << " expected " << expected_s0[i] << "\n";
            ok = false;
        }
    }

    // Sum over dimension 1: [3, 4] -> [3]
    Tensor s1 = x2_cuda.sum(1);
    float s1_h[3];
    cudaMemcpy(s1_h, s1.data(), 3 * sizeof(float), cudaMemcpyDeviceToHost);
    float expected_s1[3] = {10.0f, 26.0f, 42.0f};
    for (int i = 0; i < 3; ++i) {
        if (std::fabs(s1_h[i] - expected_s1[i]) > 1e-4f) {
            std::cout << "sum(1) mismatch at " << i << ": got " << s1_h[i]
                      << " expected " << expected_s1[i] << "\n";
            ok = false;
        }
    }

    // 3D sum over middle dimension: [2, 3, 4] -> [2, 4]
    float x3_h[24];
    for (int i = 0; i < 24; ++i) x3_h[i] = static_cast<float>(i + 1);
    Tensor x3({2, 3, 4}, DType::Float32, Device::CPU);
    std::memcpy(x3.data(), x3_h, 24 * sizeof(float));
    Tensor x3_cuda = x3.toDevice(Device::CUDA);

    Tensor s3 = x3_cuda.sum(1);
    float s3_h[8];
    cudaMemcpy(s3_h, s3.data(), 8 * sizeof(float), cudaMemcpyDeviceToHost);
    // expected: for each (b,v), sum over t: x3[b,0,v] + x3[b,1,v] + x3[b,2,v]
    float expected_s3[8];
    for (int b = 0; b < 2; ++b) {
        for (int v = 0; v < 4; ++v) {
            float val = 0.0f;
            for (int t = 0; t < 3; ++t) {
                val += x3_h[(b * 3 + t) * 4 + v];
            }
            expected_s3[b * 4 + v] = val;
        }
    }
    for (int i = 0; i < 8; ++i) {
        if (std::fabs(s3_h[i] - expected_s3[i]) > 1e-4f) {
            std::cout << "3D sum(1) mismatch at " << i << ": got " << s3_h[i]
                      << " expected " << expected_s3[i] << "\n";
            ok = false;
        }
    }

    if (ok) {
        std::cout << "All CUDA reduction checks passed.\n";
        return 0;
    }
    return 1;
}
