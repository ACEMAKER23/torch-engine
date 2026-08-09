#include <iostream>
#include <cmath>
#include <numeric>
#include "../tensor/tensor.h"
#include "../core/cuda_utils.h"

int main() {
    if (!cuda_available()) {
        std::cerr << "CUDA not available\n";
        return 1;
    }

    // Test 1: 2D softmax on last dim
    // x: [3, 4] row-major
    Tensor x({3, 4}, DType::Float32, Device::CUDA);
    float x_h[12] = {1.0f, 2.0f, 3.0f, 4.0f,
                     1.0f, 2.0f, 3.0f, 4.0f,
                     1.0f, 2.0f, 3.0f, 4.0f};
    cudaMemcpy(x.data(), x_h, 12 * sizeof(float), cudaMemcpyHostToDevice);

    x.setRequiresGrad(true);
    Tensor s = x.softmax(1);
    float s_h[12];
    cudaMemcpy(s_h, s.data(), 12 * sizeof(float), cudaMemcpyDeviceToHost);

    // Expected softmax for [1,2,3,4]
    float max_val = 4.0f;
    float exps[4] = {std::exp(1.0f - max_val), std::exp(2.0f - max_val),
                     std::exp(3.0f - max_val), std::exp(4.0f - max_val)};
    float sum = exps[0] + exps[1] + exps[2] + exps[3];
    bool ok = true;
    for (int i = 0; i < 12; ++i) {
        float expected = exps[i % 4] / sum;
        if (std::fabs(s_h[i] - expected) > 1e-5f) {
            std::cout << "softmax forward mismatch at " << i << ": got " << s_h[i]
                      << " expected " << expected << "\n";
            ok = false;
        }
        // Check rows sum to 1
        if ((i + 1) % 4 == 0) {
            float row_sum = s_h[i - 3] + s_h[i - 2] + s_h[i - 1] + s_h[i];
            if (std::fabs(row_sum - 1.0f) > 1e-5f) {
                std::cout << "row sum mismatch at row " << i / 4 << ": " << row_sum << "\n";
                ok = false;
            }
        }
    }

    // Test 2: 3D softmax on last dim
    Tensor x3({2, 3, 4}, DType::Float32, Device::CUDA);
    float x3_h[24];
    for (int i = 0; i < 24; ++i) x3_h[i] = static_cast<float>(i % 4);
    cudaMemcpy(x3.data(), x3_h, 24 * sizeof(float), cudaMemcpyHostToDevice);
    x3.setRequiresGrad(true);
    Tensor s3 = x3.softmax(2);
    float s3_h[24];
    cudaMemcpy(s3_h, s3.data(), 24 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 24; ++i) {
        float expected = exps[i % 4] / sum;
        if (std::fabs(s3_h[i] - expected) > 1e-5f) {
            std::cout << "3D softmax forward mismatch at " << i << ": got " << s3_h[i]
                      << " expected " << expected << "\n";
            ok = false;
        }
    }

    // Test 3: backward pass on 2D softmax
    // s.backward() propagates an all-ones gradient.  For softmax, dx_i = s_i * (1 - sum_j s_j),
    // and since each row of s sums to 1, the gradient should be all zeros.
    s.backward();
    float dx_h[12];
    cudaMemcpy(dx_h, x.grad()->data(), 12 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 12; ++i) {
        if (std::fabs(dx_h[i]) > 1e-5f) {
            std::cout << "softmax backward mismatch at " << i << ": got " << dx_h[i]
                      << " expected 0\n";
            ok = false;
        }
    }

    if (ok) {
        std::cout << "All CUDA softmax checks passed.\n";
        return 0;
    }
    return 1;
}
