#include <iostream>
#include <cmath>
#include <cstring>
#include "../nn/layernorm.h"
#include "../tensor/tensor.h"
#include "../core/cuda_utils.h"

int main() {
    if (!cuda_available()) {
        std::cerr << "CUDA not available\n";
        return 1;
    }

    const int batch = 3;
    const int D = 4;
    LayerNorm ln(D, DType::Float32, 1e-5f);

    // Test 1: forward pass on GPU
    Tensor x({batch, D}, DType::Float32, Device::CUDA);
    float x_h[12] = {1.0f, 2.0f, 3.0f, 4.0f,
                     2.0f, 4.0f, 6.0f, 8.0f,
                     -1.0f, -1.0f, 1.0f, 1.0f};
    cudaMemcpy(x.data(), x_h, 12 * sizeof(float), cudaMemcpyHostToDevice);

    x.setRequiresGrad(true);
    Tensor y = ln.forward(x);
    float y_h[12];
    cudaMemcpy(y_h, y.data(), 12 * sizeof(float), cudaMemcpyDeviceToHost);

    // With gamma=1, beta=0, output is (x - mean) / sqrt(var + eps)
    bool ok = true;
    for (int i = 0; i < batch; ++i) {
        float mean = 0.0f;
        for (int j = 0; j < D; ++j) mean += x_h[i * D + j];
        mean /= D;
        float var = 0.0f;
        for (int j = 0; j < D; ++j) {
            float d = x_h[i * D + j] - mean;
            var += d * d;
        }
        var /= D;
        float inv_std = 1.0f / std::sqrt(var + 1e-5f);
        for (int j = 0; j < D; ++j) {
            float expected = (x_h[i * D + j] - mean) * inv_std;
            if (std::fabs(y_h[i * D + j] - expected) > 1e-4f) {
                std::cout << "forward mismatch at (" << i << "," << j << "): got "
                          << y_h[i * D + j] << " expected " << expected << "\n";
                ok = false;
            }
        }
    }

    // Test 2: backward pass on GPU (use all-ones seed gradient)
    y.backward();

    float dx_h[12];
    cudaMemcpy(dx_h, x.grad()->data(), 12 * sizeof(float), cudaMemcpyDeviceToHost);

    float dg_h[4];
    float db_h[4];
    // weight_/bias_ and their grads live on CPU, so copy with std::memcpy.
    std::memcpy(dg_h, ln.parameters()[0].grad()->data(), 4 * sizeof(float));
    std::memcpy(db_h, ln.parameters()[1].grad()->data(), 4 * sizeof(float));

    // For all-ones dy, d_beta[j] = batch
    for (int j = 0; j < D; ++j) {
        if (std::fabs(db_h[j] - static_cast<float>(batch)) > 1e-4f) {
            std::cout << "d_beta mismatch at " << j << ": got " << db_h[j]
                      << " expected " << batch << "\n";
            ok = false;
        }
    }

    // d_gamma[j] = sum over batch of x_hat[i,j]
    for (int j = 0; j < D; ++j) {
        float expected_dg = 0.0f;
        for (int i = 0; i < batch; ++i) expected_dg += y_h[i * D + j];
        if (std::fabs(dg_h[j] - expected_dg) > 1e-4f) {
            std::cout << "d_gamma mismatch at " << j << ": got " << dg_h[j]
                      << " expected " << expected_dg << "\n";
            ok = false;
        }
    }

    // Check d_x rows sum to 0 (consistency check for all-ones dy)
    for (int i = 0; i < batch; ++i) {
        float row_sum = 0.0f;
        for (int j = 0; j < D; ++j) row_sum += dx_h[i * D + j];
        if (std::fabs(row_sum) > 1e-3f) {
            std::cout << "d_x row " << i << " sum not zero: " << row_sum << "\n";
            ok = false;
        }
    }

    if (ok) {
        std::cout << "All CUDA LayerNorm checks passed.\n";
        return 0;
    }
    return 1;
}
