#include <iostream>
#include <cmath>
#include <cstring>
#include "../nn/gelu.h"
#include "../tensor/tensor.h"
#include "../core/cuda_utils.h"

static float reference_gelu(float x) {
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x3)));
}

static float reference_gelu_grad(float x) {
    float c = 0.7978845608f;
    float k = 0.044715f;
    float x_cubed = x * x * x;
    float z = c * (x + k * x_cubed);
    float t = std::tanh(z);
    float sech_sq = 1.0f - t * t;
    float inner = c * (1.0f + 3.0f * k * x * x);
    return 0.5f * (1.0f + t) * (1.0f + x * sech_sq * inner);
}

int main() {
    if (!cuda_available()) {
        std::cerr << "CUDA not available\n";
        return 1;
    }

    const int size = 12;
    float x_h[12] = {
        0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f,
        -0.5f, 3.0f, -3.0f, 0.1f, -0.1f, 1.5f
    };

    Tensor x({size}, DType::Float32, Device::CPU);
    std::memcpy(x.data(), x_h, size * sizeof(float));
    Tensor x_cuda = x.toDevice(Device::CUDA);
    x_cuda.setRequiresGrad(true);

    GeLU gelu;
    Tensor y = gelu.forward(x_cuda);

    if (y.device() != Device::CUDA) {
        std::cerr << "Expected output to be on CUDA\n";
        return 1;
    }

    float y_h[12];
    cudaMemcpy(y_h, y.data(), size * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < size; ++i) {
        float expected = reference_gelu(x_h[i]);
        if (std::fabs(y_h[i] - expected) > 1e-4f) {
            std::cout << "forward mismatch at " << i << ": got " << y_h[i]
                      << " expected " << expected << "\n";
            ok = false;
        }
    }

    // Backward with all-ones upstream gradient.
    y.backward();

    if (!x_cuda.grad()) {
        std::cerr << "x_cuda.grad() is null after backward\n";
        return 1;
    }

    float dx_h[12];
    cudaMemcpy(dx_h, x_cuda.grad()->data(), size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        float expected = reference_gelu_grad(x_h[i]);  // upstream = 1
        if (std::fabs(dx_h[i] - expected) > 1e-4f) {
            std::cout << "backward mismatch at " << i << ": got " << dx_h[i]
                      << " expected " << expected << "\n";
            ok = false;
        }
    }

    if (ok) {
        std::cout << "All CUDA GELU checks passed.\n";
        return 0;
    }
    return 1;
}
