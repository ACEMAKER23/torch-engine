#include <iostream>
#include <cmath>
#include <cstring>
#include "../nn/embedding.h"
#include "../tensor/tensor.h"
#include "../core/cuda_utils.h"

int main() {
    if (!cuda_available()) {
        std::cerr << "CUDA not available\n";
        return 1;
    }

    const int num_embeddings = 5;
    const int D = 4;
    Embedding emb(num_embeddings, D, DType::Float32);

    // Initialize weight matrix to known values on CPU.
    float weight_h[20] = {
        0.0f,  1.0f,  2.0f,  3.0f,
        4.0f,  5.0f,  6.0f,  7.0f,
        8.0f,  9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, 19.0f
    };
    std::memcpy(emb.parameters()[0].data(), weight_h, num_embeddings * D * sizeof(float));

    // indices: [2, 0, 3, 2] -> two occurrences of index 2.
    const int total_indices = 4;
    int64_t indices_h[4] = {2, 0, 3, 2};
    Tensor indices({total_indices}, DType::Int64, Device::CPU);
    std::memcpy(indices.data(), indices_h, total_indices * sizeof(int64_t));
    Tensor indices_cuda = indices.toDevice(Device::CUDA);

    Tensor y = emb.forward(indices_cuda);
    if (y.device() != Device::CUDA) {
        std::cerr << "Expected output to be on CUDA\n";
        return 1;
    }

    // Verify forward output.
    float y_h[16];
    cudaMemcpy(y_h, y.data(), total_indices * D * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < total_indices; ++i) {
        int64_t idx = indices_h[i];
        for (int j = 0; j < D; ++j) {
            float expected = weight_h[idx * D + j];
            if (std::fabs(y_h[i * D + j] - expected) > 1e-5f) {
                std::cout << "forward mismatch at (" << i << "," << j << "): got "
                          << y_h[i * D + j] << " expected " << expected << "\n";
                ok = false;
            }
        }
    }

    // Backward with all-ones seed gradient.
    y.backward();

    float dw_h[20];
    std::memcpy(dw_h, emb.parameters()[0].grad()->data(), num_embeddings * D * sizeof(float));

    int64_t counts[5] = {0};
    for (int i = 0; i < total_indices; ++i) counts[indices_h[i]]++;

    for (int idx = 0; idx < num_embeddings; ++idx) {
        for (int j = 0; j < D; ++j) {
            float expected = static_cast<float>(counts[idx]) * 1.0f;
            float got = dw_h[idx * D + j];
            if (std::fabs(got - expected) > 1e-5f) {
                std::cout << "d_weight mismatch at (" << idx << "," << j << "): got "
                          << got << " expected " << expected << "\n";
                ok = false;
            }
        }
    }

    if (ok) {
        std::cout << "All CUDA Embedding checks passed.\n";
        return 0;
    }
    return 1;
}
