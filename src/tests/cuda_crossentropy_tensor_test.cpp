#include <iostream>
#include <cmath>
#include <cstring>
#include "../loss/CrossEntropyLoss.h"
#include "../tensor/tensor.h"
#include "../core/cuda_utils.h"

int main() {
    if (!cuda_available()) {
        std::cerr << "CUDA not available\n";
        return 1;
    }

    const int B = 2;
    const int T = 3;
    const int V = 4;
    const int N = B * T;

    // Logits on CPU first, then upload.
    float logits_h[24] = {
        0.0f, 1.0f, 2.0f, 3.0f,
        1.0f, 2.0f, 3.0f, 4.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        2.0f, 1.0f, 0.0f, -1.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.5f, 0.5f, 0.5f, 0.5f
    };
    int64_t targets_h[6] = {0, 2, 1, 3, 2, 0};

    Tensor logits({B, T, V}, DType::Float32, Device::CPU);
    Tensor targets({B, T}, DType::Int64, Device::CPU);
    std::memcpy(logits.data(), logits_h, 24 * sizeof(float));
    std::memcpy(targets.data(), targets_h, 6 * sizeof(int64_t));

    Tensor logits_cuda  = logits.toDevice(Device::CUDA);
    Tensor targets_cuda = targets.toDevice(Device::CUDA);
    logits_cuda.setRequiresGrad(true);

    crossEntropyLoss celoss;
    Tensor loss = celoss.forward_batched(logits_cuda, targets_cuda);

    if (loss.device() != Device::CUDA) {
        std::cerr << "Expected loss to be on CUDA\n";
        return 1;
    }

    float loss_h;
    cudaMemcpy(&loss_h, loss.data(), sizeof(float), cudaMemcpyDeviceToHost);

    // Reference CPU computation.
    double total_loss = 0.0;
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const float* row = logits_h + (b * T + t) * V;
            int64_t target_class = targets_h[b * T + t];
            float max_logit = row[0];
            for (int v = 1; v < V; ++v) max_logit = std::max(max_logit, row[v]);
            float sum_exp = 0.0f;
            for (int v = 0; v < V; ++v) sum_exp += std::exp(row[v] - max_logit);
            float log_sum_exp = max_logit + std::log(sum_exp);
            total_loss += (-row[target_class] + log_sum_exp);
        }
    }
    float expected_loss = static_cast<float>(total_loss / N);

    bool ok = true;
    if (std::fabs(loss_h - expected_loss) > 1e-4f) {
        std::cout << "forward loss mismatch: got " << loss_h << " expected "
                  << expected_loss << "\n";
        ok = false;
    }

    // Backward: loss.backward() seeds gradient with ones, so upstream = 1.
    loss.backward();

    float grad_h[24];
    cudaMemcpy(grad_h, logits_cuda.grad()->data(), 24 * sizeof(float), cudaMemcpyDeviceToHost);

    float expected_grad[24];
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const float* row = logits_h + (b * T + t) * V;
            float* exp_grad = expected_grad + (b * T + t) * V;
            int64_t target_class = targets_h[b * T + t];
            float max_logit = row[0];
            for (int v = 1; v < V; ++v) max_logit = std::max(max_logit, row[v]);
            float sum_exp = 0.0f;
            for (int v = 0; v < V; ++v) sum_exp += std::exp(row[v] - max_logit);
            for (int v = 0; v < V; ++v) {
                float softmax = std::exp(row[v] - max_logit) / sum_exp;
                float one_hot = (v == target_class) ? 1.0f : 0.0f;
                exp_grad[v] = (softmax - one_hot) / static_cast<float>(N);
            }
        }
    }

    for (int i = 0; i < 24; ++i) {
        if (std::fabs(grad_h[i] - expected_grad[i]) > 1e-4f) {
            std::cout << "backward grad mismatch at " << i << ": got " << grad_h[i]
                      << " expected " << expected_grad[i] << "\n";
            ok = false;
        }
    }

    if (ok) {
        std::cout << "All CUDA CrossEntropy checks passed.\n";
        return 0;
    }
    return 1;
}
