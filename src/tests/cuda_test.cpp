#include <gtest/gtest.h>
#include "../tensor/tensor.h"
#include "../core/dtype_utils.h"
#include "../core/cuda_utils.h"
#include "../cuda/elementwise_cuda.h"

#include <cublas_v2.h>
#include <random>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <string>
#include <vector>

TEST(CUDATest, ToDeviceRoundTrip) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    Tensor a({2, 3}, DType::Float32, Device::CPU);
    for (int i = 0; i < 6; ++i) {
        a.at<float>(i) = static_cast<float>(i);
    }

    Tensor gpu = a.toDevice(Device::CUDA);
    EXPECT_EQ(gpu.device(), Device::CUDA);
    EXPECT_EQ(gpu.shape(), a.shape());

    Tensor back = gpu.toDevice(Device::CPU);
    EXPECT_EQ(back.device(), Device::CPU);

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(back.at<float>(i), a.at<float>(i));
    }
}

TEST(CUDATest, ConstructGPU) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    Tensor a({2, 3}, DType::Float32, Device::CUDA);
    EXPECT_EQ(a.device(), Device::CUDA);
    EXPECT_EQ(a.numel(), 6);
}

TEST(CUDATest, SetDevice) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    cuda_set_device(0);
    EXPECT_EQ(cuda_get_device(), 0);
}

TEST(CUDATest, ElementwiseAdd) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    Tensor a({2, 3}, DType::Float32, Device::CPU);
    Tensor b({2, 3}, DType::Float32, Device::CPU);
    for (int i = 0; i < 6; ++i) {
        a.at<float>(i) = static_cast<float>(i);
        b.at<float>(i) = static_cast<float>(i * 2);
    }

    Tensor a_gpu = a.toDevice(Device::CUDA);
    Tensor b_gpu = b.toDevice(Device::CUDA);
    Tensor c_gpu = a_gpu + b_gpu;
    Tensor c = c_gpu.toDevice(Device::CPU);

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(c.at<float>(i), a.at<float>(i) + b.at<float>(i));
    }
}

TEST(CUDATest, ElementwiseSub) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    Tensor a({2, 3}, DType::Float32, Device::CPU);
    Tensor b({2, 3}, DType::Float32, Device::CPU);
    for (int i = 0; i < 6; ++i) {
        a.at<float>(i) = static_cast<float>(i);
        b.at<float>(i) = static_cast<float>(i * 2);
    }

    Tensor a_gpu = a.toDevice(Device::CUDA);
    Tensor b_gpu = b.toDevice(Device::CUDA);
    Tensor c_gpu = a_gpu - b_gpu;
    Tensor c = c_gpu.toDevice(Device::CPU);

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(c.at<float>(i), a.at<float>(i) - b.at<float>(i));
    }
}

TEST(CUDATest, ElementwiseMul) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    Tensor a({2, 3}, DType::Float32, Device::CPU);
    Tensor b({2, 3}, DType::Float32, Device::CPU);
    for (int i = 0; i < 6; ++i) {
        a.at<float>(i) = static_cast<float>(i);
        b.at<float>(i) = static_cast<float>(i * 2);
    }

    Tensor a_gpu = a.toDevice(Device::CUDA);
    Tensor b_gpu = b.toDevice(Device::CUDA);
    Tensor c_gpu = a_gpu * b_gpu;
    Tensor c = c_gpu.toDevice(Device::CPU);

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(c.at<float>(i), a.at<float>(i) * b.at<float>(i));
    }
}

TEST(CUDATest, ElementwiseDiv) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    Tensor a({2, 3}, DType::Float32, Device::CPU);
    Tensor b({2, 3}, DType::Float32, Device::CPU);
    for (int i = 0; i < 6; ++i) {
        a.at<float>(i) = static_cast<float>(i * 3);
        b.at<float>(i) = static_cast<float>(i + 1);
    }

    Tensor a_gpu = a.toDevice(Device::CUDA);
    Tensor b_gpu = b.toDevice(Device::CUDA);
    Tensor c_gpu = a_gpu / b_gpu;
    Tensor c = c_gpu.toDevice(Device::CPU);

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(c.at<float>(i), a.at<float>(i) / b.at<float>(i));
    }
}

TEST(CUDATest, ElementwiseAddFloat16) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    Tensor a({2, 3}, DType::Float16, Device::CPU);
    Tensor b({2, 3}, DType::Float16, Device::CPU);
    for (int i = 0; i < 6; ++i) {
        a.at<float16_t>(i) = float16_t::from_float32(static_cast<float>(i));
        b.at<float16_t>(i) = float16_t::from_float32(static_cast<float>(i * 2));
    }

    Tensor a_gpu = a.toDevice(Device::CUDA);
    Tensor b_gpu = b.toDevice(Device::CUDA);
    Tensor c_gpu = a_gpu + b_gpu;
    Tensor c = c_gpu.toDevice(Device::CPU);

    for (int i = 0; i < 6; ++i) {
        float expected = a.at<float16_t>(i).to_float32() + b.at<float16_t>(i).to_float32();
        EXPECT_NEAR(c.at<float16_t>(i).to_float32(), expected, 1e-3f);
    }
}

TEST(CUDATest, ElementwiseAddBFloat16) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    Tensor a({2, 3}, DType::BFloat16, Device::CPU);
    Tensor b({2, 3}, DType::BFloat16, Device::CPU);
    for (int i = 0; i < 6; ++i) {
        a.at<bfloat16_t>(i) = bfloat16_t::from_float32(static_cast<float>(i));
        b.at<bfloat16_t>(i) = bfloat16_t::from_float32(static_cast<float>(i * 2));
    }

    Tensor a_gpu = a.toDevice(Device::CUDA);
    Tensor b_gpu = b.toDevice(Device::CUDA);
    Tensor c_gpu = a_gpu + b_gpu;
    Tensor c = c_gpu.toDevice(Device::CPU);

    for (int i = 0; i < 6; ++i) {
        float expected = a.at<bfloat16_t>(i).to_float32() + b.at<bfloat16_t>(i).to_float32();
        EXPECT_NEAR(c.at<bfloat16_t>(i).to_float32(), expected, 1e-3f);
    }
}

namespace {

std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        s += std::to_string(shape[i]);
        if (i + 1 < shape.size()) s += ",";
    }
    s += "]";
    return s;
}

template <typename T, DType DT, typename Op, typename RefOp>
void test_elementwise_mixed_precision(const char* op_name, Op op, RefOp ref_op) {
    const std::vector<std::vector<int64_t>> shapes = {
        {2, 3},
        {16, 16},
        {128, 256},
        {512, 512}
    };

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (const auto& shape : shapes) {
        SCOPED_TRACE(std::string(op_name) + " " + shape_to_string(shape));
        Tensor a(shape, DT, Device::CPU);
        Tensor b(shape, DT, Device::CPU);

        for (size_t i = 0; i < a.numel(); ++i) {
            float av = dist(gen);
            float bv = dist(gen);
            if (std::string(op_name) == "div" && std::fabs(bv) < 0.25f) {
                bv += (bv >= 0.0f ? 0.5f : -0.5f);
            }
            a.at<T>(i) = T::from_float32(av);
            b.at<T>(i) = T::from_float32(bv);
        }

        Tensor a_gpu = a.toDevice(Device::CUDA);
        Tensor b_gpu = b.toDevice(Device::CUDA);
        Tensor c_gpu = op(a_gpu, b_gpu);
        Tensor c = c_gpu.toDevice(Device::CPU);

        for (size_t i = 0; i < a.numel(); ++i) {
            float expected = ref_op(a.at<T>(i).to_float32(), b.at<T>(i).to_float32());
            float actual = c.at<T>(i).to_float32();
            float abs_err = std::fabs(actual - expected);
            float tol = 1e-2f * std::max(1.0f, std::fabs(expected));
            EXPECT_LE(abs_err, tol) << op_name << " mismatch at flat index " << i;
        }
    }
}

}

TEST(CUDATest, MixedPrecisionElementwiseAddFloat16) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA not available"; }
    test_elementwise_mixed_precision<float16_t, DType::Float16>(
        "add",
        [](const Tensor& x, const Tensor& y) { return x + y; },
        [](float x, float y) { return x + y; });
}

TEST(CUDATest, MixedPrecisionElementwiseSubFloat16) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA not available"; }
    test_elementwise_mixed_precision<float16_t, DType::Float16>(
        "sub",
        [](const Tensor& x, const Tensor& y) { return x - y; },
        [](float x, float y) { return x - y; });
}

TEST(CUDATest, MixedPrecisionElementwiseMulFloat16) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA not available"; }
    test_elementwise_mixed_precision<float16_t, DType::Float16>(
        "mul",
        [](const Tensor& x, const Tensor& y) { return x * y; },
        [](float x, float y) { return x * y; });
}

TEST(CUDATest, MixedPrecisionElementwiseDivFloat16) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA not available"; }
    test_elementwise_mixed_precision<float16_t, DType::Float16>(
        "div",
        [](const Tensor& x, const Tensor& y) { return x / y; },
        [](float x, float y) { return x / y; });
}

TEST(CUDATest, MixedPrecisionElementwiseAddBFloat16) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA not available"; }
    test_elementwise_mixed_precision<bfloat16_t, DType::BFloat16>(
        "add",
        [](const Tensor& x, const Tensor& y) { return x + y; },
        [](float x, float y) { return x + y; });
}

TEST(CUDATest, MixedPrecisionElementwiseSubBFloat16) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA not available"; }
    test_elementwise_mixed_precision<bfloat16_t, DType::BFloat16>(
        "sub",
        [](const Tensor& x, const Tensor& y) { return x - y; },
        [](float x, float y) { return x - y; });
}

TEST(CUDATest, MixedPrecisionElementwiseMulBFloat16) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA not available"; }
    test_elementwise_mixed_precision<bfloat16_t, DType::BFloat16>(
        "mul",
        [](const Tensor& x, const Tensor& y) { return x * y; },
        [](float x, float y) { return x * y; });
}

TEST(CUDATest, MixedPrecisionElementwiseDivBFloat16) {
    if (!cuda_available()) { GTEST_SKIP() << "CUDA not available"; }
    test_elementwise_mixed_precision<bfloat16_t, DType::BFloat16>(
        "div",
        [](const Tensor& x, const Tensor& y) { return x / y; },
        [](float x, float y) { return x / y; });
}

static void cublas_check(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": cublas error " + std::to_string(status));
    }
}

static void cublas_matmul_ref(const float* A, const float* B, float* C,
                              int M, int K, int N) {
    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle), "cublasCreate");

    float alpha = 1.0f;
    float beta  = 0.0f;

    cublas_check(cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, N,
                A, K,
                &beta,
                C, N), "cublasSgemm");

    cublas_check(cublasDestroy(handle), "cublasDestroy");
}

using matmul_fn_t = void(*)(const float*, const float*, float*, int, int, int);

static void test_matmul_against_cublas(matmul_fn_t kernel, const char* name,
                                       int M, int K, int N) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_ref = nullptr;

    // CUDA misaligned-address errors are sticky per-context; if this test fails,
    // reset the context so the next test starts clean.
    try {
        const size_t A_elems = static_cast<size_t>(M) * K;
        const size_t B_elems = static_cast<size_t>(K) * N;
        const size_t C_elems = static_cast<size_t>(M) * N;

        std::vector<float> h_A(A_elems), h_B(B_elems), h_C(C_elems), h_ref(C_elems);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& x : h_A) x = dist(gen);
        for (auto& x : h_B) x = dist(gen);

        cuda_check_error(cudaMalloc(&d_A, A_elems * sizeof(float)), "cudaMalloc d_A");
        cuda_check_error(cudaMalloc(&d_B, B_elems * sizeof(float)), "cudaMalloc d_B");
        cuda_check_error(cudaMalloc(&d_C, C_elems * sizeof(float)), "cudaMalloc d_C");
        cuda_check_error(cudaMalloc(&d_ref, C_elems * sizeof(float)), "cudaMalloc d_ref");

        cuda_check_error(cudaMemcpy(d_A, h_A.data(), A_elems * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy d_A");
        cuda_check_error(cudaMemcpy(d_B, h_B.data(), B_elems * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy d_B");

        kernel(d_A, d_B, d_C, M, K, N);
        cuda_check_error(cudaGetLastError(), "kernel launch");
        cuda_check_error(cudaDeviceSynchronize(), "kernel sync");

        cublas_matmul_ref(d_A, d_B, d_ref, M, K, N);
        cuda_check_error(cudaDeviceSynchronize(), "cublas sync");

        cuda_check_error(cudaMemcpy(h_C.data(), d_C, C_elems * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy h_C");
        cuda_check_error(cudaMemcpy(h_ref.data(), d_ref, C_elems * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy h_ref");

        float max_err = 0.0f;
        for (size_t i = 0; i < C_elems; ++i) {
            max_err = std::max(max_err, std::fabs(h_C[i] - h_ref[i]));
        }

        float tol = 1e-4f * std::max(K, 1);
        EXPECT_LT(max_err, tol) << name << " (" << M << "x" << K << "x" << N << ") max error vs cuBLAS: " << max_err;

        cuda_check_error(cudaFree(d_A), "cudaFree d_A");
        cuda_check_error(cudaFree(d_B), "cudaFree d_B");
        cuda_check_error(cudaFree(d_C), "cudaFree d_C");
        cuda_check_error(cudaFree(d_ref), "cudaFree d_ref");
    } catch (const std::runtime_error&) {
        cudaGetLastError();  // clear sticky CUDA error without destroying the context
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_ref);
        throw;
    }
}

static void run_matmul_accuracy_suite(matmul_fn_t kernel, const char* name,
                                      const std::vector<std::tuple<int, int, int>>& shapes) {
    for (const auto& shape : shapes) {
        int M, K, N;
        std::tie(M, K, N) = shape;
        test_matmul_against_cublas(kernel, name, M, K, N);
    }
}

TEST(CUDATest, MatmulNaiveAccuracy) {
    run_matmul_accuracy_suite(cuda_matmul, "naive", {
        std::make_tuple(1, 1, 1),
        std::make_tuple(3, 5, 7),
        std::make_tuple(15, 17, 19),
        std::make_tuple(16, 16, 16),
        std::make_tuple(17, 31, 13),
        std::make_tuple(32, 32, 32),
        std::make_tuple(33, 65, 17),
        std::make_tuple(64, 64, 64),
        std::make_tuple(65, 33, 129)
    });
}

TEST(CUDATest, MatmulSharedMemoryAccuracy) {
    run_matmul_accuracy_suite(cuda_matmul_shared_memory, "shared_memory", {
        std::make_tuple(1, 16, 16),
        std::make_tuple(3, 5, 7),
        std::make_tuple(15, 17, 19),
        std::make_tuple(16, 16, 16),
        std::make_tuple(17, 31, 13),
        std::make_tuple(32, 32, 32),
        std::make_tuple(33, 65, 17),
        std::make_tuple(64, 64, 64),
        std::make_tuple(65, 33, 129),
        std::make_tuple(128, 128, 128)
    });
}

TEST(CUDATest, MatmulRegisterBlockingAccuracy) {
    run_matmul_accuracy_suite(cuda_matmul_register_blocking, "register_blocking", {
        std::make_tuple(33, 65, 17),
        std::make_tuple(127, 255, 129),
        std::make_tuple(128, 128, 128),
        std::make_tuple(129, 65, 17),
        std::make_tuple(256, 256, 256),
        std::make_tuple(128, 256, 512),
        std::make_tuple(256, 128, 512)
    });
}

TEST(CUDATest, MatmulVectorizedInputAccuracy) {
    run_matmul_accuracy_suite(cuda_matmul_vectorized_input, "vectorized_input", {
        std::make_tuple(128, 128, 128),
        std::make_tuple(129, 128, 128),
        std::make_tuple(128, 128, 132),
        std::make_tuple(128, 136, 128),
        std::make_tuple(256, 256, 256),
        std::make_tuple(128, 256, 512),
        std::make_tuple(256, 128, 512)
    });
}

TEST(CUDATest, MatmulWarpTilingAccuracy) {
    run_matmul_accuracy_suite(cuda_matmul_warp_tiling, "warp_tiling", {
        std::make_tuple(33, 128, 128),
        std::make_tuple(129, 128, 128),
        std::make_tuple(128, 128, 132),
        std::make_tuple(128, 136, 128),
        std::make_tuple(129, 256, 512),
        std::make_tuple(256, 128, 512),
        std::make_tuple(128, 256, 512)
    });
}

TEST(CUDATest, MatmulDoubleBufferedAccuracy) {
    run_matmul_accuracy_suite(cuda_matmul_double_buffered, "double_buffered", {
        std::make_tuple(33, 128, 128),
        std::make_tuple(129, 128, 128),
        std::make_tuple(128, 128, 132),
        std::make_tuple(128, 136, 128),
        std::make_tuple(129, 256, 512),
        std::make_tuple(256, 128, 512),
        std::make_tuple(128, 256, 512)
    });
}

TEST(CUDATest, MatmulDoubleBufferedCpAsyncAccuracy) {
    run_matmul_accuracy_suite(cuda_matmul_double_buffered_cpasync, "double_buffered_cpasync", {
        std::make_tuple(33, 128, 128),
        std::make_tuple(129, 128, 128),
        std::make_tuple(128, 128, 132),
        std::make_tuple(128, 136, 128),
        std::make_tuple(129, 256, 512),
        std::make_tuple(256, 128, 512),
        std::make_tuple(128, 256, 512)
    });
}

TEST(CUDATest, MatmulDoubleBufferedSwizzledAccuracy) {
    run_matmul_accuracy_suite(cuda_matmul_double_buffered_swizzled, "double_buffered_swizzled", {
        std::make_tuple(33, 128, 128),
        std::make_tuple(129, 128, 128),
        std::make_tuple(128, 128, 132),
        std::make_tuple(128, 136, 128),
        std::make_tuple(129, 256, 512),
        std::make_tuple(256, 128, 512),
        std::make_tuple(128, 256, 512)
    });
}

TEST(CUDATest, MatmulVectorStorageAccuracy) {
    run_matmul_accuracy_suite(cuda_matmul_vector_storage, "vector_storage", {
        std::make_tuple(33, 128, 128),
        std::make_tuple(129, 128, 128),
        std::make_tuple(128, 128, 132),
        std::make_tuple(128, 136, 128),
        std::make_tuple(129, 256, 512),
        std::make_tuple(256, 128, 512),
        std::make_tuple(128, 256, 512)
    });
}

TEST(CUDATest, Matmul3StageCpAsyncAccuracy) {
    run_matmul_accuracy_suite(cuda_matmul_3stage_cpasync, "3stage_cpasync", {
        std::make_tuple(33, 128, 128),
        std::make_tuple(129, 128, 128),
        std::make_tuple(128, 128, 132),
        std::make_tuple(128, 136, 128),
        std::make_tuple(129, 256, 512),
        std::make_tuple(256, 128, 512),
        std::make_tuple(128, 256, 512)
    });
}

using half_matmul_fn_t = void(*)(const half*, const half*, half*, int, int, int);

static void test_half_matmul_against_cublas(half_matmul_fn_t kernel, const char* name,
                                            int M, int K, int N) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    half *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    float *d_ref = nullptr;

    try {
        const size_t A_elems = static_cast<size_t>(M) * K;
        const size_t B_elems = static_cast<size_t>(K) * N;
        const size_t C_elems = static_cast<size_t>(M) * N;

        std::vector<float> h_A(A_elems), h_B(B_elems), h_C(C_elems), h_ref(C_elems);
        std::vector<half> h_A_half(A_elems), h_B_half(B_elems), h_C_half(C_elems);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        for (size_t i = 0; i < A_elems; ++i) {
            h_A[i] = dist(gen);
            h_A_half[i] = __float2half(h_A[i]);
        }
        for (size_t i = 0; i < B_elems; ++i) {
            h_B[i] = dist(gen);
            h_B_half[i] = __float2half(h_B[i]);
        }

        cuda_check_error(cudaMalloc(&d_A, A_elems * sizeof(half)), "cudaMalloc d_A");
        cuda_check_error(cudaMalloc(&d_B, B_elems * sizeof(half)), "cudaMalloc d_B");
        cuda_check_error(cudaMalloc(&d_C, C_elems * sizeof(half)), "cudaMalloc d_C");
        cuda_check_error(cudaMalloc(&d_ref, C_elems * sizeof(float)), "cudaMalloc d_ref");

        cuda_check_error(cudaMemcpy(d_A, h_A_half.data(), A_elems * sizeof(half), cudaMemcpyHostToDevice), "cudaMemcpy d_A");
        cuda_check_error(cudaMemcpy(d_B, h_B_half.data(), B_elems * sizeof(half), cudaMemcpyHostToDevice), "cudaMemcpy d_B");

        kernel(d_A, d_B, d_C, M, K, N);
        cuda_check_error(cudaGetLastError(), "kernel launch");
        cuda_check_error(cudaDeviceSynchronize(), "kernel sync");

        cublasHandle_t handle;
        cublas_check(cublasCreate(&handle), "cublasCreate");
        float alpha = 1.0f;
        float beta  = 0.0f;
        cublas_check(cublasGemmEx(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, CUDA_R_16F, N,
                    d_A, CUDA_R_16F, K,
                    &beta,
                    d_ref, CUDA_R_32F, N,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT), "cublasGemmEx");
        cublas_check(cublasDestroy(handle), "cublasDestroy");
        cuda_check_error(cudaDeviceSynchronize(), "cublas sync");

        cuda_check_error(cudaMemcpy(h_C_half.data(), d_C, C_elems * sizeof(half), cudaMemcpyDeviceToHost), "cudaMemcpy d_C");
        cuda_check_error(cudaMemcpy(h_ref.data(), d_ref, C_elems * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy h_ref");

        for (size_t i = 0; i < C_elems; ++i) {
            h_C[i] = __half2float(h_C_half[i]);
        }

        float max_err = 0.0f;
        for (size_t i = 0; i < C_elems; ++i) {
            max_err = std::max(max_err, std::fabs(h_C[i] - h_ref[i]));
        }

        float tol = 1e-1f * std::max(K, 1);
        EXPECT_LT(max_err, tol) << name << " (" << M << "x" << K << "x" << N << ") max error vs cuBLAS: " << max_err;

        cuda_check_error(cudaFree(d_A), "cudaFree d_A");
        cuda_check_error(cudaFree(d_B), "cudaFree d_B");
        cuda_check_error(cudaFree(d_C), "cudaFree d_C");
        cuda_check_error(cudaFree(d_ref), "cudaFree d_ref");
    } catch (const std::runtime_error&) {
        cudaGetLastError();
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_ref);
        throw;
    }
}

static void run_half_matmul_accuracy_suite(half_matmul_fn_t kernel, const char* name,
                                           const std::vector<std::tuple<int, int, int>>& shapes) {
    for (const auto& shape : shapes) {
        int M, K, N;
        std::tie(M, K, N) = shape;
        test_half_matmul_against_cublas(kernel, name, M, K, N);
    }
}

using fp16_matmul_fn_t = void(*)(const half*, const half*, float*, int, int, int);

static void test_fp16_matmul_against_cublas(fp16_matmul_fn_t kernel, const char* name,
                                            int M, int K, int N) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    // SM80+ is required for tensor-core mma.sync and cp.async used by these kernels.
    int major = 0;
    cuda_check_error(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0), "cc major");
    if (major < 8) {
        GTEST_SKIP() << name << " requires SM80+ (Ampere or newer)";
    }

    half *d_A = nullptr, *d_B = nullptr;
    float *d_C = nullptr, *d_ref = nullptr;

    try {
        const size_t A_elems = static_cast<size_t>(M) * K;
        const size_t B_elems = static_cast<size_t>(K) * N;
        const size_t C_elems = static_cast<size_t>(M) * N;

        std::vector<float> h_A(A_elems), h_B(B_elems), h_C(C_elems), h_ref(C_elems);
        std::vector<float16_t> h_A_fp16(A_elems), h_B_fp16(B_elems);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        for (size_t i = 0; i < A_elems; ++i) {
            h_A[i] = dist(gen);
            h_A_fp16[i] = float16_t::from_float32(h_A[i]);
        }
        for (size_t i = 0; i < B_elems; ++i) {
            h_B[i] = dist(gen);
            h_B_fp16[i] = float16_t::from_float32(h_B[i]);
        }

        cuda_check_error(cudaMalloc(&d_A, A_elems * sizeof(half)), "cudaMalloc d_A");
        cuda_check_error(cudaMalloc(&d_B, B_elems * sizeof(half)), "cudaMalloc d_B");
        cuda_check_error(cudaMalloc(&d_C, C_elems * sizeof(float)), "cudaMalloc d_C");
        cuda_check_error(cudaMalloc(&d_ref, C_elems * sizeof(float)), "cudaMalloc d_ref");

        cuda_check_error(cudaMemcpy(d_A, h_A_fp16.data(), A_elems * sizeof(half), cudaMemcpyHostToDevice), "cudaMemcpy d_A");
        cuda_check_error(cudaMemcpy(d_B, h_B_fp16.data(), B_elems * sizeof(half), cudaMemcpyHostToDevice), "cudaMemcpy d_B");

        kernel(d_A, d_B, d_C, M, K, N);
        cuda_check_error(cudaGetLastError(), "kernel launch");
        cuda_check_error(cudaDeviceSynchronize(), "kernel sync");

        cublasHandle_t handle;
        cublas_check(cublasCreate(&handle), "cublasCreate");
        float alpha = 1.0f;
        float beta  = 0.0f;
        cublas_check(cublasGemmEx(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, CUDA_R_16F, N,
                    d_A, CUDA_R_16F, K,
                    &beta,
                    d_ref, CUDA_R_32F, N,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT), "cublasGemmEx");
        cublas_check(cublasDestroy(handle), "cublasDestroy");
        cuda_check_error(cudaDeviceSynchronize(), "cublas sync");

        cuda_check_error(cudaMemcpy(h_C.data(), d_C, C_elems * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy h_C");
        cuda_check_error(cudaMemcpy(h_ref.data(), d_ref, C_elems * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy h_ref");

        float max_err = 0.0f;
        for (size_t i = 0; i < C_elems; ++i) {
            max_err = std::max(max_err, std::fabs(h_C[i] - h_ref[i]));
        }

        float tol = 1e-2f * std::max(K, 1);
        EXPECT_LT(max_err, tol) << name << " (" << M << "x" << K << "x" << N << ") max error vs cuBLAS: " << max_err;

        cuda_check_error(cudaFree(d_A), "cudaFree d_A");
        cuda_check_error(cudaFree(d_B), "cudaFree d_B");
        cuda_check_error(cudaFree(d_C), "cudaFree d_C");
        cuda_check_error(cudaFree(d_ref), "cudaFree d_ref");
    } catch (const std::runtime_error&) {
        cudaGetLastError();
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_ref);
        throw;
    }
}

static void run_fp16_matmul_accuracy_suite(fp16_matmul_fn_t kernel, const char* name,
                                             const std::vector<std::tuple<int, int, int>>& shapes) {
    for (const auto& shape : shapes) {
        int M, K, N;
        std::tie(M, K, N) = shape;
        test_fp16_matmul_against_cublas(kernel, name, M, K, N);
    }
}

TEST(CUDATest, MatmulTensorCoreAccuracy) {
    run_fp16_matmul_accuracy_suite(launch_matmul_tensor_core, "tensor_core", {
        std::make_tuple(16, 16, 16),
        std::make_tuple(32, 32, 32),
        std::make_tuple(128, 128, 128),
        std::make_tuple(129, 65, 17),
        std::make_tuple(127, 255, 129),
        std::make_tuple(257, 129, 511),
        std::make_tuple(256, 256, 256),
        std::make_tuple(512, 512, 512),
        std::make_tuple(128, 256, 512),
        std::make_tuple(256, 128, 512)
    });
}

TEST(CUDATest, MatmulAmpereAccuracy) {
    run_fp16_matmul_accuracy_suite(launch_matmul_ampere, "ampere", {
        std::make_tuple(16, 16, 16),
        std::make_tuple(32, 32, 32),
        std::make_tuple(128, 128, 128),
        std::make_tuple(129, 65, 17),
        std::make_tuple(127, 255, 129),
        std::make_tuple(257, 129, 511),
        std::make_tuple(256, 256, 256),
        std::make_tuple(512, 512, 512),
        std::make_tuple(128, 256, 512),
        std::make_tuple(256, 128, 512)
    });
}

TEST(CUDATest, MatmulNaiveHalfAccuracy) {
    run_half_matmul_accuracy_suite(cuda_matmul, "naive_half", {
        std::make_tuple(1, 1, 1),
        std::make_tuple(3, 5, 7),
        std::make_tuple(15, 17, 19),
        std::make_tuple(16, 16, 16),
        std::make_tuple(17, 31, 13),
        std::make_tuple(32, 32, 32),
        std::make_tuple(33, 65, 17),
        std::make_tuple(65, 33, 129)
    });
}

TEST(CUDATest, MatmulSharedMemoryHalfAccuracy) {
    run_half_matmul_accuracy_suite(cuda_matmul_shared_memory, "shared_memory_half", {
        std::make_tuple(1, 16, 16),
        std::make_tuple(3, 5, 7),
        std::make_tuple(15, 17, 19),
        std::make_tuple(16, 16, 16),
        std::make_tuple(17, 31, 13),
        std::make_tuple(32, 32, 32),
        std::make_tuple(33, 65, 17),
        std::make_tuple(65, 33, 129)
    });
}

TEST(CUDATest, MatmulRegisterBlockingHalfAccuracy) {
    run_half_matmul_accuracy_suite(cuda_matmul_register_blocking, "register_blocking_half", {
        std::make_tuple(33, 65, 17),
        std::make_tuple(127, 255, 129),
        std::make_tuple(128, 128, 128),
        std::make_tuple(129, 65, 17)
    });
}

TEST(CUDATest, MatmulVectorizedInputHalfAccuracy) {
    run_half_matmul_accuracy_suite(cuda_matmul_vectorized_input, "vectorized_input_half", {
        std::make_tuple(128, 128, 128),
        std::make_tuple(129, 128, 128),
        std::make_tuple(128, 128, 136),
        std::make_tuple(128, 136, 128),
        std::make_tuple(256, 256, 256),
        std::make_tuple(128, 256, 512),
        std::make_tuple(256, 128, 512)
    });
}

TEST(CUDATest, MatmulWarpTilingHalfAccuracy) {
    run_half_matmul_accuracy_suite(cuda_matmul_warp_tiling, "warp_tiling_half", {
        std::make_tuple(33, 128, 128),
        std::make_tuple(129, 128, 128),
        std::make_tuple(128, 128, 136),
        std::make_tuple(128, 136, 128),
        std::make_tuple(129, 256, 512),
        std::make_tuple(256, 128, 512),
        std::make_tuple(128, 256, 512)
    });
}

TEST(CUDATest, MatmulDoubleBufferedHalfAccuracy) {
    run_half_matmul_accuracy_suite(cuda_matmul_double_buffered, "double_buffered_half", {
        std::make_tuple(33, 128, 128),
        std::make_tuple(129, 128, 128),
        std::make_tuple(128, 128, 136),
        std::make_tuple(128, 136, 128),
        std::make_tuple(129, 256, 512),
        std::make_tuple(256, 128, 512),
        std::make_tuple(128, 256, 512)
    });
}

TEST(CUDATest, MatmulDoubleBufferedCpAsyncHalfAccuracy) {
    run_half_matmul_accuracy_suite(cuda_matmul_double_buffered_cpasync, "double_buffered_cpasync_half", {
        std::make_tuple(33, 128, 128),
        std::make_tuple(129, 128, 128),
        std::make_tuple(128, 128, 136),
        std::make_tuple(128, 136, 128),
        std::make_tuple(129, 256, 512),
        std::make_tuple(256, 128, 512),
        std::make_tuple(128, 256, 512)
    });
}


TEST(CUDATest, MatmulVectorStorageHalfAccuracy) {
    run_half_matmul_accuracy_suite(cuda_matmul_vector_storage, "vector_storage_half", {
        std::make_tuple(33, 128, 128),
        std::make_tuple(129, 128, 128),
        std::make_tuple(128, 128, 136),
        std::make_tuple(128, 136, 128),
        std::make_tuple(129, 256, 512),
        std::make_tuple(256, 128, 512),
        std::make_tuple(128, 256, 512)
    });
}

TEST(CUDATest, MatmulDoubleBufferedSwizzledHalfAccuracy) {
    run_half_matmul_accuracy_suite(cuda_matmul_double_buffered_swizzled, "double_buffered_swizzled_half", {
        std::make_tuple(33, 128, 128),
        std::make_tuple(129, 128, 128),
        std::make_tuple(128, 128, 136),
        std::make_tuple(128, 136, 128),
        std::make_tuple(129, 256, 512),
        std::make_tuple(256, 128, 512),
        std::make_tuple(128, 256, 512)
    });
}

TEST(CUDATest, Matmul3StageCpAsyncHalfAccuracy) {
    run_half_matmul_accuracy_suite(cuda_matmul_3stage_cpasync, "3stage_cpasync_half", {
        std::make_tuple(33, 128, 128),
        std::make_tuple(129, 128, 128),
        std::make_tuple(128, 128, 136),
        std::make_tuple(128, 136, 128),
        std::make_tuple(129, 256, 512),
        std::make_tuple(256, 128, 512),
        std::make_tuple(128, 256, 512)
    });
}

