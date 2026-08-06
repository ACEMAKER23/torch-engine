// Phase 2 kernel performance benchmark (see TEST_PLAN.md section 2).
//
// Times every matmul kernel variant against cuBLAS across a range of square
// and rectangular problem shapes, and reports achieved TFLOPS / memory
// bandwidth / percentage of cuBLAS performance.
//
// Output: human readable table on stdout + JSON file (default:
// build/benchmark_results.json) matching the schema documented in
// TEST_PLAN.md's "Appendix: Benchmark Data Format".

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "../core/cuda_utils.h"
#include "../core/dtype_utils.h"
#include "../cuda/elementwise_cuda.h"

namespace {

struct Shape {
    int M, K, N;
};

struct BenchmarkResult {
    std::string kernel_name;
    std::string dtype;
    int M, K, N;
    double time_ms;
    double tflops;
    double bandwidth_gb_s;
    double pct_of_cublas = -1.0;  // filled in during post-processing
};

std::string gpu_name_cached;

// ---------------------------------------------------------------------
// Timing helper: warm up, then time `iterations` launches with CUDA events.
// ---------------------------------------------------------------------
double time_kernel_ms(const std::function<void()>& launch, int warmup, int iterations) {
    for (int i = 0; i < warmup; ++i) launch();
    cuda_check_error(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start, stop;
    cuda_check_error(cudaEventCreate(&start), "event create");
    cuda_check_error(cudaEventCreate(&stop), "event create");

    cuda_check_error(cudaEventRecord(start), "event record start");
    for (int i = 0; i < iterations; ++i) launch();
    cuda_check_error(cudaEventRecord(stop), "event record stop");
    cuda_check_error(cudaEventSynchronize(stop), "event sync");

    float ms = 0.0f;
    cuda_check_error(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<double>(ms) / iterations;
}

double tflops_for(int M, int K, int N, double time_ms) {
    double flops = 2.0 * M * K * N;  // multiply-add counts as 2 ops
    return flops / (time_ms * 1e-3) / 1e12;
}

double bandwidth_gb_s(int M, int K, int N, size_t in_elem_size, size_t out_elem_size, double time_ms) {
    size_t bytes_read = (static_cast<size_t>(M) * K + static_cast<size_t>(K) * N) * in_elem_size;
    size_t bytes_written = static_cast<size_t>(M) * N * out_elem_size;
    return static_cast<double>(bytes_read + bytes_written) / (time_ms * 1e-3) / 1e9;
}

// Pick iteration count based on problem size so the whole sweep finishes
// in a reasonable time on a single GPU.
int iterations_for(int M, int K, int N) {
    long long work = static_cast<long long>(M) * K * N;
    if (work <= (128LL * 128 * 128)) return 100;
    if (work <= (512LL * 512 * 512)) return 50;
    if (work <= (2048LL * 2048 * 2048)) return 20;
    return 5;
}

int warmup_for(int iterations) { return std::max(3, iterations / 5); }

// ---------------------------------------------------------------------
// float32 kernels
// ---------------------------------------------------------------------
using f32_matmul_fn_t = void (*)(const float*, const float*, float*, int, int, int);

BenchmarkResult run_f32_kernel(const std::string& name, f32_matmul_fn_t kernel, const Shape& s) {
    const size_t A_elems = static_cast<size_t>(s.M) * s.K;
    const size_t B_elems = static_cast<size_t>(s.K) * s.N;
    const size_t C_elems = static_cast<size_t>(s.M) * s.N;

    std::vector<float> h_A(A_elems), h_B(B_elems);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : h_A) x = dist(gen);
    for (auto& x : h_B) x = dist(gen);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cuda_check_error(cudaMalloc(&d_A, A_elems * sizeof(float)), "malloc A");
    cuda_check_error(cudaMalloc(&d_B, B_elems * sizeof(float)), "malloc B");
    cuda_check_error(cudaMalloc(&d_C, C_elems * sizeof(float)), "malloc C");
    cuda_check_error(cudaMemcpy(d_A, h_A.data(), A_elems * sizeof(float), cudaMemcpyHostToDevice), "copy A");
    cuda_check_error(cudaMemcpy(d_B, h_B.data(), B_elems * sizeof(float), cudaMemcpyHostToDevice), "copy B");

    int iters = iterations_for(s.M, s.K, s.N);
    double time_ms = time_kernel_ms([&]() { kernel(d_A, d_B, d_C, s.M, s.K, s.N); }, warmup_for(iters), iters);
    cuda_check_error(cudaGetLastError(), (name + " launch").c_str());

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return {name, "float32", s.M, s.K, s.N, time_ms, tflops_for(s.M, s.K, s.N, time_ms),
            bandwidth_gb_s(s.M, s.K, s.N, sizeof(float), sizeof(float), time_ms)};
}

BenchmarkResult run_cublas_f32_baseline(const Shape& s) {
    const size_t A_elems = static_cast<size_t>(s.M) * s.K;
    const size_t B_elems = static_cast<size_t>(s.K) * s.N;
    const size_t C_elems = static_cast<size_t>(s.M) * s.N;

    std::vector<float> h_A(A_elems), h_B(B_elems);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : h_A) x = dist(gen);
    for (auto& x : h_B) x = dist(gen);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cuda_check_error(cudaMalloc(&d_A, A_elems * sizeof(float)), "malloc A");
    cuda_check_error(cudaMalloc(&d_B, B_elems * sizeof(float)), "malloc B");
    cuda_check_error(cudaMalloc(&d_C, C_elems * sizeof(float)), "malloc C");
    cuda_check_error(cudaMemcpy(d_A, h_A.data(), A_elems * sizeof(float), cudaMemcpyHostToDevice), "copy A");
    cuda_check_error(cudaMemcpy(d_B, h_B.data(), B_elems * sizeof(float), cudaMemcpyHostToDevice), "copy B");

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    int iters = iterations_for(s.M, s.K, s.N);
    double time_ms = time_kernel_ms(
        [&]() {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, s.N, s.M, s.K, &alpha, d_B, s.N, d_A, s.K, &beta, d_C, s.N);
        },
        warmup_for(iters), iters);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return {"cuBLAS (Sgemm)", "float32", s.M, s.K, s.N, time_ms, tflops_for(s.M, s.K, s.N, time_ms),
            bandwidth_gb_s(s.M, s.K, s.N, sizeof(float), sizeof(float), time_ms)};
}

// ---------------------------------------------------------------------
// half-in/half-out kernels
// ---------------------------------------------------------------------
using half_matmul_fn_t = void (*)(const half*, const half*, half*, int, int, int);

BenchmarkResult run_half_kernel(const std::string& name, half_matmul_fn_t kernel, const Shape& s) {
    const size_t A_elems = static_cast<size_t>(s.M) * s.K;
    const size_t B_elems = static_cast<size_t>(s.K) * s.N;
    const size_t C_elems = static_cast<size_t>(s.M) * s.N;

    std::vector<half> h_A(A_elems), h_B(B_elems);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& x : h_A) x = __float2half(dist(gen));
    for (auto& x : h_B) x = __float2half(dist(gen));

    half *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cuda_check_error(cudaMalloc(&d_A, A_elems * sizeof(half)), "malloc A");
    cuda_check_error(cudaMalloc(&d_B, B_elems * sizeof(half)), "malloc B");
    cuda_check_error(cudaMalloc(&d_C, C_elems * sizeof(half)), "malloc C");
    cuda_check_error(cudaMemcpy(d_A, h_A.data(), A_elems * sizeof(half), cudaMemcpyHostToDevice), "copy A");
    cuda_check_error(cudaMemcpy(d_B, h_B.data(), B_elems * sizeof(half), cudaMemcpyHostToDevice), "copy B");

    int iters = iterations_for(s.M, s.K, s.N);
    double time_ms = time_kernel_ms([&]() { kernel(d_A, d_B, d_C, s.M, s.K, s.N); }, warmup_for(iters), iters);
    cuda_check_error(cudaGetLastError(), (name + " launch").c_str());

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return {name, "float16", s.M, s.K, s.N, time_ms, tflops_for(s.M, s.K, s.N, time_ms),
            bandwidth_gb_s(s.M, s.K, s.N, sizeof(half), sizeof(half), time_ms)};
}

BenchmarkResult run_cublas_half_baseline(const Shape& s) {
    const size_t A_elems = static_cast<size_t>(s.M) * s.K;
    const size_t B_elems = static_cast<size_t>(s.K) * s.N;
    const size_t C_elems = static_cast<size_t>(s.M) * s.N;

    std::vector<half> h_A(A_elems), h_B(B_elems);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& x : h_A) x = __float2half(dist(gen));
    for (auto& x : h_B) x = __float2half(dist(gen));

    half *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cuda_check_error(cudaMalloc(&d_A, A_elems * sizeof(half)), "malloc A");
    cuda_check_error(cudaMalloc(&d_B, B_elems * sizeof(half)), "malloc B");
    cuda_check_error(cudaMalloc(&d_C, C_elems * sizeof(half)), "malloc C");
    cuda_check_error(cudaMemcpy(d_A, h_A.data(), A_elems * sizeof(half), cudaMemcpyHostToDevice), "copy A");
    cuda_check_error(cudaMemcpy(d_B, h_B.data(), B_elems * sizeof(half), cudaMemcpyHostToDevice), "copy B");

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    int iters = iterations_for(s.M, s.K, s.N);
    double time_ms = time_kernel_ms(
        [&]() {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, s.N, s.M, s.K, &alpha, d_B, CUDA_R_16F, s.N, d_A,
                         CUDA_R_16F, s.K, &beta, d_C, CUDA_R_16F, s.N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        },
        warmup_for(iters), iters);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return {"cuBLAS (GemmEx fp16)", "float16", s.M, s.K, s.N, time_ms, tflops_for(s.M, s.K, s.N, time_ms),
            bandwidth_gb_s(s.M, s.K, s.N, sizeof(half), sizeof(half), time_ms)};
}

// ---------------------------------------------------------------------
// half-in/float32-out tensor core kernels
// ---------------------------------------------------------------------
using fp16_tc_matmul_fn_t = void (*)(const half*, const half*, float*, int, int, int);

BenchmarkResult run_tensorcore_kernel(const std::string& name, fp16_tc_matmul_fn_t kernel, const Shape& s) {
    const size_t A_elems = static_cast<size_t>(s.M) * s.K;
    const size_t B_elems = static_cast<size_t>(s.K) * s.N;
    const size_t C_elems = static_cast<size_t>(s.M) * s.N;

    std::vector<half> h_A(A_elems), h_B(B_elems);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& x : h_A) x = __float2half(dist(gen));
    for (auto& x : h_B) x = __float2half(dist(gen));

    half *d_A = nullptr, *d_B = nullptr;
    float* d_C = nullptr;
    cuda_check_error(cudaMalloc(&d_A, A_elems * sizeof(half)), "malloc A");
    cuda_check_error(cudaMalloc(&d_B, B_elems * sizeof(half)), "malloc B");
    cuda_check_error(cudaMalloc(&d_C, C_elems * sizeof(float)), "malloc C");
    cuda_check_error(cudaMemcpy(d_A, h_A.data(), A_elems * sizeof(half), cudaMemcpyHostToDevice), "copy A");
    cuda_check_error(cudaMemcpy(d_B, h_B.data(), B_elems * sizeof(half), cudaMemcpyHostToDevice), "copy B");

    int iters = iterations_for(s.M, s.K, s.N);
    double time_ms = time_kernel_ms([&]() { kernel(d_A, d_B, d_C, s.M, s.K, s.N); }, warmup_for(iters), iters);
    cuda_check_error(cudaGetLastError(), (name + " launch").c_str());

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return {name, "float16(tc)", s.M, s.K, s.N, time_ms, tflops_for(s.M, s.K, s.N, time_ms),
            bandwidth_gb_s(s.M, s.K, s.N, sizeof(half), sizeof(float), time_ms)};
}

BenchmarkResult run_cublas_tensorcore_baseline(const Shape& s) {
    const size_t A_elems = static_cast<size_t>(s.M) * s.K;
    const size_t B_elems = static_cast<size_t>(s.K) * s.N;
    const size_t C_elems = static_cast<size_t>(s.M) * s.N;

    std::vector<half> h_A(A_elems), h_B(B_elems);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& x : h_A) x = __float2half(dist(gen));
    for (auto& x : h_B) x = __float2half(dist(gen));

    half *d_A = nullptr, *d_B = nullptr;
    float* d_C = nullptr;
    cuda_check_error(cudaMalloc(&d_A, A_elems * sizeof(half)), "malloc A");
    cuda_check_error(cudaMalloc(&d_B, B_elems * sizeof(half)), "malloc B");
    cuda_check_error(cudaMalloc(&d_C, C_elems * sizeof(float)), "malloc C");
    cuda_check_error(cudaMemcpy(d_A, h_A.data(), A_elems * sizeof(half), cudaMemcpyHostToDevice), "copy A");
    cuda_check_error(cudaMemcpy(d_B, h_B.data(), B_elems * sizeof(half), cudaMemcpyHostToDevice), "copy B");

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    int iters = iterations_for(s.M, s.K, s.N);
    double time_ms = time_kernel_ms(
        [&]() {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, s.N, s.M, s.K, &alpha, d_B, CUDA_R_16F, s.N, d_A,
                         CUDA_R_16F, s.K, &beta, d_C, CUDA_R_32F, s.N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        },
        warmup_for(iters), iters);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return {"cuBLAS (GemmEx tensor-core)", "float16(tc)", s.M, s.K, s.N, time_ms, tflops_for(s.M, s.K, s.N, time_ms),
            bandwidth_gb_s(s.M, s.K, s.N, sizeof(half), sizeof(float), time_ms)};
}

// ---------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------
void print_table(const std::vector<BenchmarkResult>& results) {
    std::cout << std::left << std::setw(34) << "Kernel" << std::setw(10) << "DType" << std::setw(16) << "Shape (MxKxN)"
              << std::right << std::setw(12) << "Time(ms)" << std::setw(12) << "TFLOPS" << std::setw(14) << "BW(GB/s)"
              << std::setw(12) << "% cuBLAS" << "\n";
    std::cout << std::string(110, '-') << "\n";
    for (const auto& r : results) {
        std::ostringstream shape;
        shape << r.M << "x" << r.K << "x" << r.N;
        std::cout << std::left << std::setw(34) << r.kernel_name << std::setw(10) << r.dtype << std::setw(16)
                  << shape.str() << std::right << std::fixed << std::setprecision(3) << std::setw(12) << r.time_ms
                  << std::setw(12) << r.tflops << std::setw(14) << r.bandwidth_gb_s;
        if (r.pct_of_cublas >= 0.0) {
            std::cout << std::setw(11) << std::setprecision(1) << r.pct_of_cublas << "%";
        } else {
            std::cout << std::setw(12) << "-";
        }
        std::cout << "\n";
    }
}

void write_json(const std::vector<BenchmarkResult>& results, const std::string& path) {
    std::ofstream out(path);
    out << "[\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out << "  {\n";
        out << "    \"kernel_name\": \"" << r.kernel_name << "\",\n";
        out << "    \"dtype\": \"" << r.dtype << "\",\n";
        out << "    \"M\": " << r.M << ",\n";
        out << "    \"K\": " << r.K << ",\n";
        out << "    \"N\": " << r.N << ",\n";
        out << "    \"time_ms\": " << r.time_ms << ",\n";
        out << "    \"tflops\": " << r.tflops << ",\n";
        out << "    \"bandwidth_gb_s\": " << r.bandwidth_gb_s << ",\n";
        out << "    \"pct_of_cublas\": " << r.pct_of_cublas << ",\n";
        out << "    \"gpu_name\": \"" << gpu_name_cached << "\"\n";
        out << "  }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    out << "]\n";
    std::cout << "\nWrote " << results.size() << " results to " << path << "\n";
}

// Fill in pct_of_cublas for every non-baseline result by matching
// (dtype, M, K, N) against the corresponding cuBLAS baseline entry.
void compute_relative_performance(std::vector<BenchmarkResult>& results) {
    std::map<std::tuple<std::string, int, int, int>, double> baseline_tflops;
    for (const auto& r : results) {
        if (r.kernel_name.rfind("cuBLAS", 0) == 0) {
            baseline_tflops[{r.dtype, r.M, r.K, r.N}] = r.tflops;
        }
    }
    for (auto& r : results) {
        auto it = baseline_tflops.find({r.dtype, r.M, r.K, r.N});
        if (it != baseline_tflops.end() && it->second > 0.0) {
            r.pct_of_cublas = 100.0 * r.tflops / it->second;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (!cuda_available()) {
        std::cerr << "CUDA not available; cannot run benchmark.\n";
        return 1;
    }

    std::string out_path = "benchmark_results.json";
    if (argc > 1) out_path = argv[1];

    gpu_name_cached = cuda_device_name(0);
    int cc_major = 0, cc_minor = 0;
    cuda_device_capability(0, &cc_major, &cc_minor);
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    std::cout << "GPU: " << gpu_name_cached << " (SM " << cc_major << "." << cc_minor << "), "
              << (total_mem / (1024 * 1024)) << " MiB total, " << (free_mem / (1024 * 1024)) << " MiB free\n\n";

    std::vector<BenchmarkResult> results;

    // ---- Square shapes, tiered by kernel sophistication ----
    std::vector<Shape> small_sizes = {{128, 128, 128}, {256, 256, 256}, {512, 512, 512}};
    std::vector<Shape> medium_sizes = {{128, 128, 128}, {256, 256, 256}, {512, 512, 512}, {1024, 1024, 1024}};
    std::vector<Shape> large_sizes = {{128, 128, 128},  {256, 256, 256},   {512, 512, 512},
                                       {1024, 1024, 1024}, {2048, 2048, 2048}};
    std::vector<Shape> xlarge_sizes = {{128, 128, 128},  {256, 256, 256},   {512, 512, 512},   {1024, 1024, 1024},
                                        {2048, 2048, 2048}, {4096, 4096, 4096}};
    // Tensor cores handle much larger problems efficiently.
    std::vector<Shape> tc_sizes = {{128, 128, 128},  {256, 256, 256},   {512, 512, 512},   {1024, 1024, 1024},
                                    {2048, 2048, 2048}, {4096, 4096, 4096}, {8192, 8192, 8192}};

    // Representative rectangular LLM-workload shapes (scaled to fit a 6GB GPU).
    std::vector<Shape> rect_sizes = {
        {1024, 4096, 1024},  // attention A*V-like
        {1024, 4096, 4096},  // attention Q*K^T-like
        {4096, 1024, 4096},  // linear layer-like
        {2048, 2048, 2048},
    };

    // Non-16-aligned shapes: launch_matmul_tensor_core / launch_matmul_ampere take
    // a *padded* code path for these (extra cudaMalloc/cudaMemset/cudaMemcpy2D
    // around the matmul_kernel_tensor_core / matmul_kernel_ampere launch), which
    // the aligned shapes above never exercise.
    std::vector<Shape> tc_padded_sizes = {
        {129, 65, 17},     // small, all three dims unaligned
        {127, 255, 129},   // small, all three dims unaligned
        {257, 129, 511},   // medium, all three dims unaligned
        {1000, 1000, 1000},  // large-ish, all three dims unaligned
        {2001, 2001, 2001},  // large, all three dims unaligned
    };

    auto sweep_f32 = [&](const std::string& name, f32_matmul_fn_t fn, const std::vector<Shape>& shapes) {
        for (const auto& s : shapes) results.push_back(run_f32_kernel(name, fn, s));
    };
    auto sweep_cublas_f32 = [&](const std::vector<Shape>& shapes) {
        for (const auto& s : shapes) results.push_back(run_cublas_f32_baseline(s));
    };
    auto sweep_half = [&](const std::string& name, half_matmul_fn_t fn, const std::vector<Shape>& shapes) {
        for (const auto& s : shapes) results.push_back(run_half_kernel(name, fn, s));
    };
    auto sweep_cublas_half = [&](const std::vector<Shape>& shapes) {
        for (const auto& s : shapes) results.push_back(run_cublas_half_baseline(s));
    };
    auto sweep_tc = [&](const std::string& name, fp16_tc_matmul_fn_t fn, const std::vector<Shape>& shapes) {
        for (const auto& s : shapes) results.push_back(run_tensorcore_kernel(name, fn, s));
    };
    auto sweep_cublas_tc = [&](const std::vector<Shape>& shapes) {
        for (const auto& s : shapes) results.push_back(run_cublas_tensorcore_baseline(s));
    };

    std::cout << "Running float32 cuBLAS baseline...\n";
    sweep_cublas_f32(xlarge_sizes);
    sweep_cublas_f32(rect_sizes);

    std::cout << "Benchmarking matmul_kernel (naive)...\n";
    sweep_f32("naive", cuda_matmul, small_sizes);

    std::cout << "Benchmarking matmul_kernel_shared_memory...\n";
    sweep_f32("shared_memory", cuda_matmul_shared_memory, medium_sizes);

    std::cout << "Benchmarking matmul_kernel_register_blocking...\n";
    sweep_f32("register_blocking", cuda_matmul_register_blocking, large_sizes);

    std::cout << "Benchmarking matmul_kernel_vectorized_input...\n";
    sweep_f32("vectorized_input", cuda_matmul_vectorized_input, large_sizes);

    std::cout << "Benchmarking matmul_kernel_warp_tiling...\n";
    sweep_f32("warp_tiling", cuda_matmul_warp_tiling, xlarge_sizes);
    sweep_f32("warp_tiling", cuda_matmul_warp_tiling, rect_sizes);

    std::cout << "Benchmarking matmul_kernel_double_buffered...\n";
    sweep_f32("double_buffered", cuda_matmul_double_buffered, xlarge_sizes);
    sweep_f32("double_buffered", cuda_matmul_double_buffered, rect_sizes);

    std::cout << "Benchmarking matmul_kernel_double_buffered_cpasync...\n";
    sweep_f32("double_buffered_cpasync", cuda_matmul_double_buffered_cpasync, xlarge_sizes);
    sweep_f32("double_buffered_cpasync", cuda_matmul_double_buffered_cpasync, rect_sizes);

    std::cout << "Benchmarking matmul_kernel_double_buffered_swizzled...\n";
    sweep_f32("double_buffered_swizzled", cuda_matmul_double_buffered_swizzled, xlarge_sizes);
    sweep_f32("double_buffered_swizzled", cuda_matmul_double_buffered_swizzled, rect_sizes);

    std::cout << "Benchmarking matmul_kernel_vector_storage...\n";
    sweep_f32("vector_storage", cuda_matmul_vector_storage, xlarge_sizes);
    sweep_f32("vector_storage", cuda_matmul_vector_storage, rect_sizes);

    std::cout << "Benchmarking matmul_kernel_3stage_cpasync...\n";
    sweep_f32("3stage_cpasync", cuda_matmul_3stage_cpasync, xlarge_sizes);
    sweep_f32("3stage_cpasync", cuda_matmul_3stage_cpasync, rect_sizes);

    // FP16 (non tensor-core) kernels
    std::cout << "Running float16 cuBLAS baseline...\n";
    sweep_cublas_half(large_sizes);

    std::cout << "Benchmarking matmul_kernel (naive, half)...\n";
    sweep_half("naive_half", cuda_matmul, small_sizes);
    std::cout << "Benchmarking matmul_kernel_shared_memory (half)...\n";
    sweep_half("shared_memory_half", cuda_matmul_shared_memory, medium_sizes);
    std::cout << "Benchmarking matmul_kernel_register_blocking (half)...\n";
    sweep_half("register_blocking_half", cuda_matmul_register_blocking, large_sizes);
    std::cout << "Benchmarking matmul_kernel_vectorized_input (half)...\n";
    sweep_half("vectorized_input_half", cuda_matmul_vectorized_input, large_sizes);
    std::cout << "Benchmarking matmul_kernel_warp_tiling (half)...\n";
    sweep_half("warp_tiling_half", cuda_matmul_warp_tiling, large_sizes);
    std::cout << "Benchmarking matmul_kernel_double_buffered (half)...\n";
    sweep_half("double_buffered_half", cuda_matmul_double_buffered, large_sizes);
    std::cout << "Benchmarking matmul_kernel_double_buffered_cpasync (half)...\n";
    sweep_half("double_buffered_cpasync_half", cuda_matmul_double_buffered_cpasync, large_sizes);
    std::cout << "Benchmarking matmul_kernel_double_buffered_swizzled (half)...\n";
    sweep_half("double_buffered_swizzled_half", cuda_matmul_double_buffered_swizzled, large_sizes);
    std::cout << "Benchmarking matmul_kernel_vector_storage (half)...\n";
    sweep_half("vector_storage_half", cuda_matmul_vector_storage, large_sizes);
    std::cout << "Benchmarking matmul_kernel_3stage_cpasync (half)...\n";
    sweep_half("3stage_cpasync_half", cuda_matmul_3stage_cpasync, large_sizes);

    // Tensor-core kernels (require SM80+)
    if (cc_major >= 8) {
        std::cout << "Running tensor-core cuBLAS baseline...\n";
        sweep_cublas_tc(tc_sizes);
        sweep_cublas_tc(rect_sizes);

        std::cout << "Benchmarking matmul_kernel_tensor_core (WMMA)...\n";
        sweep_tc("tensor_core", launch_matmul_tensor_core, tc_sizes);
        sweep_tc("tensor_core", launch_matmul_tensor_core, rect_sizes);

        std::cout << "Benchmarking matmul_kernel_ampere (PTX mma.sync)...\n";
        sweep_tc("ampere", launch_matmul_ampere, tc_sizes);
        sweep_tc("ampere", launch_matmul_ampere, rect_sizes);

        std::cout << "Running cuBLAS baseline for non-16-aligned (padded-path) shapes...\n";
        sweep_cublas_tc(tc_padded_sizes);

        std::cout << "Benchmarking matmul_kernel_tensor_core (padded path, non-16-aligned shapes)...\n";
        sweep_tc("tensor_core_padded", launch_matmul_tensor_core, tc_padded_sizes);

        std::cout << "Benchmarking matmul_kernel_ampere (padded path, non-16-aligned shapes)...\n";
        sweep_tc("ampere_padded", launch_matmul_ampere, tc_padded_sizes);
    } else {
        std::cout << "Skipping tensor-core kernels (requires SM80+, found SM" << cc_major << "." << cc_minor
                   << ")\n";
    }

    compute_relative_performance(results);
    std::cout << "\n";
    print_table(results);
    write_json(results, out_path);

    return 0;
}
