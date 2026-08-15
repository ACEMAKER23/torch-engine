#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <array>
#include <cstdio>
#include <cstdlib>

#include "../nn/gpt.h"
#include "../optimizer/adamw.h"
#include "../loss/CrossEntropyLoss.h"
#include "../tensor/tensor.h"
#include "../core/dtype.h"
#include "../core/cuda_utils.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace std;

std::string shell_capture(const std::string& cmd) {
    std::array<char, 256> buffer{};
    std::string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "unknown";
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        result += buffer.data();
    }
    pclose(pipe);
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) result.pop_back();
    return result.empty() ? "unknown" : result;
}

std::string getenv_or(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string(fallback);
}

class ShakespeareDataLoader {
public:
    ShakespeareDataLoader(const string& path, int seed = 42) {
        ifstream file(path);
        if (!file.is_open()) {
            throw runtime_error("Cannot open dataset: " + path);
        }
        stringstream buffer;
        buffer << file.rdbuf();
        text_ = buffer.str();
        file.close();

        set<char> chars(text_.begin(), text_.end());
        for (char c : chars) chars_.push_back(c);
        for (size_t i = 0; i < chars_.size(); ++i) stoi_[chars_[i]] = static_cast<int64_t>(i);
        for (char c : text_) data_.push_back(stoi_[c]);
        rng_.seed(seed);
    }

    int64_t vocab_size() const { return static_cast<int64_t>(chars_.size()); }

    void next_batch(int64_t batch_size, int64_t seq_len,
                    Tensor& input, Tensor& target) {
        int64_t N = static_cast<int64_t>(data_.size());
        uniform_int_distribution<int64_t> dist(0, N - seq_len - 1);

        // Build on CPU first, then move to CUDA
        Tensor input_cpu ({batch_size, seq_len}, DType::Int64, Device::CPU);
        Tensor target_cpu({batch_size, seq_len}, DType::Int64, Device::CPU);

        for (int64_t b = 0; b < batch_size; ++b) {
            int64_t start = dist(rng_);
            for (int64_t t = 0; t < seq_len; ++t) {
                input_cpu .at<int64_t>(b * seq_len + t) = data_[start + t];
                target_cpu.at<int64_t>(b * seq_len + t) = data_[start + t + 1];
            }
        }
        input  = input_cpu .toDevice(Device::CUDA);
        target = target_cpu.toDevice(Device::CUDA);
    }

private:
    string text_;
    vector<char> chars_;
    map<char, int64_t> stoi_;
    vector<int64_t> data_;
    mt19937_64 rng_;
};

struct BenchConfig {
    int64_t max_seq_len = 128;
    int64_t embed_dim   = 64;
    int64_t num_heads   = 4;
    int64_t num_layers  = 2;
    int64_t ff_dim      = 256;
    int64_t batch_size  = 2;
    int64_t seq_len     = 16;
    int64_t warmup_steps    = 10;
    int64_t measured_steps  = 40;
    float   learning_rate   = 0.001f;
    float   weight_decay    = 0.01f;
    float   dropout         = 0.0f;

    static BenchConfig from_env() {
        BenchConfig cfg;
        const char* v = nullptr;
        if ((v = getenv("TENSOR_EMBED_DIM")))   cfg.embed_dim   = std::stoll(v);
        if ((v = getenv("TENSOR_NUM_HEADS")))   cfg.num_heads   = std::stoll(v);
        if ((v = getenv("TENSOR_NUM_LAYERS")))  cfg.num_layers  = std::stoll(v);
        if ((v = getenv("TENSOR_FF_DIM")))      cfg.ff_dim      = std::stoll(v);
        if ((v = getenv("TENSOR_BATCH_SIZE")))  cfg.batch_size  = std::stoll(v);
        if ((v = getenv("TENSOR_SEQ_LEN")))     cfg.seq_len     = std::stoll(v);
        if ((v = getenv("TENSOR_MAX_SEQ_LEN"))) cfg.max_seq_len = std::stoll(v);
        if ((v = getenv("TENSOR_MEASURED_STEPS"))) cfg.measured_steps = std::stoll(v);
        if ((v = getenv("TENSOR_WARMUP_STEPS")))   cfg.warmup_steps   = std::stoll(v);
        return cfg;
    }
};

int main(int argc, char* argv[]) {
    if (!cuda_available()) {
        cerr << "CUDA not available\n";
        return 1;
    }

    string data_path = "data/tinyshakespeare.txt";
    if (argc > 1) data_path = argv[1];

    ShakespeareDataLoader loader(data_path);
    BenchConfig cfg = BenchConfig::from_env();

    // Build model on CPU, then move all parameters to CUDA
    GPT model(loader.vocab_size(), cfg.max_seq_len, cfg.embed_dim,
              cfg.num_heads, cfg.num_layers, cfg.ff_dim,
              DType::Float32, cfg.dropout);
    model.to_cuda();

    auto params = model.parameters();
    adamw optimizer(params, 0.9f, 0.999f, 1e-8f, cfg.weight_decay);
    crossEntropyLoss criterion;

    Tensor input ({1}, DType::Int64, Device::CUDA);
    Tensor target({1}, DType::Int64, Device::CUDA);

    cout << "Starting GPU warmup (" << cfg.warmup_steps << " steps)...\n";

    // Warmup
    for (int64_t s = 0; s < cfg.warmup_steps; ++s) {
        loader.next_batch(cfg.batch_size, cfg.seq_len, input, target);
        model.zero_grad();
        Tensor logits = model.forward(input);
        Tensor loss   = criterion.forward_batched(logits, target);
        loss.backward();
        optimizer.step(cfg.learning_rate);
        if (s == 0) cout << "  step 1 OK\n" << flush;
    }

    bool profile_backward = getenv("TENSOR_PROFILE_BACKWARD") != nullptr;
    bool profile_phases = getenv("TENSOR_PROFILE_PHASES") != nullptr;
    if (profile_backward) {
        cout << "Backward profiling enabled (per-GradFn synced GPU timing).\n";
        Tensor::enable_grad_profile(true);
        Tensor::reset_grad_profile();
    }
    if (profile_phases) {
        cout << "Phase profiling enabled; each phase is synchronized separately.\n";
    }

    cout << "Warmup done. Benchmarking " << cfg.measured_steps << " steps...\n";

    // Default mode measures training throughput with only one synchronization around
    // the measured region. TENSOR_PROFILE_PHASES=1 enables synchronized phase timing
    // for diagnosis, but that mode is not used for headline throughput comparisons.
    double t_data_ms = 0.0, t_zero_ms = 0.0, t_forward_ms = 0.0;
    double t_loss_ms = 0.0, t_backward_ms = 0.0, t_opt_ms = 0.0;

    auto now = []() { return chrono::high_resolution_clock::now(); };
    auto ms  = [](auto a, auto b) {
        return chrono::duration<double, std::milli>(b - a).count();
    };

    Tensor last_loss({1}, DType::Float32, Device::CUDA);
    cudaDeviceSynchronize();
    auto start = now();
    for (int64_t s = 0; s < cfg.measured_steps; ++s) {
        if (profile_phases) {
            auto t0 = now();
            loader.next_batch(cfg.batch_size, cfg.seq_len, input, target);
            cudaDeviceSynchronize();
            auto t1 = now();
            t_data_ms += ms(t0, t1);

            t0 = now();
            model.zero_grad();
            cudaDeviceSynchronize();
            t1 = now();
            t_zero_ms += ms(t0, t1);

            t0 = now();
            Tensor logits = model.forward(input);
            cudaDeviceSynchronize();
            t1 = now();
            t_forward_ms += ms(t0, t1);

            t0 = now();
            Tensor loss = criterion.forward_batched(logits, target);
            last_loss = loss;
            cudaDeviceSynchronize();
            t1 = now();
            t_loss_ms += ms(t0, t1);

            t0 = now();
            loss.backward();
            cudaDeviceSynchronize();
            t1 = now();
            t_backward_ms += ms(t0, t1);

            t0 = now();
            optimizer.step(cfg.learning_rate);
            cudaDeviceSynchronize();
            t1 = now();
            t_opt_ms += ms(t0, t1);
        } else {
            loader.next_batch(cfg.batch_size, cfg.seq_len, input, target);
            model.zero_grad();
            Tensor logits = model.forward(input);
            Tensor loss = criterion.forward_batched(logits, target);
            last_loss = loss;
            loss.backward();
            optimizer.step(cfg.learning_rate);
        }
    }
    cudaDeviceSynchronize();
    auto end = now();
    Tensor final_loss_cpu = last_loss.toDevice(Device::CPU);
    float final_loss = final_loss_cpu.at<float>(0);

    double elapsed     = ms(start, end) / 1000.0;
    double total_tokens = static_cast<double>(cfg.measured_steps * cfg.batch_size * cfg.seq_len);
    double tokens_per_sec = total_tokens / elapsed;
    double steps_per_sec  = static_cast<double>(cfg.measured_steps) / elapsed;
    double time_per_step  = (elapsed * 1000.0) / static_cast<double>(cfg.measured_steps);

    double total_ms = elapsed * 1000.0;
    auto pct = [total_ms](double t) { return (total_ms > 1e-6) ? (t / total_ms) * 100.0 : 0.0; };

    int active_device = cuda_get_device();
    int cc_major = 0, cc_minor = 0;
    cuda_device_capability(active_device, &cc_major, &cc_minor);
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    string out_path = "gpt_benchmark_gpu_results.json";
    ofstream out(out_path);
    out << "{\n";
    out << "  \"framework\": \"TENSOR\",\n";
    out << "  \"device\": \"CUDA\",\n";
    out << "  \"hardware\": {\n";
    out << "    \"os\": \"" << shell_capture("uname -srmo") << "\",\n";
    out << "    \"cpu\": \"" << shell_capture("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//'") << "\",\n";
    out << "    \"gpu\": \"" << cuda_device_name(active_device) << "\",\n";
    out << "    \"cuda_device\": " << active_device << ",\n";
    out << "    \"compute_capability\": \"" << cc_major << "." << cc_minor << "\",\n";
    out << "    \"cuda_runtime\": " << CUDART_VERSION << ",\n";
    out << "    \"gpu_total_mem_mib\": " << (total_mem / (1024 * 1024)) << ",\n";
    out << "    \"compiler\": \"" << __VERSION__ << "\"\n";
    out << "  },\n";
    out << "  \"model_config\": {\n";
    out << "    \"vocab_size\": " << loader.vocab_size() << ",\n";
    out << "    \"max_seq_len\": " << cfg.max_seq_len << ",\n";
    out << "    \"embed_dim\": " << cfg.embed_dim << ",\n";
    out << "    \"num_heads\": " << cfg.num_heads << ",\n";
    out << "    \"num_layers\": " << cfg.num_layers << ",\n";
    out << "    \"ff_dim\": " << cfg.ff_dim << "\n";
    out << "  },\n";
    out << "  \"training_config\": {\n";
    out << "    \"batch_size\": " << cfg.batch_size << ",\n";
    out << "    \"seq_len\": " << cfg.seq_len << ",\n";
    out << "    \"measured_steps\": " << cfg.measured_steps << ",\n";
    out << "    \"learning_rate\": " << cfg.learning_rate << ",\n";
    out << "    \"dropout\": " << cfg.dropout << ",\n";
    out << "    \"gemm_backend\": \"" << getenv_or("TENSOR_GEMM_BACKEND", "cublas") << "\"\n";
    out << "  },\n";
    out << "  \"metrics\": {\n";
    out << "    \"timing_mode\": \"" << (profile_phases ? "phase_profile" : "throughput") << "\",\n";
    out << "    \"total_time_s\": " << fixed << setprecision(4) << elapsed << ",\n";
    out << "    \"total_tokens\": " << static_cast<int64_t>(total_tokens) << ",\n";
    out << "    \"tokens_per_sec\": " << fixed << setprecision(2) << tokens_per_sec << ",\n";
    out << "    \"steps_per_sec\": " << fixed << setprecision(2) << steps_per_sec << ",\n";
    out << "    \"time_per_step_ms\": " << fixed << setprecision(2) << time_per_step << ",\n";
    out << "    \"final_loss\": " << fixed << setprecision(6) << final_loss;
    if (profile_phases) {
        out << ",\n";
        out << "    \"phase_breakdown_ms\": {\n";
    out << "      \"data_batch\": " << fixed << setprecision(4) << (t_data_ms / cfg.measured_steps) << ",\n";
    out << "      \"zero_grad\": "  << fixed << setprecision(4) << (t_zero_ms  / cfg.measured_steps) << ",\n";
    out << "      \"forward\": "    << fixed << setprecision(4) << (t_forward_ms / cfg.measured_steps) << ",\n";
    out << "      \"loss\": "       << fixed << setprecision(4) << (t_loss_ms    / cfg.measured_steps) << ",\n";
    out << "      \"backward\": "   << fixed << setprecision(4) << (t_backward_ms / cfg.measured_steps) << ",\n";
        out << "      \"optimizer\": "  << fixed << setprecision(4) << (t_opt_ms     / cfg.measured_steps) << "\n";
        out << "    }\n";
    } else {
        out << "\n";
    }
    out << "  }\n";
    out << "}\n";
    out.close();

    if (profile_backward) {
        Tensor::print_grad_profile();
        Tensor::enable_grad_profile(false);
    }

    cout << "\nTENSOR GPT GPU benchmark complete.\n";
    cout << "Total time:    " << fixed << setprecision(2) << elapsed << " s\n";
    cout << "Tokens/sec:    " << fixed << setprecision(2) << tokens_per_sec << "\n";
    cout << "Steps/sec:     " << fixed << setprecision(2) << steps_per_sec << "\n";
    cout << "ms/step:       " << fixed << setprecision(2) << time_per_step << "\n";
    cout << "Final loss:    " << fixed << setprecision(6) << final_loss << "\n";
    cout << "GEMM backend:  " << getenv_or("TENSOR_GEMM_BACKEND", "cublas") << "\n";
    if (profile_phases) {
        cout << "\nPer-phase breakdown (avg ms / step):\n";
        cout << "  data batch:  " << fixed << setprecision(4) << (t_data_ms     / cfg.measured_steps) << " ms  (" << setprecision(2) << pct(t_data_ms)     << "%)\n";
        cout << "  zero_grad:   " << fixed << setprecision(4) << (t_zero_ms     / cfg.measured_steps) << " ms  (" << setprecision(2) << pct(t_zero_ms)     << "%)\n";
        cout << "  forward:     " << fixed << setprecision(4) << (t_forward_ms  / cfg.measured_steps) << " ms  (" << setprecision(2) << pct(t_forward_ms)  << "%)\n";
        cout << "  loss:        " << fixed << setprecision(4) << (t_loss_ms     / cfg.measured_steps) << " ms  (" << setprecision(2) << pct(t_loss_ms)     << "%)\n";
        cout << "  backward:    " << fixed << setprecision(4) << (t_backward_ms / cfg.measured_steps) << " ms  (" << setprecision(2) << pct(t_backward_ms) << "%)\n";
        cout << "  optimizer:   " << fixed << setprecision(4) << (t_opt_ms      / cfg.measured_steps) << " ms  (" << setprecision(2) << pct(t_opt_ms)      << "%)\n";
    }
    cout << "Results written to " << out_path << "\n";
    return 0;
}
