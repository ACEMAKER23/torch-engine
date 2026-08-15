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
#include <cstdio>
#include <cstdlib>
#include <array>

#include "../nn/gpt.h"
#include "../optimizer/adamw.h"
#include "../loss/CrossEntropyLoss.h"
#include "../tensor/tensor.h"
#include "../core/dtype.h"

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

        // Build vocabulary from sorted unique characters
        set<char> chars(text_.begin(), text_.end());
        for (char c : chars) {
            chars_.push_back(c);
        }
        for (size_t i = 0; i < chars_.size(); ++i) {
            stoi_[chars_[i]] = static_cast<int64_t>(i);
        }

        for (char c : text_) {
            data_.push_back(stoi_[c]);
        }

        rng_.seed(seed);
    }

    int64_t vocab_size() const { return static_cast<int64_t>(chars_.size()); }

    void next_batch(int64_t batch_size, int64_t seq_len,
                    Tensor& input, Tensor& target) {
        int64_t N = static_cast<int64_t>(data_.size());
        uniform_int_distribution<int64_t> dist(0, N - seq_len - 1);

        input = Tensor({batch_size, seq_len}, DType::Int64, Device::CPU);
        target = Tensor({batch_size, seq_len}, DType::Int64, Device::CPU);

        for (int64_t b = 0; b < batch_size; ++b) {
            int64_t start = dist(rng_);
            for (int64_t t = 0; t < seq_len; ++t) {
                input.at<int64_t>(b * seq_len + t) = data_[start + t];
                target.at<int64_t>(b * seq_len + t) = data_[start + t + 1];
            }
        }
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
    int64_t embed_dim = 64;
    int64_t num_heads = 4;
    int64_t num_layers = 2;
    int64_t ff_dim = 256;
    int64_t batch_size = 2;
    int64_t seq_len = 16;
    int64_t warmup_steps = 10;
    int64_t measured_steps = 40;
    float learning_rate = 0.001f;
    float weight_decay = 0.01f;
    float dropout = 0.0f;

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
    string data_path = "data/tinyshakespeare.txt";
    if (argc > 1) data_path = argv[1];

    ShakespeareDataLoader loader(data_path);
    BenchConfig cfg = BenchConfig::from_env();

    // Small model config for CPU benchmarking
    GPT model(loader.vocab_size(), cfg.max_seq_len, cfg.embed_dim,
              cfg.num_heads, cfg.num_layers, cfg.ff_dim,
              DType::Float32, cfg.dropout);

    auto params = model.parameters();
    adamw optimizer(params, 0.9f, 0.999f, 1e-8f, cfg.weight_decay);
    crossEntropyLoss criterion;

    Tensor input({1}, DType::Int64, Device::CPU);
    Tensor target({1}, DType::Int64, Device::CPU);

    // Warmup
    for (int64_t s = 0; s < cfg.warmup_steps; ++s) {
        loader.next_batch(cfg.batch_size, cfg.seq_len, input, target);
        model.zero_grad();
        Tensor logits = model.forward(input);
        Tensor loss = criterion.forward_batched(logits, target);
        loss.backward();
        optimizer.step(cfg.learning_rate);
    }

    // Benchmark. Keep the last loss tensor handle, but read its scalar value
    // only after timing so metric collection does not affect throughput.
    Tensor last_loss({1}, DType::Float32, Device::CPU);
    auto start = chrono::high_resolution_clock::now();
    for (int64_t s = 0; s < cfg.measured_steps; ++s) {
        loader.next_batch(cfg.batch_size, cfg.seq_len, input, target);
        model.zero_grad();
        Tensor logits = model.forward(input);
        Tensor loss = criterion.forward_batched(logits, target);
        loss.backward();
        optimizer.step(cfg.learning_rate);
        last_loss = loss;
    }
    auto end = chrono::high_resolution_clock::now();
    float final_loss = last_loss.at<float>(0);

    double elapsed = chrono::duration<double>(end - start).count();
    double total_tokens = static_cast<double>(cfg.measured_steps * cfg.batch_size * cfg.seq_len);
    double tokens_per_sec = total_tokens / elapsed;
    double steps_per_sec = static_cast<double>(cfg.measured_steps) / elapsed;
    double time_per_step = elapsed / static_cast<double>(cfg.measured_steps);

    string out_path = "gpt_benchmark_results.json";
    ofstream out(out_path);
    out << "{\n";
    out << "  \"framework\": \"TENSOR\",\n";
    out << "  \"device\": \"CPU\",\n";
    out << "  \"hardware\": {\n";
    out << "    \"os\": \"" << shell_capture("uname -srmo") << "\",\n";
    out << "    \"cpu\": \"" << shell_capture("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//'") << "\",\n";
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
    out << "    \"dropout\": " << cfg.dropout << "\n";
    out << "  },\n";
    out << "  \"metrics\": {\n";
    out << "    \"total_time_s\": " << fixed << setprecision(4) << elapsed << ",\n";
    out << "    \"total_tokens\": " << static_cast<int64_t>(total_tokens) << ",\n";
    out << "    \"tokens_per_sec\": " << fixed << setprecision(2) << tokens_per_sec << ",\n";
    out << "    \"steps_per_sec\": " << fixed << setprecision(2) << steps_per_sec << ",\n";
    out << "    \"time_per_step_ms\": " << fixed << setprecision(2) << (time_per_step * 1000.0) << ",\n";
    out << "    \"final_loss\": " << fixed << setprecision(6) << final_loss << "\n";
    out << "  }\n";
    out << "}\n";
    out.close();

    cout << "\nTENSOR GPT CPU benchmark complete.\n";
    cout << "Total time: " << fixed << setprecision(2) << elapsed << " s\n";
    cout << "Tokens/sec: " << fixed << setprecision(2) << tokens_per_sec << "\n";
    cout << "Steps/sec:  " << fixed << setprecision(2) << steps_per_sec << "\n";
    cout << "Final loss: " << fixed << setprecision(6) << final_loss << "\n";
    cout << "Results written to " << out_path << "\n";
    return 0;
}
