# Test Plan

This document describes how TorchEngine is validated for correctness and how performance changes should be measured.

## 1. Correctness (required)

### 1.1 Build configurations

At minimum, validate one of:

- CPU-only build (no CUDA toolchain detected)
- CUDA-enabled build (CUDA toolkit + cuBLAS available)

### 1.2 Unit/integration tests

From a fresh build directory:

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
cd build
ctest --output-on-failure
```

Notes:

- The CTest suite covers core tensor ops, autograd, NN modules, losses, optimizers, and an end-to-end training sanity check.
- When CUDA is enabled, additional CUDA-specific test executables may be built. Some of these may not be registered in CTest yet and can be run directly.

### 1.3 Numerical expectations

TorchEngine is not a reference implementation. For many ops (especially CUDA kernels), correctness is validated with:

- Relative/absolute tolerances appropriate for dtype
- Shape/layout coverage for the supported paths
- Cross-checks against a known-correct CPU implementation where practical

If you change numerics (e.g., fused kernels, mixed precision, fast math), explicitly document tolerance changes and add targeted tests.

## 2. Performance (optional; requires methodology)

Performance claims must be backed by reproducible commands and multiple trials. Follow the methodology in `BENCHMARKING.md`.

### 2.1 End-to-end GPT throughput

Use the Python suite runner:

```bash
python3 scripts/run_benchmark_suite.py --build-dir build --data data/tinyshakespeare.txt --output-dir benchmark_outputs --trials 7
```

### 2.2 Isolated matmul benchmark

The suite runs `matmul_benchmark` and aggregates per-trial JSON outputs.

## Appendix: Benchmark data format

Benchmark JSON/CSV outputs are generated under `benchmark_outputs/` (gitignored by default). The format is intended to be stable enough to:

- audit per-trial results
- compute aggregates (median/stddev)
- track regressions across local runs

If you change output fields, keep backward compatibility when practical or document the change in `BENCHMARKING.md`.