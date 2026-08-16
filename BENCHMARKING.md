# Benchmarking Guide

Note: the repository is named **TorchEngine**, but some binaries, environment variables, and CMake identifiers still use the legacy project name TENSOR. This document uses TENSOR where it matches the current code/scripts.

This repository uses a consistent benchmark suite for two questions:

1. End-to-end GPT training throughput: TENSOR CPU vs PyTorch CPU, and TENSOR GPU vs PyTorch GPU, across multiple model sizes.
2. TENSOR GPU training throughput with GEMM routed through cuBLAS vs the custom CUDA GEMM path.
3. Isolated matrix multiplication throughput: TENSOR CUDA matmul kernels vs cuBLAS.

The suite runs repeated trials and reports mean, median, standard deviation, min, max, and raw trial values. Use the median for headline comparisons and the standard deviation to discuss variance.

## Prerequisites

Build the project first:

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

Install PyTorch in the same environment used to run the benchmarks. For CPU-only comparisons, a CPU PyTorch build is enough. For GPU comparisons, install a CUDA-enabled PyTorch build that can see the same GPU as TENSOR.

Check the TENSOR tests before benchmarking:

```bash
cd build
ctest --output-on-failure
cd ..
```

## Run The Full Suite

From the repository root:

```bash
python3 scripts/run_benchmark_suite.py \
  --build-dir build \
  --data data/tinyshakespeare.txt \
  --output-dir benchmark_outputs \
  --trials 7 \
  --sizes small,medium,large \
  --tensor-gemm-backends cublas,custom
```

Outputs:

```text
benchmark_outputs/gpt_summary.json
benchmark_outputs/gpt_summary.csv
benchmark_outputs/matmul_summary.json
benchmark_outputs/trials/*.json
```

The trial files are intentionally kept. They make it possible to audit outliers instead of only trusting an aggregate table.

## Model Sizes

The suite currently defines three GPT-style configurations in `scripts/run_benchmark_suite.py`:

| Size | Embed | Heads | Layers | FFN | Batch | Seq | Measured Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 64 | 4 | 2 | 256 | 2 | 16 | 40 |
| medium | 128 | 8 | 3 | 512 | 4 | 32 | 40 |
| large | 256 | 8 | 4 | 1024 | 4 | 64 | 20 |

These are intentionally modest so they can run on a laptop GPU. If you use a larger GPU, increase the model sizes and measured steps in the runner before producing a final report.

## What The GPT Suite Compares

For every selected size, the runner executes:

```text
TENSOR CPU
PyTorch CPU
TENSOR CUDA, GEMM backend = cuBLAS
TENSOR CUDA, GEMM backend = custom CUDA kernel where supported
PyTorch CUDA
```

Each case uses the same batch size, sequence length, embedding dimension, number of heads, number of layers, FFN dimension, learning rate, weight decay, dropout, warmup steps, measured steps, and dataset path.

`--tensor-gemm-backends` controls the TENSOR CUDA GEMM variants. The default is `cublas,custom`. `cublas` is the normal TENSOR path. `custom` sets `TENSOR_GEMM_BACKEND=custom`, which uses the custom CUDA GEMM for contiguous row-major FP32 matmul calls and falls back to cuBLAS for layouts the custom kernel does not support.

The TENSOR GPU benchmark defaults to throughput mode: it synchronizes before and after the measured region, not after every phase. This is the mode to use for headline TENSOR GPU vs PyTorch GPU comparisons.

For diagnostic phase timing, run the TENSOR GPU benchmark directly with:

```bash
TENSOR_MAX_SEQ_LEN=256 \
TENSOR_EMBED_DIM=256 \
TENSOR_NUM_HEADS=8 \
TENSOR_NUM_LAYERS=4 \
TENSOR_FF_DIM=1024 \
TENSOR_BATCH_SIZE=4 \
TENSOR_SEQ_LEN=64 \
TENSOR_WARMUP_STEPS=10 \
TENSOR_MEASURED_STEPS=20 \
TENSOR_GEMM_BACKEND=cublas \
TENSOR_PROFILE_PHASES=1 \
./build/gpt_benchmark_gpu data/tinyshakespeare.txt
```

Do not mix phase-profile timings with throughput timings in the same comparison table. To profile the custom GEMM variant, use the same command with `TENSOR_GEMM_BACKEND=custom`.

## Matmul Benchmark

The runner executes `build/matmul_benchmark` once per trial and aggregates the generated JSON files. This gives process-level trial statistics for each `(dtype, kernel, M, K, N)` row.

The matmul summary includes:

- `time_ms`: statistics over trial-level mean kernel time
- `tflops`: statistics over trial-level TFLOPS
- `pct_of_cublas`: statistics relative to the median cuBLAS result for the same dtype and shape
- `gpu_name`: reported CUDA device name

Run only the matmul benchmark:

```bash
python3 scripts/run_benchmark_suite.py \
  --skip-gpt \
  --trials 7 \
  --output-dir benchmark_outputs_matmul_only
```

## CPU-Only Runs

For machines without CUDA:

```bash
python3 scripts/run_benchmark_suite.py \
  --cpu-only \
  --skip-matmul \
  --trials 7 \
  --sizes small,medium \
  --output-dir benchmark_outputs_cpu
```

## Interpreting Results

Use this hierarchy for report claims:

1. Correctness tests pass on the same build.
2. Median throughput or median ms/step is the headline number.
3. Standard deviation is reported next to the median.
4. Hardware, driver, CUDA version, PyTorch version, model config, and trial count are included.
5. Raw trial files are retained for auditability.

Avoid direct comparisons across different GPUs, different PyTorch builds, different measured-step counts, or different model configs.

## Report Checklist

Before writing a report, verify:

- `ctest --output-on-failure` passes.
- TENSOR CPU and PyTorch CPU rows use the same model size.
- TENSOR GPU and PyTorch GPU rows use the same model size and same GPU.
- TENSOR CUDA rows clearly identify `gemm_backend` as `cublas` or `custom`.
- GPU headline timings use throughput mode, not `TENSOR_PROFILE_PHASES=1`.
- At least 7 trials were collected for each headline table.
- Median and standard deviation are reported together.
- Any skipped GPU rows are explicitly marked as skipped, not silently omitted.
- Matmul claims are separated from end-to-end GPT training claims.

## Files

```text
scripts/benchmark_pytorch_gpt.py   Single PyTorch CPU/CUDA GPT benchmark invocation.
scripts/run_benchmark_suite.py     Orchestrates repeated TENSOR/PyTorch and matmul trials.
src/benchmarks/gpt_benchmark.cpp   TENSOR CPU GPT benchmark.
src/benchmarks/gpt_benchmark_gpu.cpp TENSOR CUDA GPT benchmark.
src/benchmarks/matmul_benchmark.cpp TENSOR CUDA matmul vs cuBLAS sweep.
```
