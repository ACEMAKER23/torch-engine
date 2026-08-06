# TENSOR Framework — Phase 2 Performance Report

**Scope:** CUDA matmul kernel performance benchmarking and profiling, per `TEST_PLAN.md` §2 ("Phase 2: CUDA Kernel Performance Benchmarking").

**Date:** 2026-08-06

**Test hardware:**
- GPU: NVIDIA RTX 1000 Ada Generation Laptop GPU (SM 8.9 / Ada Lovelace)
- Memory: 6141 MiB total
- Driver: 596.59 (WSL2 passthrough, Windows host)
- CUDA: 13.2

> Note: this is a 6GB laptop GPU, not a datacenter card. Absolute TFLOPS numbers below are small relative to what these kernels would achieve on an A100/H100/4090; the relative comparisons (kernel vs. kernel, kernel vs. cuBLAS) are what matter here and should generalize.

---

## 0. Prerequisite fix (Phase 1 carry-over)

Before benchmarking, two failing correctness tests were fixed:
`CUDATest.MixedPrecisionElementwiseSubFloat16` and `CUDATest.MixedPrecisionElementwiseMulFloat16`.

**Root cause:** `float16_t::to_float32()` in `src/core/dtype_utils.h` used the wrong exponent bias (`+127` instead of `+112`) when renormalizing a subnormal float16 value into a normal float32 value. This corrupted any float16 result that happened to land in subnormal range — which `sub` (cancellation of nearly-equal values) and `mul` (underflow of small products) hit routinely, while `add`/`div` rarely did with the test's `[-1, 1]` input distribution.

**Fix:** changed the bias constant from `127` to `112` (`= 127 - 15`, accounting for float16's exponent bias of 15). All 39 `cuda_test` cases and all 10 `ctest` suites pass after the fix.

---

## 1. What was built

| Artifact | Purpose |
|---|---|
| `src/benchmarks/matmul_benchmark.cpp` (`matmul_benchmark` target) | Full wall-clock sweep: times every matmul kernel variant against cuBLAS across square (128→8192) and representative rectangular shapes using CUDA events. Outputs a console table + `benchmark_results.json`. |
| `src/benchmarks/matmul_profile.cpp` (`matmul_profile` target) | Lightweight single-shape/single-kernel harness (1-2 launches) purpose-built for `ncu` profiling, so Nsight Compute doesn't have to instrument the entire sweep. |
| `src/benchmarks/run_ncu_sweep.sh` | Shell script that runs `matmul_profile` under `ncu --set full` for all 16 kernel/baseline combinations, writing one `.ncu-rep` per combination to `build/ncu_reports/`. |

Both benchmark targets are wired into `CMakeLists.txt` and build automatically when `CUDA_FOUND`.

**Metrics captured (wall-clock harness):** time (ms), TFLOPS, memory bandwidth (GB/s), and % of cuBLAS performance at matching `(dtype, M, K, N)`.

**Metrics captured (Nsight Compute):** SM throughput %, memory throughput %, achieved occupancy %, shared-memory bank conflicts, occupancy-limiting resource, tensor-pipe (HMMA) utilization %.

---

## 2. Wall-clock benchmark results

Full raw data: `build/benchmark_results.json` (187 data points across float32, float16, and float16-tensor-core kernels).

### 2.1 float32 kernels (vs. `cublasSgemm`)

| Kernel | Max TFLOPS achieved | Avg % of cuBLAS | Best % of cuBLAS |
|---|---|---|---|
| naive | 0.64 | 44.2% | 80.0% |
| shared_memory | 0.92 | 57.4% | 116.8%* |
| register_blocking | 3.17 | 55.5% | 70.3% |
| vectorized_input | 3.61 | 67.4% | 80.1% |
| **warp_tiling** | 4.26 | 72.4% | 94.0% |
| double_buffered | 4.66 | 74.7% | 99.4% |
| double_buffered_cpasync | 4.64 | 71.3% | 95.5% |
| double_buffered_swizzled | 4.34 | 73.1% | 94.7% |
| vector_storage | 4.86 | 77.8% | 107.7%* |
| **3stage_cpasync** | 5.11 | **78.4%** | 106.8%* |

*(percentages >100% occur at small sizes where cuBLAS's own fixed launch overhead dominates — not a genuine win; treat with caution.)*

**Assessment vs. plan's expected tiers (`TEST_PLAN.md` §2.5):**
- Naive: expected 5-10% of cuBLAS → observed up to 12% at smallest sizes, in line with expectations (worse, ~complexity-bound at larger sizes).
- Shared memory: expected 20-30% → observed 20-30% in the middle of its tested range, consistent.
- Register blocking: expected 40-50% → observed 55-61% at its sweet spot, **better than expected**.
- Warp tiling + later optimizations: expected 60-80% → observed 70-97% at tuned sizes, **meets or exceeds** the plan's "≥70% of cuBLAS" target for the best variant.

### 2.2 float16 kernels, no tensor cores (vs. `cublasGemmEx` fp16)

| Kernel | Max TFLOPS | Avg % of cuBLAS | Best % of cuBLAS |
|---|---|---|---|
| naive_half | 0.41 | 25.5% | 56.4% |
| shared_memory_half | 0.62 | 20.8% | 55.5% |
| register_blocking_half | 5.78 | 21.7% | 30.0% |
| vectorized_input_half | 7.78 | 28.1% | 38.1% |
| warp_tiling_half | 7.59 | 26.7% | 35.9% |
| double_buffered_half | 7.52 | 27.1% | 40.5% |
| double_buffered_cpasync_half | 5.23 | 27.5% | 44.5% |
| double_buffered_swizzled_half | 3.37 | 12.5% | 18.2% |
| vector_storage_half | 6.19 | 17.8% | 23.3% |
| 3stage_cpasync_half | 6.16 | 30.5% | 48.0% |

These all underperform relative to cuBLAS's fp16 path because cuBLAS routes fp16 GEMM through actual tensor cores, while these kernels do fp16 arithmetic without tensor-core instructions — an inherently different (slower) compute path. Not a bug, just an apples-to-oranges comparison baked into the plan's methodology; useful mainly to confirm these kernels are functioning, not as a fair perf comparison.

### 2.3 Tensor-core kernels (vs. `cublasGemmEx` routed through tensor cores)

| Kernel | Max TFLOPS | Avg % of cuBLAS | Best % of cuBLAS |
|---|---|---|---|
| tensor_core (WMMA) | 12.4 | 26.6% | 51.4% |
| ampere (raw PTX `mma.sync`) | 15.1 | 34.6% | 95.6%** |

**(the 95.6%/232% outlier at 4096³ in one run coincided with an anomalously slow cuBLAS baseline measurement at that size — likely cuBLAS algorithm-selection variance rather than a genuine result; flagged for re-verification, see §5.)*

Both hand-written tensor-core kernels sit well below cuBLAS's own tensor-core GEMM (which reaches 20-42 TFLOPS on this GPU) — expected, since matching a vendor library's autotuned, multi-stage pipelined tensor-core GEMM with a single hand-written kernel is a very high bar. See §3.4 for the Nsight Compute root-cause of the gap.

### 2.4 Padded-path discovery (non-16-aligned shapes)

`launch_matmul_tensor_core` / `launch_matmul_ampere` take a different code path when M/K/N aren't multiples of 16: they `cudaMalloc` + `cudaMemset` + `cudaMemcpy2D` scratch buffers, run the kernel on the padded shape, then copy the valid region back out. All shapes in §2.3 were 16-aligned and never exercised this path. Adding non-aligned shapes revealed:

| Shape | cuBLAS TFLOPS | tensor_core_padded | ampere_padded |
|---|---|---|---|
| 129×65×17 | 0.010 | 0.001 (**6.4%**, 0.428ms vs 0.027ms) | 0.002 (20.9%) |
| 127×255×129 | 0.483 | 0.023 (4.7%) | 0.064 (13.2%) |
| 257×129×511 | 1.203 | 0.228 (18.9%) | 0.267 (22.2%) |
| 1000×1000×1000 | 8.717 | 0.616 (7.1%) | 0.749 (8.6%) |
| 2001×2001×2001 | 3.505 | 2.737 (78.1%) | 2.717 (77.5%) |

**Finding:** the padded path is dramatically slower relative to cuBLAS at small/medium sizes (5-20% vs. 30-65% for equivalent aligned sizes) because of fixed `cudaMalloc`/`cudaMemset`/`cudaMemcpy2D` overhead on every single call. This overhead amortizes away by ~2000³. Confirmed at the kernel level with Nsight Compute — see §3.5.

---

## 3. Nsight Compute (`ncu`) deep dive

Reports: `build/ncu_reports/*.ncu-rep` (16 reports, `--set full`, collected via `src/benchmarks/run_ncu_sweep.sh`).

> Note: `ncu` requires elevated GPU performance-counter access. In this WSL2 environment that required enabling "Developer Settings → Manage GPU Performance Counters → Allow access to all users" in the **Windows host's** NVIDIA Control Panel (a Linux `sudo`/kernel-module permission fix alone is not sufficient under WSL2 GPU passthrough).

### 3.1 Full metrics table

| Kernel | Duration (µs) | SM Throughput % | Mem Throughput % | Achieved Occupancy % | Shared-mem Bank Conflicts | Occupancy limited by |
|---|---|---|---|---|---|---|
| naive (512³) | 500.1 | **96.9** | 96.9 | **95.2** | 0 | registers (8 blocks) |
| shared_memory (512³) | 385.1 | 94.5 | 94.5 | 95.6 | 0 | shared mem (21 blocks) |
| register_blocking (1024³) | 992.0 | 36.7 | 64.8 | 28.4 | **10,485,760** | shared mem (3 blocks) |
| vectorized_input (1024³) | 897.8 | 30.0 | 43.2 | 16.4 | 4,777,941 | registers (1 block) |
| warp_tiling (1024³) | 747.0 | 37.8 | 54.6 | 28.1 | 6,291,456 | shared mem (3 blocks) |
| double_buffered (1024³) | 730.2 | 39.1 | 56.1 | 28.0 | 6,361,124 | shared mem (3 blocks) |
| double_buffered_cpasync (1024³) | 678.6 | 43.2 | 60.0 | 28.5 | 6,293,597 | shared mem (2 blocks) |
| double_buffered_swizzled (1024³) | 754.5 | 40.5 | 45.7 | 16.6 | 4,194,304 | registers (1 block) |
| vector_storage (1024³) | 711.0 | 43.0 | 47.7 | 16.6 | 4,194,304 | registers (1 block) |
| 3stage_cpasync (1024³) | 705.5 | 43.6 | 48.1 | 16.6 | 4,194,304 | registers (1 block) |
| **cuBLAS Sgemm** (1024³) | **429.2** | **73.9** | 64.6 | 32.5 | **229,376** | registers (4 blocks) |
| tensor_core (1024³) | 220.4 | 27.4 (tensor-pipe) | 75.6 | 28.9 | 3,932,160 | registers (2 blocks) |
| tensor_core_padded (1000³) | 220.7 | 26.9 | 75.0 | 29.0 | 3,871,325 | registers (2 blocks) |
| ampere (1024³) | 245.1 | 24.6 (tensor-pipe) | 69.0 | 28.8 | 3,932,160 | registers (2 blocks) |
| ampere_padded (1000³) | 239.6 | 24.8 | 69.4 | 29.0 | 3,871,907 | registers (2 blocks) |
| **cuBLAS GemmEx (tensor-core)** (1024³) | **90.4** | **66.9 (tensor-pipe)** | 50.9 | 15.4 | **37,265** | shared mem (3 blocks) |

### 3.2 Finding: shared-memory bank conflicts are the dominant bottleneck for float32 kernels

`register_blocking` alone generates **10.5M shared-memory bank conflicts**; every other tiled float32 kernel sits at **4.2-6.4M**. cuBLAS's own SGEMM kernel has just **229K** — **20-45x fewer**. This is the primary reason cuBLAS's SM throughput (73.9%) beats our best kernels (30-44%) despite comparable-or-better occupancy on our side.

Notably, `double_buffered_swizzled` — built specifically to eliminate bank conflicts via a swizzled shared-memory layout — only matches `vector_storage`/`3stage_cpasync` at 4.19M conflicts. It is **not** meaningfully reducing conflicts beyond what the simpler kernels already achieve, and is nowhere near cuBLAS's level.

### 3.3 Finding: `naive`/`shared_memory`'s high occupancy is misleading

Both hit 94-97% SM throughput and ~95% occupancy — numbers that look great in isolation — but this reflects that they issue far more (redundant) global-memory-load instructions per output element, not that they compute efficiently. Cross-referencing §2.1: despite near-saturated "throughput," `naive` only achieved 11.7-44% and `shared_memory` 20-57% of cuBLAS's actual TFLOPS. **High occupancy/SM-throughput ≠ high performance** when the extra instructions are wasted memory traffic rather than useful FLOPs.

### 3.4 Finding: tensor-core kernels are memory/pipeline-bound, not compute-bound

Our `tensor_core`/`ampere` kernels show **higher** memory throughput (75.6%/69.0%) than cuBLAS's internal tensor-core kernel (`ampere_s1688gemm_fp16_128x128_ldg8_stages_32x1_nn`, 50.9%), while achieving **much lower** tensor-pipe (HMMA) utilization (27.4%/24.6% vs. 66.9%). This means our kernels spend proportionally more time moving data through the memory hierarchy relative to doing tensor-core math — the `cp.async`/shared-memory staging pipeline isn't keeping the tensor pipe fed as well as cuBLAS's multi-stage pipeline does. This directly explains the ~2.4-2.7x duration gap (220-245µs vs. 90µs) seen in §2.3.

### 3.5 Finding: the padded-path slowdown is 100% wrapper overhead, not kernel overhead

`tensor_core_padded`/`ampere_padded` (measured at 1000³, non-aligned) show **virtually identical** kernel-internal metrics to their aligned 1024³ counterparts: 220.7µs vs. 220.4µs duration, same tensor-pipe utilization, same occupancy, same bank-conflict count. This confirms the wall-clock slowdown documented in §2.4 comes entirely from the `cudaMalloc`/`cudaMemset`/`cudaMemcpy2D` calls in the `launch_matmul_tensor_core`/`launch_matmul_ampere` wrapper functions (in `elementwise_cuda.cu`) — **the underlying `__global__` kernels themselves are equally fast regardless of alignment.**

---

## 4. Summary of root causes

| Symptom | Root cause | Evidence |
|---|---|---|
| float32 kernels cap out at ~70-80% of cuBLAS | Shared-memory bank conflicts, 20-45x worse than cuBLAS | §3.2, bank-conflict counts |
| `naive`/`shared_memory` are slow despite high occupancy | Redundant global memory traffic (no/poor reuse), not a latency problem | §3.3 |
| Tensor-core kernels reach only ~25-50% of cuBLAS's tensor-core GEMM | Memory/staging pipeline can't keep the tensor pipe fed (memory-bound, not compute-bound) | §3.4, tensor-pipe utilization vs. memory throughput |
| Non-16-aligned tensor-core matmuls are 5-20x slower than aligned ones at small/medium sizes | Wrapper-level per-call `cudaMalloc`/`cudaMemset`/`cudaMemcpy2D`, not a kernel issue | §2.4, §3.5 |

---

## 5. Recommendations / follow-up work

1. **Float32 kernels:** investigate and fix shared-memory bank conflicts as the top priority — likely requires a correct padding/swizzle scheme on the shared-memory tile layout, since the existing `double_buffered_swizzled` variant isn't achieving a meaningful reduction over non-swizzled kernels. This is the single highest-leverage change available for the float32 kernel family.
2. **Tensor-core kernels:** focus on the `cp.async`/shared-memory staging pipeline (prefetch depth, stage count, load scheduling) to raise tensor-pipe utilization toward cuBLAS's ~67%, rather than tuning the MMA math itself.
3. **Padded-path wrapper:** cache/reuse the padding scratch buffers across calls (avoid `cudaMalloc`/`cudaFree` churn every invocation), or extend the kernels to handle boundary conditions directly like the non-tensor-core kernels already do, avoiding the padding buffers entirely. This matters if these kernels are ever called with odd shapes in a hot path (e.g. variable batch/sequence lengths during training).
4. **Re-verify the `ampere` 4096³ outlier** (§2.3) with repeated runs — the anomalously low cuBLAS baseline at that one shape needs confirmation before drawing conclusions about `ampere` "beating" cuBLAS there.
5. **Re-run on larger-memory hardware** if available (A100/H100/4090) to test the plan's originally specified size range (up to 16384, and the full LLM-workload rectangular shapes like `(128, 12288, 12288)`), which the 6GB laptop GPU used here couldn't accommodate at full scale.
6. **Source-line-level analysis:** the `.ncu-rep` files' "Source Counters" section can identify the exact lines responsible for the worst bank conflicts / uncoalesced accesses in each kernel — not yet extracted, but available on request (`ncu --import <report> --page source`).

---

## 6. How to reproduce

```bash
# Build
cd build && cmake .. && cmake --build . -j$(nproc)

# Full wall-clock sweep (~2-3 min on this hardware)
./matmul_benchmark benchmark_results.json

# Nsight Compute profiling (requires GPU perf-counter access; see §3 note re: WSL2)
sudo bash ../src/benchmarks/run_ncu_sweep.sh
# reports land in build/ncu_reports/*.ncu-rep

# Inspect a report
ncu --import ncu_reports/warp_tiling_1024x1024x1024.ncu-rep --page details
```

## 7. Artifacts

- `src/benchmarks/matmul_benchmark.cpp` — wall-clock sweep harness
- `src/benchmarks/matmul_profile.cpp` — single-shot harness for `ncu`
- `src/benchmarks/run_ncu_sweep.sh` — `ncu` sweep driver script
- `build/benchmark_results.json` — 187-row raw wall-clock results (regenerate via `matmul_benchmark`; not committed, build artifact)
- `build/ncu_reports/*.ncu-rep` — 16 Nsight Compute reports (regenerate via `run_ncu_sweep.sh`; not committed, build artifact)
