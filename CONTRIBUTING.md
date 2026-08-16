# Contributing to TorchEngine

Thanks for your interest in contributing.

## Scope

TorchEngine is a learning- and systems-focused C++17/CUDA deep learning framework. Contributions should prioritize:

- Correctness and reproducibility over headline performance claims
- Clear separation between **stable** APIs and **experimental** CUDA work
- Measurable changes with documented methodology

## Build

### Configure

```bash
cmake -S . -B build
```

### Build

```bash
cmake --build build -j$(nproc)
```

CUDA support is optional and enabled automatically when a CUDA toolkit is detected.

## Tests

```bash
cd build
ctest --output-on-failure
```

If CUDA is enabled, additional CUDA-specific test binaries may be built; see `README.md` for the current list.

## Workflow

1. Fork the repo
2. Create a feature branch
3. Make changes with focused commits
4. Add/extend tests where applicable
5. Open a PR describing what changed and why

## Expectations

- New features should include tests or a clear rationale when testing is impractical.
- Changes that affect numerical correctness should include a brief validation note (e.g., tolerance, dtype, device coverage).

## Performance and benchmarking changes

If your change includes performance claims:

- Use the repo benchmark methodology in `BENCHMARKING.md`.
- Include the exact command(s) you ran.
- Include hardware/software details (GPU, driver, CUDA version, compiler, build type).
- Report trial count and summary stats (median + standard deviation).
- State whether results use cuBLAS or a custom GEMM backend.
- Ensure correctness tests passed on the same build.

Do not post single-run results as headline conclusions.

## Good areas to contribute

- CUDA kernels and kernel correctness tests
- GEMM optimization and profiling harnesses
- Tensor operations (CPU/CUDA) and dispatch
- Autograd coverage (missing backward nodes)
- Mixed precision utilities (FP16/BF16, loss scaling)
- Memory management (allocators, reuse, pooling)
- Benchmarking and measurement tooling
- Documentation and diagrams