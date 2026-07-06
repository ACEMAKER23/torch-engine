# LLM Training Framework Roadmap

## Project Vision

Build a high-performance machine learning framework in C++ and CUDA focused specifically on Large Language Model (LLM) training.

The primary goal is educational:

- Understand every major component of modern deep learning frameworks.
- Understand how transformers and LLM training work internally.
- Understand GPU programming and performance optimization.

The secondary goal is performance:

- Create a framework capable of competing with or exceeding PyTorch performance for LLM training workloads.
- Prioritize LLM-specific optimizations over general-purpose flexibility.

---

# Core Philosophy

## Learning First

The project is designed to teach the underlying concepts behind:

- Tensor systems
- Automatic differentiation
- Neural network layers
- CUDA programming
- Transformer architectures
- Distributed training
- Performance optimization

We will implement components from scratch instead of relying heavily on external frameworks.

---

## Build in Layers

We will not optimize prematurely.

Each phase should:

1. Work correctly
2. Be thoroughly understood
3. Be benchmarked
4. Be optimized

Correctness always comes before optimization.

---

## Performance-Oriented Design

Even in early stages we should make architectural decisions that support future optimization.

Examples:

- Separate Tensor from Storage
- Support tensor views
- Avoid unnecessary memory copies
- Design with CUDA in mind
- Enable future kernel fusion

---

# High Level Architecture

```text
Application Layer
│
├── GPT Models
├── Transformer Blocks
└── Training Loops

Neural Network Layer
│
├── Linear
├── Embedding
├── LayerNorm
├── Attention
└── Activations

Autograd Engine
│
├── Computation Graph
├── GradFn Nodes
└── Backward Pass

Tensor Layer
│
├── Tensor
├── TensorImpl
├── Storage
└── Shape/Stride Logic

Backend Layer
│
├── CPU Kernels
├── CUDA Kernels
├── Memory Pools
└── Streams
```

---

# Phase 0: Architecture Design

## Goal

Design the framework before implementation.

## Deliverables

- Roadmap
- Directory structure
- Tensor architecture
- Coding standards

## Why

Changing core architecture later is extremely expensive.

---

# Phase 1: Tensor System

## Goal

Build a production-quality tensor abstraction.

No autograd.

No CUDA.

CPU only.

---

## Tensor Architecture

```text
Tensor
    │
    ▼
TensorImpl
    │
    ▼
Storage
```

---

## Storage

Responsible for raw memory ownership.

Contains:

```cpp
void* data;
size_t bytes;
Device device;
DType dtype;
```

Responsibilities:

- Memory allocation
- Memory deallocation
- Device tracking

---

## TensorImpl

Responsible for metadata.

Contains:

```cpp
shape
strides
offset
storage
```

Responsibilities:

- Shape management
- View support
- Slicing support

---

## Tensor

User-facing API.

Example:

```cpp
Tensor a({2, 3});
Tensor b = a.view({3, 2});
```

Responsibilities:

- Public interface
- Reference semantics

---

## Features

### Shapes

```cpp
tensor.shape()
```

### Strides

```cpp
tensor.strides()
```

### Number of Elements

```cpp
tensor.numel()
```

### Views

```cpp
tensor.view(...)
```

### Slices

```cpp
tensor.slice(...)
```

---

## Why This Design

Allows:

- Zero-copy views
- Tensor slicing
- Efficient memory sharing
- Future CUDA compatibility

This is similar to PyTorch's internal design.

---

# Phase 2: CPU Tensor Operations

## Goal

Build a numerical backend.

Still no gradients.

---

## Operations

### Elementwise

```cpp
add
sub
mul
div
```

### Reductions

```cpp
sum
mean
max
```

### Matrix Operations

```cpp
matmul
transpose
```

---

## Why Before Autograd

We want to separate:

```text
Math Engine
```

from

```text
Gradient Engine
```

This makes debugging significantly easier.

---

# Phase 3: Automatic Differentiation

## Goal

Implement reverse-mode autodiff.

Equivalent to PyTorch autograd.

---

## Tensor Additions

```cpp
requires_grad
grad
grad_fn
```

---

## Graph Nodes

Base class:

```cpp
class GradFn
{
public:
    virtual std::vector<Tensor>
    backward(const Tensor& grad) = 0;
};
```

---

## Backward Operations

Implement:

```cpp
AddBackward
SubBackward
MulBackward
MatMulBackward
```

---

## Why Node Classes

Alternative:

```cpp
std::function
```

We will avoid this.

Reasons:

- Extra allocations
- Harder to optimize
- Worse cache behavior

Node classes are closer to PyTorch.

---

# Phase 4: Backward Engine

## Goal

Implement:

```cpp
loss.backward();
```

---

## Components

### Graph Traversal

Topological sort

### Gradient Propagation

Reverse graph execution

### Gradient Accumulation

```cpp
param.grad += incoming_grad;
```

---

## Deliverable

Train a simple neural network entirely on CPU.

---

# Phase 5: CUDA Infrastructure

## Goal

Add GPU support.

---

## Device Abstraction

```cpp
enum class Device
{
    CPU,
    CUDA
};
```

---

## Storage Update

```cpp
void* ptr;
Device device;
```

---

## Why

GPU memory cannot be treated like normal CPU memory.

We need a generic abstraction.

---

# Phase 6: CUDA Kernels

## Goal

Execute tensor operations on GPU.

---

## Initial Kernels

### Elementwise

```cpp
add
sub
mul
div
```

### Reduction

```cpp
sum
```

### Matrix Multiply

```cpp
matmul
```

---

## Learning Objectives

Understand:

- Thread hierarchy
- Memory coalescing
- Shared memory
- Occupancy
- Synchronization

---

# Phase 7: Neural Network Components

## Goal

Build transformer building blocks.

---

## Layers

### Linear

```cpp
y = xW + b
```

### Embedding

```cpp
lookup table
```

### LayerNorm

### GELU

### Dropout

---

## Deliverable

Train an MLP using framework primitives.

---

# Phase 8: Transformer Architecture

## Goal

Build GPT-style transformers.

---

## Components

### Multi-Head Attention

### Feed Forward Network

### Residual Connections

### LayerNorm

### Positional Embeddings

---

## Deliverable

Train a tiny GPT model.

---

# Phase 9: Optimizers

## Goal

Implement training algorithms.

---

## Optimizers

### SGD

### Adam

### AdamW

---

## State Tensors

Store:

```cpp
m
v
```

as tensors.

---

## Deliverable

Stable transformer training.

---

# Phase 10: Mixed Precision Training

## Goal

Support modern training formats.

---

## DTypes

```cpp
float32
float16
bfloat16
```

---

## Features

### Master Weights

### Loss Scaling

### Mixed Precision Kernels

---

## Deliverable

Reduced memory usage and faster training.

---

# Phase 11: LLM-Specific Optimization

## Goal

Compete with PyTorch.

---

## Memory Pool

Avoid:

```cpp
cudaMalloc
cudaFree
```

during training.

---

## Kernel Fusion

Combine operations:

```cpp
bias + gelu
```

into a single kernel.

---

## Flash Attention

Implement:

```cpp
QK^T
Softmax
AV
```

without materializing large intermediate matrices.

---

## CUDA Graphs

Reduce launch overhead.

---

## Tensor Core Optimization

Target:

```cpp
FP16
BF16
```

Tensor Core execution.

---

# Phase 12: Distributed Training

## Goal

Scale beyond a single GPU.

---

## Techniques

### Data Parallelism

### Tensor Parallelism

### Pipeline Parallelism

---

## Deliverable

Multi-GPU transformer training.

---

# Long-Term Vision

The final framework should:

- Train GPT-style models
- Support CUDA acceleration
- Support mixed precision
- Support Flash Attention
- Support distributed training
- Be optimized specifically for LLM workloads

Potential advantages over PyTorch:

- Reduced abstraction overhead
- LLM-focused execution model
- Specialized kernels
- Static graph opportunities
- Better memory planning

---

# Development Rules

## Rule 1

Understand every line before writing the next layer.

---

## Rule 2

Benchmark every major feature.

---

## Rule 3

Do not optimize code that is not measured.

---

## Rule 4

Correctness before performance.

---

## Rule 5

Keep APIs simple.

---

## Rule 6

Every abstraction must justify its runtime cost.

---

# Current Status

## Completed

- **Phase 1 — Tensor System**: `Tensor` / `TensorImpl` / `Storage` separation, shapes,
  strides, offsets, zero-copy views, slicing, transpose, broadcasting. (58 tests passing)
- **Phase 2 — CPU Tensor Operations**: elementwise `+ - * /`, unary negation, `matmul`,
  broadcasting-aware elementwise ops, `Float32` / `Int32` / `Int64` dtypes.
- **Phase 3 — Automatic Differentiation**: `requires_grad`, `grad`, `grad_fn`; `GradFn`
  base class with `AddBackward`, `SubBackward`, `MulBackward`, `DivBackward`,
  `MatMulBackward` node classes; operators build the computation graph.
- **Phase 4 — Backward Engine**: `loss.backward()`, reverse graph traversal, gradient
  accumulation into leaf tensors. PyTorch-style architecture (autograd metadata in `TensorImpl`).
  (43 autograd tests passing, including chained ops)
- **Phase 5 — CPU Autograd & Broadcasting**:
  - Broadcasting gradient reduction: Implemented `Tensor::sum(int64_t dim)` and
    `reduce_gradient` helper. Updated all backward functions (Add, Sub, Mul, Div, MatMul) to
    reduce gradients over broadcast dimensions.
  - Activation function gradients: Implemented `ReluBackward`, `GeluBackward`,
    `SigmoidBackward` with correct gradient formulas.
  - Data type optimizations: Eliminated memory allocations, template-based dispatch,
    reduced code duplication.
  - Test coverage: 43 autograd tests passing (including complex tests, broadcasting
    reduction tests, activation tests, edge case tests).
- **Phase 7 — Neural Network Components**: All major components implemented on CPU
  - `Module` base class with `forward()`, `parameters()`, `zero_grad()` interface
  - `Linear` layer: Fully connected layer with weight/bias parameters, Xavier initialization
  - `Embedding` layer: Token embedding lookup with learnable weights
  - `LayerNorm`: Layer normalization with learnable weight/bias
  - `Dropout`: Random neuron dropout with training/inference modes
  - `Sequential`: Container for stacking layers
  - `Residual`: Residual connection wrapper
  - Test coverage: 51 tests passing (10 linear + 22 nn_layers + 19 attention)
- **Phase 8 — Transformer Architecture (Partial)**:
  - `ScaledDot`: Scaled dot-product attention with softmax
  - `MultiHeadAttention`: Multi-head attention with Q/K/V projections
  - `PositionalEmbedding`: Sinusoidal positional embeddings
  - Missing: Feed Forward Network, complete Transformer block, GPT model

## Key Architectural Decision (Phase 4)

Autograd metadata (`requires_grad`, `grad`, `grad_fn`) lives inside the **shared**
`TensorImpl`, not in the lightweight `Tensor` handle. This is the PyTorch
(`Variable` / `TensorImpl` + `AutogradMeta`) model. It is what makes
`loss.backward()` correct for chained operations — see `tensor.md` for the full
explanation of the bug it fixes and why.

## Key Implementation (Phase 5)

Broadcasting gradient reduction: When a tensor is broadcast during forward pass
(e.g., shape `[1,2]` → `[2,2]`), its gradient during backward pass must be summed
over the broadcast dimensions to match the original shape. Implemented via
`Tensor::sum(int64_t dim)` and `reduce_gradient` helper applied to all binary
operation backward functions.

## Not Yet Started

- **Phase 5 — CUDA Infrastructure**: Device abstraction, GPU memory management (CUDA kernels exist but not integrated)
- **Phase 6 — CUDA Kernels**: GPU elementwise ops, reductions, matmul (kernels exist but not integrated)
- **Phase 8 — Transformer Architecture (Complete)**: Feed Forward Network, Transformer block, GPT model
- **Phase 9 — Optimizers**: SGD, Adam, AdamW
- **Phase 10 — Mixed Precision Training**: FP16/BF16 support
- **Phase 11 — LLM-Specific Optimization**: Memory pools, kernel fusion, Flash Attention, CUDA graphs
- **Phase 12 — Distributed Training**: Data parallelism, tensor parallelism, pipeline parallelism

## In Progress / Next

Decision point: Choose next phase
- **Option A**: Phase 5-6 (CUDA Infrastructure & Kernels) - Integrate existing CUDA kernels
- **Option B**: Phase 8 (Complete Transformer Architecture) - Build Feed Forward Network and GPT model
- **Option C**: Phase 9 (Optimizers) - Implement SGD, Adam, AdamW for training

## Building & Testing

```bash
cmake -S . -B build
cmake --build build
./build/tensor_test       # tensor system + ops (58 tests)
./build/autograd_test     # autograd + backward engine (43 tests)
./build/linear_test       # linear layer tests (10 tests)
./build/nn_layers_test    # neural network layers (22 tests)
./build/attention_test    # attention mechanisms (19 tests)
```

## Future Optimizations

The following optimization opportunities have been identified but deferred for future work:

1. **SIMD Vectorization for Element-Wise Operations** (MEDIUM)
   - Current implementation uses scalar element-wise loops
   - Could add SIMD intrinsics (x86 AVX, ARM NEON) for 2-4x speedup on large tensors
   - Could use compiler auto-vectorization hints (`#pragma omp simd`)
   - Consider external libraries like xsimd or oneDNN

2. **In-Place Operations Where Safe** (LOW-MEDIUM)
   - Current implementation creates new tensors for intermediate results
   - Could use in-place operations when tensor has single reference
   - Requires reference counting to track tensor usage
   - Would reduce memory bandwidth and improve performance

3. **Broadcasting Gradient Reduction Efficiency** (LOW)
   - Current `reduce_gradient` sums dimensions one at a time using `Tensor::sum(dim)`
   - Each sum creates a new tensor allocation
   - Could implement multi-dimensional sum operation in single pass
   - Specialize for common broadcast patterns

4. **Kernel Fusion Opportunities** (MEDIUM)
   - Chain operations like `bias + gelu` into single kernel
   - Reduces memory traffic and kernel launch overhead
   - Particularly important for transformer blocks

## Building & Testing

```bash
cmake -S . -B build
cmake --build build
./build/tensor_test      # tensor system + ops (48 tests)
./build/autograd_test    # autograd + backward engine + broadcasting reduction (43 tests)
```
