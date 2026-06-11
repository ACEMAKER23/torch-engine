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

