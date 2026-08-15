#!/usr/bin/env python3
"""
PyTorch GPU benchmark matching TENSOR's GPU model configs exactly.

Part 1: Isolated matmul throughput (float32, same shapes TENSOR uses via cuBLAS)
Part 2: Full GPT training step throughput (same config as gpt_benchmark_gpu)

Run from repo root:
    python3 scripts/pytorch_gpu_benchmark.py data/tinyshakespeare.txt
"""
import json
import sys
import time
import torch
import torch.nn as nn

DEVICE = torch.device("cuda")

# ─── CUDA timing helpers ───────────────────────────────────────────────────────

def cuda_time(fn, warmup=5, iters=50):
    """Return mean ms for fn() over iters, after warmup."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


# ─── Part 1: Matmul benchmark (float32) ───────────────────────────────────────

def benchmark_matmul():
    print("=" * 60)
    print("Part 1: Isolated matmul (float32, PyTorch GPU via cuBLAS)")
    print("=" * 60)

    # Shapes drawn from a 2-layer, 4-head, embed=64, ff=256 GPT
    # with batch=4, seq=32 — matching TENSOR GPU medium config
    shapes = [
        # (M, K, N, label)
        (128,  64,  64,  "QKV proj      batch*seq x embed x embed"),
        (128,  64,  64,  "Attn out proj batch*seq x embed x embed"),
        (128,  64,  256, "FFN linear1   batch*seq x embed x ff"),
        (128, 256,  64,  "FFN linear2   batch*seq x ff    x embed"),
        (128,  64,  65,  "LM head       batch*seq x embed x vocab"),
        (1024, 64,  64,  "Large seq     1024 x embed x embed"),
        (4096, 256, 256, "Stress        4096 x 256   x 256"),
    ]

    results = []
    for M, K, N, label in shapes:
        a = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float32)

        ms = cuda_time(lambda: torch.mm(a, b))
        flops = 2.0 * M * K * N
        tflops = flops / (ms * 1e-3) / 1e12

        print(f"  {label}")
        print(f"    ({M}x{K}) @ ({K}x{N})  →  {ms:.3f} ms  {tflops:.3f} TFLOPS")
        results.append({"M": M, "K": K, "N": N, "label": label,
                        "time_ms": ms, "tflops": tflops})

    return results


# ─── Part 2: GPT training benchmark ───────────────────────────────────────────

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = [stoi[c] for c in text]
    return data, len(chars)


def get_batch(data, batch_size, seq_len):
    N = len(data)
    ix = torch.randint(0, N - seq_len - 1, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+seq_len],   dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+seq_len+1], dtype=torch.long) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, embed_dim, num_heads,
                 num_layers, ff_dim, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        b, t = x.size()
        pos = torch.arange(t, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=x.device)
        h = self.transformer(h, mask=mask, is_causal=True)
        h = self.ln_f(h)
        return self.head(h)


def benchmark_gpt(data, vocab_size, config, label):
    print(f"\n{'='*60}")
    print(f"Part 2 — {label}")
    print(f"  embed={config['embed_dim']}  heads={config['num_heads']}  "
          f"layers={config['num_layers']}  ff={config['ff_dim']}")
    print(f"  batch={config['batch_size']}  seq={config['seq_len']}  "
          f"steps={config['measured_steps']}")
    print("=" * 60)

    model = GPT(
        vocab_size,
        config["max_seq_len"],
        config["embed_dim"],
        config["num_heads"],
        config["num_layers"],
        config["ff_dim"],
    ).to(DEVICE)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    # Warmup
    for _ in range(config["warmup_steps"]):
        x, y = get_batch(data, config["batch_size"], config["seq_len"])
        optimizer.zero_grad()
        loss = criterion(model(x).view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Measure
    loss_curve = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(config["measured_steps"]):
        x, y = get_batch(data, config["batch_size"], config["seq_len"])
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        loss_curve.append(loss.item())  # .item() forces one sync per step — same as TENSOR
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_tok = config["measured_steps"] * config["batch_size"] * config["seq_len"]
    tok_per_s = total_tok / elapsed
    ms_per_step = elapsed / config["measured_steps"] * 1000

    print(f"  Total time : {elapsed:.3f} s")
    print(f"  Tokens/sec : {tok_per_s:.0f}")
    print(f"  ms/step    : {ms_per_step:.2f}")
    print(f"  Final loss : {loss_curve[-1]:.4f}")

    return {
        "label": label,
        "config": config,
        "total_time_s": round(elapsed, 4),
        "total_tokens": total_tok,
        "tokens_per_sec": round(tok_per_s, 2),
        "ms_per_step": round(ms_per_step, 3),
        "final_loss": round(loss_curve[-1], 6),
    }


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/tinyshakespeare.txt"
    data, vocab_size = load_data(data_path)

    matmul_results = benchmark_matmul()

    # Small config — matches TENSOR gpt_benchmark (CPU default, GPU small)
    small_cfg = dict(
        max_seq_len=128, embed_dim=64, num_heads=4, num_layers=2, ff_dim=256,
        batch_size=2, seq_len=16, warmup_steps=10, measured_steps=40,
        lr=0.001, weight_decay=0.01,
    )

    # Medium config — matches TENSOR gpt_benchmark_gpu (previous result)
    medium_cfg = dict(
        max_seq_len=128, embed_dim=128, num_heads=8, num_layers=3, ff_dim=512,
        batch_size=4, seq_len=32, warmup_steps=10, measured_steps=40,
        lr=0.001, weight_decay=0.01,
    )

    r_small  = benchmark_gpt(data, vocab_size, small_cfg,  "Small model  (TENSOR CPU default config)")
    r_medium = benchmark_gpt(data, vocab_size, medium_cfg, "Medium model (TENSOR GPU previous config)")

    out = {
        "framework": "PyTorch",
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "matmul_float32": matmul_results,
        "gpt_small": r_small,
        "gpt_medium": r_medium,
    }
    out_path = "pytorch_gpu_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
