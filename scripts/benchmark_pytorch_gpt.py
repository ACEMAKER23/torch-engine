#!/usr/bin/env python3
"""Consistent PyTorch GPT benchmark for CPU or CUDA.

This script is intentionally small and explicit so it can be compared with the
TENSOR GPT benchmark binaries. It uses the same model-size fields as the TENSOR
BenchConfig environment variables and writes one JSON result per invocation.
"""
import argparse
import json
import platform
import random
import time
from pathlib import Path

import torch
import torch.nn as nn


class TinyShakespeare:
    def __init__(self, path: Path, seed: int):
        text = path.read_text(encoding="utf-8")
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        self.data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        self.vocab_size = len(chars)
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(seed)

    def next_batch(self, batch_size: int, seq_len: int, device: torch.device):
        hi = self.data.numel() - seq_len - 1
        ix = torch.randint(0, hi, (batch_size,), generator=self.rng)
        x = torch.stack([self.data[i : i + seq_len] for i in ix])
        y = torch.stack([self.data[i + 1 : i + seq_len + 1] for i in ix])
        return x.to(device), y.to(device)


class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, embed_dim, num_heads, num_layers, ff_dim, dropout):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        batch, seq = x.shape
        pos = torch.arange(seq, device=x.device).unsqueeze(0).expand(batch, seq)
        h = self.token_embed(x) + self.pos_embed(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(seq, device=x.device)
        h = self.transformer(h, mask=mask, is_causal=True)
        h = self.ln_f(h)
        return self.head(h)


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tinyshakespeare.txt")
    ap.add_argument("--output", default="pytorch_gpt_result.json")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--ff-dim", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--measured-steps", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.001)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--torch-threads", type=int, default=0)
    args = ap.parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("PyTorch CUDA is not available")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    data = TinyShakespeare(Path(args.data), args.seed)
    model = GPT(
        data.vocab_size,
        args.max_seq_len,
        args.embed_dim,
        args.num_heads,
        args.num_layers,
        args.ff_dim,
        args.dropout,
    ).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    last_loss = None
    for _ in range(args.warmup_steps):
        x, y = data.next_batch(args.batch_size, args.seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits.reshape(-1, data.vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        last_loss = loss

    synchronize(device)
    start = time.perf_counter()
    for _ in range(args.measured_steps):
        x, y = data.next_batch(args.batch_size, args.seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits.reshape(-1, data.vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        last_loss = loss
    synchronize(device)
    elapsed = time.perf_counter() - start

    total_tokens = args.measured_steps * args.batch_size * args.seq_len
    result = {
        "framework": "PyTorch",
        "device": args.device,
        "hardware": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu_threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        },
        "model_config": {
            "vocab_size": data.vocab_size,
            "max_seq_len": args.max_seq_len,
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "ff_dim": args.ff_dim,
        },
        "training_config": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "seed": args.seed,
        },
        "metrics": {
            "timing_mode": "throughput",
            "total_time_s": elapsed,
            "total_tokens": total_tokens,
            "tokens_per_sec": total_tokens / elapsed,
            "steps_per_sec": args.measured_steps / elapsed,
            "time_per_step_ms": elapsed * 1000.0 / args.measured_steps,
            "final_loss": float(last_loss.detach().cpu()) if last_loss is not None else None,
        },
    }

    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()