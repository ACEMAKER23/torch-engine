#!/usr/bin/env python3
"""PyTorch CPU GPT benchmark matching TENSOR's small model config."""
import os
import sys
import time
import torch
import torch.nn as nn

# Match TENSOR small config by default; allow env overrides
MAX_SEQ_LEN = int(os.environ.get("PT_MAX_SEQ_LEN", "128"))
EMBED_DIM = int(os.environ.get("PT_EMBED_DIM", "64"))
NUM_HEADS = int(os.environ.get("PT_NUM_HEADS", "4"))
NUM_LAYERS = int(os.environ.get("PT_NUM_LAYERS", "2"))
FF_DIM = int(os.environ.get("PT_FF_DIM", "256"))
BATCH_SIZE = int(os.environ.get("PT_BATCH_SIZE", "2"))
SEQ_LEN = int(os.environ.get("PT_SEQ_LEN", "16"))
WARMUP_STEPS = int(os.environ.get("PT_WARMUP_STEPS", "10"))
MEASURED_STEPS = int(os.environ.get("PT_MEASURED_STEPS", "40"))
LR = 0.001
WEIGHT_DECAY = 0.01
DROPOUT = 0.0
DEVICE = torch.device("cpu")

def load_data(path):
    with open(path, "r") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = [stoi[ch] for ch in text]
    return data, stoi, itos, len(chars)

def get_batch(data, batch_size, seq_len):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+seq_len], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+1+seq_len], dtype=torch.long) for i in ix])
    return x, y

class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, embed_dim, num_heads, num_layers, ff_dim, dropout):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        b, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long).unsqueeze(0).expand(b, t)
        tok_emb = self.token_embed(x)
        pos_emb = self.pos_embed(pos)
        x = tok_emb + pos_emb
        # causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(t)
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        return self.head(x)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 pytorch_gpt_benchmark.py <data_path>")
        sys.exit(1)
    data_path = sys.argv[1]
    data, stoi, itos, vocab_size = load_data(data_path)

    model = GPT(vocab_size, MAX_SEQ_LEN, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, DROPOUT)
    model = model.to(DEVICE)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Warmup
    for _ in range(WARMUP_STEPS):
        x, y = get_batch(data, BATCH_SIZE, SEQ_LEN)
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

    # Benchmark
    start = time.perf_counter()
    for _ in range(MEASURED_STEPS):
        x, y = get_batch(data, BATCH_SIZE, SEQ_LEN)
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
    end = time.perf_counter()

    elapsed = end - start
    total_tokens = MEASURED_STEPS * BATCH_SIZE * SEQ_LEN
    tokens_per_sec = total_tokens / elapsed
    steps_per_sec = MEASURED_STEPS / elapsed
    ms_per_step = (elapsed / MEASURED_STEPS) * 1000.0

    print(f"\nPyTorch CPU GPT benchmark")
    print(f"Total time:    {elapsed:.2f} s")
    print(f"Tokens/sec:    {tokens_per_sec:.2f}")
    print(f"Steps/sec:     {steps_per_sec:.2f}")
    print(f"ms/step:       {ms_per_step:.2f}")

if __name__ == "__main__":
    main()
