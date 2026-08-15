import json
import sys
import time
import torch
import torch.nn as nn


def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    data = [stoi[c] for c in text]
    return text, chars, stoi, data


class ShakespeareGPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, embed_dim, num_heads, num_layers, ff_dim, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_embedding', pe)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            dtype=torch.float32
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        # x: [batch, seq_len]
        b, t = x.size()
        tok_emb = self.token_embedding(x)  # [batch, seq_len, embed_dim]
        pos_emb = self.pos_embedding[:t, :].unsqueeze(0).expand(b, -1, -1)
        x = tok_emb + pos_emb
        x = self.transformer(x)
        x = self.final_norm(x)
        return self.lm_head(x)


def get_batch(data, batch_size, seq_len, device):
    N = len(data)
    starts = torch.randint(0, N - seq_len - 1, (batch_size,))
    inputs = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    for b in range(batch_size):
        s = starts[b].item()
        inputs[b] = torch.tensor(data[s:s+seq_len], dtype=torch.long, device=device)
        targets[b] = torch.tensor(data[s+1:s+seq_len+1], dtype=torch.long, device=device)
    return inputs, targets


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/tinyshakespeare.txt'
    device_name = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
    device = torch.device(device_name)

    _, chars, _, data = load_data(data_path)
    vocab_size = len(chars)

    config = {
        'vocab_size': vocab_size,
        'max_seq_len': 128,
        'embed_dim': 64,
        'num_heads': 4,
        'num_layers': 2,
        'ff_dim': 128,
        'batch_size': 4,
        'seq_len': 128,
        'warmup_steps': 10,
        'measured_steps': 40,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'dropout': 0.0
    }

    model = ShakespeareGPT(
        vocab_size,
        config['max_seq_len'],
        config['embed_dim'],
        config['num_heads'],
        config['num_layers'],
        config['ff_dim'],
        config['dropout']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    # Warmup
    for _ in range(config['warmup_steps']):
        inputs, targets = get_batch(data, config['batch_size'], config['seq_len'], device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

    # Benchmark
    loss_curve = []
    log_interval = max(1, config['measured_steps'] // 5)

    start = time.time()
    for s in range(config['measured_steps']):
        inputs, targets = get_batch(data, config['batch_size'], config['seq_len'], device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        loss_curve.append(loss.item())
        if (s + 1) % log_interval == 0 or s == 0:
            print(f"Step {s+1}/{config['measured_steps']}  loss = {loss.item():.4f}")
    end = time.time()

    elapsed = end - start
    total_tokens = config['measured_steps'] * config['batch_size'] * config['seq_len']
    tokens_per_sec = total_tokens / elapsed
    steps_per_sec = config['measured_steps'] / elapsed
    time_per_step_ms = (elapsed / config['measured_steps']) * 1000.0

    results = {
        'framework': 'PyTorch',
        'device': str(device),
        'model_config': {k: config[k] for k in ['vocab_size', 'max_seq_len', 'embed_dim', 'num_heads', 'num_layers', 'ff_dim']},
        'training_config': {k: config[k] for k in ['batch_size', 'seq_len', 'measured_steps', 'learning_rate', 'dropout']},
        'metrics': {
            'total_time_s': round(elapsed, 4),
            'total_tokens': total_tokens,
            'tokens_per_sec': round(tokens_per_sec, 2),
            'steps_per_sec': round(steps_per_sec, 2),
            'time_per_step_ms': round(time_per_step_ms, 2),
            'final_loss': round(loss_curve[-1], 6),
            'loss_curve': [round(x, 6) for x in loss_curve]
        }
    }

    out_path = 'pytorch_gpt_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nPyTorch GPT {device} benchmark complete.")
    print(f"Total time: {elapsed:.2f} s")
    print(f"Tokens/sec: {tokens_per_sec:.2f}")
    print(f"Steps/sec:  {steps_per_sec:.2f}")
    print(f"Final loss: {loss_curve[-1]:.6f}")
    print(f"Results written to {out_path}")


if __name__ == '__main__':
    main()
