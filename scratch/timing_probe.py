"""Measure REAL current throughput on this GPU, current codebase, no hypothetical upgrades."""
import time, sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn.functional as F
from model.nanogpt import HyperTransformerLM

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# Flagship 2-expert core config, exactly as used throughout this conversation's control script
model = HyperTransformerLM(
    vocab_size=50304, d_model=384, n_layers=4, n_heads=6, d_ff=768, d_hyper=2048,
    top_k=2, max_k=4, top_p=0.85, dynamic_k=True, spawn_threshold=1.0,
    max_experts=2, initial_experts=2, use_sparse_attn=True, foveal_window=128,
    num_landmarks=4, num_sinks=4, max_seq_len=576, dropout=0.0, use_bus=True,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
scaler = torch.amp.GradScaler('cuda')

seq_len = 256
micro_batch = 12
accum_steps = 3
tokens_per_step = seq_len * micro_batch * accum_steps

def run_n_steps(n):
    model.train()
    t0 = time.perf_counter()
    for _ in range(n):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum_steps):
            x = torch.randint(0, 50304, (micro_batch, seq_len), device=device)
            y = torch.randint(0, 50304, (micro_batch, seq_len), device=device)
            with torch.amp.autocast('cuda'):
                logits, loss, _ = model(x, targets=y, allow_spawning=True)
                loss = loss / accum_steps
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    return time.perf_counter() - t0

# Warmup (excluded from timing -- covers spawning, cudnn autotune, allocator warmup)
run_n_steps(5)

# Timed measurement
elapsed = run_n_steps(20)
toks_per_sec = (tokens_per_step * 20) / elapsed
print(f"\nConfig: seq_len={seq_len}, micro_batch={micro_batch}, accum_steps={accum_steps}, tokens/step={tokens_per_step}")
print(f"20 steps in {elapsed:.2f}s")
print(f"REAL measured throughput: {toks_per_sec:.0f} tokens/sec")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB")

for horizon, label in [(50_000_000, "50M"), (100_000_000, "100M"), (250_000_000, "250M"), (500_000_000, "500M")]:
    hours = horizon / toks_per_sec / 3600
    print(f"{label} tokens -> {hours:.1f} hours at this measured rate")
