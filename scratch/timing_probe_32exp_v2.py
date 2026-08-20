import time, sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
from model.nanogpt import HyperTransformerLM

device = torch.device("cuda")
model = HyperTransformerLM(
    vocab_size=50304, d_model=384, n_layers=4, n_heads=6, d_ff=768, d_hyper=2048,
    top_k=2, max_k=4, top_p=0.85, dynamic_k=True, spawn_threshold=1.0,
    max_experts=32, initial_experts=32, use_sparse_attn=True, foveal_window=128,
    num_landmarks=4, num_sinks=4, max_seq_len=576, dropout=0.0, use_bus=True,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
scaler = torch.amp.GradScaler('cuda')
seq_len, micro_batch, accum_steps = 256, 12, 3
tokens_per_step = seq_len * micro_batch * accum_steps

def run_n_steps(n, label):
    model.train()
    t0 = time.perf_counter()
    for i in range(n):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum_steps):
            x = torch.randint(0, 50304, (micro_batch, seq_len), device=device)
            y = torch.randint(0, 50304, (micro_batch, seq_len), device=device)
            with torch.amp.autocast('cuda'):
                logits, loss, _ = model(x, targets=y, allow_spawning=False)
                loss = loss / accum_steps
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        print(f"  [{label}] step {i+1}/{n} done at {time.perf_counter()-t0:.1f}s", flush=True)
    return time.perf_counter() - t0

print("2 warmup steps...")
run_n_steps(2, "warmup")
print("5 timed steps...")
elapsed = run_n_steps(5, "timed")
toks_per_sec = (tokens_per_step * 5) / elapsed
print(f"\nREAL measured throughput @ 32 experts: {toks_per_sec:.0f} tokens/sec ({elapsed:.1f}s for 5 steps)")
print(f"3,600-step run estimate: {(3600*tokens_per_step)/toks_per_sec/60:.1f} minutes")
