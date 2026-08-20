"""Directly measure Tier 1 (d=384, L=4) at the ACTUAL proposed max_experts=32, fully spawned."""
import time, sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
from model.nanogpt import HyperTransformerLM
from hyperspace.vsa import ComplexPhasorVSA

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

model = HyperTransformerLM(
    vocab_size=50304, d_model=384, n_layers=4, n_heads=6, d_ff=768, d_hyper=2048,
    top_k=2, max_k=4, top_p=0.85, dynamic_k=True, spawn_threshold=1.0,
    max_experts=32, initial_experts=32, use_sparse_attn=True, foveal_window=128,
    num_landmarks=4, num_sinks=4, max_seq_len=576, dropout=0.0, use_bus=True,
).to(device)
print(f"Experts per layer: {[b.hyper_moe.num_experts for b in model.blocks]}")
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
scaler = torch.amp.GradScaler('cuda')

seq_len, micro_batch, accum_steps = 256, 12, 3
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
                logits, loss, _ = model(x, targets=y, allow_spawning=False)
                loss = loss / accum_steps
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    return time.perf_counter() - t0

run_n_steps(5)  # warmup
elapsed = run_n_steps(20)
toks_per_sec = (tokens_per_step * 20) / elapsed
print(f"\n20 steps in {elapsed:.2f}s")
print(f"REAL measured throughput @ 32 experts fully spawned: {toks_per_sec:.0f} tokens/sec")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB")
for total_steps, label in [(3600, "3,600-step run (1 seed)")]:
    print(f"{label} -> {(total_steps*tokens_per_step)/toks_per_sec/60:.1f} minutes")
