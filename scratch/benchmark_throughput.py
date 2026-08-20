import time
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from model.nanogpt import HyperTransformerLM
from hyperspace.dynamic_optimizer import DynamicWarmupAdamW
from train_scaled_production_engine import ScaledProductionDataStreamer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[PROBE] Device: {device} ({torch.cuda.get_device_name(0)})")

seq_len = 256
micro_batch_size = 2
accum_steps = 6
effective_batch_size = micro_batch_size * accum_steps # 12
tokens_per_step = effective_batch_size * seq_len # 3072 tokens

# Model configuration matching Flagship
model = HyperTransformerLM(
    vocab_size=50304,
    d_model=384,
    n_layers=4,
    n_heads=6,
    d_ff=768,
    d_hyper=2048,
    top_k=2,
    max_k=4,
    top_p=0.85,
    dynamic_k=True,
    spawn_threshold=1.0,
    max_experts=2,
    initial_experts=2,
    use_sparse_attn=True,
    foveal_window=128,
    num_landmarks=4,
    num_sinks=4,
    max_seq_len=576,
    dropout=0.0,
    use_bus=True
).to(device)

optimizer = DynamicWarmupAdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
scaler = torch.amp.GradScaler('cuda')

streamer = ScaledProductionDataStreamer(cache_dir="data/scaled_real_corpus", seq_len=seq_len, batch_size=micro_batch_size)

num_warmup = 15
num_benchmark_steps = 50

print(f"[PROBE] Tokens per step: {tokens_per_step} (Effective Batch={effective_batch_size}, SeqLen={seq_len})")
print(f"[PROBE] Running {num_warmup} warmup steps...")

model.train()
for step in range(num_warmup):
    optimizer.zero_grad(set_to_none=True)
    for _ in range(accum_steps):
        x, y = streamer.get_domain_batch("fineweb_edu", split="train")
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda'):
            logits, loss, _ = model(x, targets=y, allow_spawning=False, is_replay=False)
            loss = loss / accum_steps
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

torch.cuda.synchronize()
print(f"[PROBE] Benchmarking {num_benchmark_steps} steps...")
start_time = time.perf_counter()

for step in range(num_benchmark_steps):
    optimizer.zero_grad(set_to_none=True)
    for _ in range(accum_steps):
        x, y = streamer.get_domain_batch("fineweb_edu", split="train")
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda'):
            logits, loss, _ = model(x, targets=y, allow_spawning=False, is_replay=False)
            loss = loss / accum_steps
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

torch.cuda.synchronize()
end_time = time.perf_counter()

total_time = end_time - start_time
steps_per_sec = num_benchmark_steps / total_time
tok_per_sec = (num_benchmark_steps * tokens_per_step) / total_time
peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 * 1024)

print("\n" + "=" * 70)
print("  [EMPIRICAL HARDWARE THROUGHPUT BENCHMARK (RTX 3060)]")
print("=" * 70)
print(f"  Benchmark Steps:          {num_benchmark_steps}")
print(f"  Total Wall Time:          {total_time:.2f} seconds")
print(f"  Steps / Second:           {steps_per_sec:.2f} step/s")
print(f"  Throughput (Tokens/s):    {tok_per_sec:.1f} tok/s")
print(f"  Peak VRAM Allocated:      {peak_vram_mb:.1f} MiB")
print(f"  Peak VRAM Reserved:       {peak_reserved_mb:.1f} MiB")
print("=" * 70)
print("\n--- GROUNDED REALISTIC TIME ESTIMATES ON CURRENT CODEBASE ---")
for scale_name, total_tok in [("10 Million Tokens", 10_000_000), ("25 Million Tokens", 25_000_000), ("50 Million Tokens", 50_000_000), ("100 Million Tokens", 100_000_000)]:
    sec = total_tok / tok_per_sec
    hrs = sec / 3600
    print(f"  {scale_name:<20}: {hrs:5.2f} hours ({sec/60:6.1f} minutes)")
print("=" * 70)
