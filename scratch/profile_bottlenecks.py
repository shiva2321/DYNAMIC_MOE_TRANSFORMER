import time
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.nanogpt import HyperTransformerLM

def benchmark_moe_forward_backward(model, x, y, steps=20):
    # Warmup
    for _ in range(5):
        logits, loss, _ = model(x, targets=y, allow_spawning=False)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(steps):
        logits, loss, _ = model(x, targets=y, allow_spawning=False)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    total_tokens = steps * x.shape[0] * x.shape[1]
    tok_s = total_tokens / (t1 - t0)
    step_time_ms = ((t1 - t0) / steps) * 1000.0
    return tok_s, step_time_ms

def run_diagnostic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 90)
    print("  [MOE BOTTLENECK PROFILER & DISPATCH OPTIMIZER]")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print("=" * 90)

    b, s = 2, 256 # micro batch
    x = torch.randint(0, 50304, (b, s), device=device)
    y = torch.randint(0, 50304, (b, s), device=device)

    for num_exp in [2, 8, 16, 32]:
        print(f"\n--- Testing Topology: {num_exp} Experts per Layer ---")
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
            max_experts=32,
            initial_experts=num_exp,
            use_sparse_attn=True,
            foveal_window=128,
            num_landmarks=4,
            num_sinks=4,
            max_seq_len=576,
            dropout=0.0,
            use_bus=True
        ).to(device)

        tok_s, ms = benchmark_moe_forward_backward(model, x, y, steps=20)
        print(f"  Result ({num_exp} experts): {tok_s:.1f} tokens/sec | {ms:.2f} ms/step")

if __name__ == "__main__":
    run_diagnostic()
