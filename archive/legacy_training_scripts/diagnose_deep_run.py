import os
import sys
import math
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.nanogpt import HyperTransformerLM
from hyperspace.vsa import ComplexPhasorVSA
from hyperspace.dynamic_optimizer import DynamicWarmupAdamW
from train_deep_real_multidomain import RealBlendMemmapLoader

def diagnose():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_in = "experiments/checkpoints/hyperspace_real_multidomain_longcontext.pt"
    checkpoint = torch.load(ckpt_in, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    state_dict = checkpoint.get("model_state", checkpoint)

    vocab_size = config.get("vocab_size", 50304)
    d_model = config.get("d_model", 384)
    n_layers = config.get("n_layers", 6)
    n_heads = config.get("n_heads", 6)
    d_ff = config.get("d_ff", 768)
    d_hyper = config.get("d_hyper", 2048)

    per_layer_experts = []
    for l in range(n_layers):
        exp_keys = set()
        for k in state_dict.keys():
            if f"blocks.{l}.hyper_moe.experts." in k:
                exp_keys.add(int(k.split(".")[4]))
        per_layer_experts.append(max(2, len(exp_keys)))

    model = HyperTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        d_hyper=d_hyper,
        top_k=2,
        max_k=4,
        top_p=0.85,
        dynamic_k=True,
        spawn_threshold=0.35,
        max_experts=32,
        initial_experts=2,
        use_sparse_attn=True,
        foveal_window=128,
        num_landmarks=4,
        num_sinks=4,
        max_seq_len=1088,
        dropout=0.0,
    ).to(device)

    for l_idx, block in enumerate(model.blocks):
        needed = per_layer_experts[l_idx] - block.hyper_moe.num_experts
        for exp_i in range(needed):
            dummy_key = ComplexPhasorVSA.random_hyperspace_vector((1, d_hyper), device=device)
            block.hyper_moe._spawn_expert(dummy_key, label=f"restored_L{l_idx}_E{exp_i}")

    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd, strict=True)
    print("Checkpoint successfully restored!")

    loader = RealBlendMemmapLoader(cache_dir="data/real_blend_cache", seq_len=1024, batch_size=1)
    optimizer = DynamicWarmupAdamW(model.parameters(), lr=1e-4, default_group_warmup_steps=50)
    scaler = torch.amp.GradScaler('cuda')

    model.train()
    torch.autograd.set_detect_anomaly(True)

    grad_accum_steps = 4
    for step in range(1, 51):
        optimizer.zero_grad(set_to_none=True)
        domain = loader.domains[step % len(loader.domains)]
        accum_loss = 0.0

        for accum_i in range(grad_accum_steps):
            x, y = loader.get_batch(domain, split="train")
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast('cuda'):
                logits, loss, telem = model(x, targets=y, allow_spawning=False)
                loss = loss / grad_accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[NAN DETECTED] Step {step} Loss is {loss.item()}!")
                return

            scaler.scale(loss).backward()
            accum_loss += loss.item() * grad_accum_steps

        scaler.unscale_(optimizer)
        
        # Check grad norms
        total_norm = 0.0
        has_nan_grad = False
        for p in model.parameters():
            if p.grad is not None:
                p_norm = p.grad.data.norm(2).item()
                if math.isnan(p_norm) or math.isinf(p_norm):
                    has_nan_grad = True
                total_norm += p_norm ** 2
        total_norm = total_norm ** 0.5
        
        if has_nan_grad:
            print(f"[NAN GRADIENT DETECTED] Step {step} Grad Norm: {total_norm}")
            return

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 5 == 0 or step == 1:
            print(f"Step {step:2d} | Domain: [{domain[:14]:<14}] | Loss: {accum_loss:.4f} | Grad Norm: {total_norm:.4f} | Total Exp: {sum(b.hyper_moe.num_experts for b in model.blocks)}")

if __name__ == "__main__":
    diagnose()
