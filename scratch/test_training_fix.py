import sys
sys.path.insert(0, ".")
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from ishmn.slm_adapter import SLMManifoldAdapter
from ishmn.universal_world_model import UniversalWorldModel
from ishmn.geometry import geodesic_distance_sphere
from ishmn.sigreg import SIGReg
from experiments.train_world_model_on_layer14 import harvest_layer14_transitions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Harvest small dataset
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
slm = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)
adapter = SLMManifoldAdapter(hidden_size=1536, d_model=32, hyperspace_dim=8, device=device)
adapter.load_weights("artifacts/slm_manifold_adapter.pt", device=device)
adapter.eval()

gsm_train = load_dataset("gsm8k", "main", split="train[:20]")
train_texts = [f"Question: {item['question']}\nThought: {item['answer']}" for item in gsm_train]
gsm_val = load_dataset("gsm8k", "main", split="test[:10]")
val_texts = [f"Question: {item['question']}\nThought: {item['answer']}" for item in gsm_val]

print("[*] Harvesting transitions...")
train_curr, train_next, train_ctx = harvest_layer14_transitions(tokenizer, slm, adapter, train_texts, max_len=128)
val_curr, val_next, val_ctx = harvest_layer14_transitions(tokenizer, slm, adapter, val_texts, max_len=128)

del slm
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print(f"[+] Train: {train_curr.shape[0]} | Val: {val_curr.shape[0]}")

# 2. Load World Model
brain = torch.load("artifacts/unified_lifelong_brain.pt", map_location="cpu", weights_only=False)
wm = UniversalWorldModel(d_model=32, hyperspace_dim=8, device=device)
wm_sd = {k.replace("world_model.", ""): v for k, v in brain["state_dict"].items() if k.startswith("world_model.")}
wm.load_state_dict(wm_sd, strict=False)

init_val_res = wm.causal_residual(val_curr, val_ctx).mean().item()
print(f"[*] Initial Validation Causal Residual: {init_val_res:.4f} rad")

# 3. Train with Balanced Dual-Cycle Objective
params = list(wm.w_fwd.parameters()) + list(wm.w_bwd.parameters())
optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)
sigreg = SIGReg(num_slices=64, lambda_reg=0.05).to(device)

num_train = train_curr.shape[0]
batch_size = 128

for epoch in range(40):
    wm.train()
    perm = torch.randperm(num_train, device=device)
    epoch_causal = 0.0
    num_b = 0

    for b_start in range(0, num_train, batch_size):
        idx = perm[b_start:b_start + batch_size]
        zc = train_curr[idx]
        zn = train_next[idx]
        ctx = train_ctx[idx]

        optimizer.zero_grad()

        # Forward & Backward transitions
        z_pred_next = wm.forward_transition(zc, ctx)
        loss_fwd = F.mse_loss(z_pred_next, zn)

        z_pred_curr = wm.backward_transition(zn, ctx)
        loss_bwd = F.mse_loss(z_pred_curr, zc)

        # True Pearl-L3 Cycle 1: Forward -> Backward roundtrip
        z_rec_curr = wm.backward_transition(z_pred_next, ctx)
        loss_rec_curr = F.mse_loss(z_rec_curr, zc)
        causal_res_fwd = wm.causal_residual(zc, ctx).mean()

        # True Pearl-L3 Cycle 2: Backward -> Forward roundtrip
        z_rec_next = wm.forward_transition(z_pred_curr, ctx)
        loss_rec_next = F.mse_loss(z_rec_next, zn)

        # Geodesic alignment
        u_pred_next, r_pred_next = wm.project_manifold(z_pred_next)
        u_next, r_next = wm.project_manifold(zn)
        loss_geo = geodesic_distance_sphere(u_pred_next, u_next).mean() + 0.1 * F.l1_loss(r_pred_next, r_next)

        # Anti-collapse
        sig_loss, _ = sigreg(z_pred_next)

        # Combined Loss with primary weight on reversibility
        loss = (
            0.5 * loss_fwd
            + 0.5 * loss_bwd
            + 1.0 * loss_rec_curr
            + 1.0 * loss_rec_next
            + 1.0 * causal_res_fwd
            + 0.2 * loss_geo
            + 0.05 * sig_loss
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        epoch_causal += causal_res_fwd.item()
        num_b += 1

    scheduler.step()

    if (epoch + 1) % 5 == 0 or epoch == 0:
        wm.eval()
        with torch.no_grad():
            v_res = wm.causal_residual(val_curr, val_ctx).mean().item()
            v_frac = (wm.causal_residual(val_curr, val_ctx) < 0.35).float().mean().item() * 100
            print(f"Epoch {epoch+1:2d} | Val Causal Res: {v_res:.4f} rad | <0.35 rad: {v_frac:.1f}%")
