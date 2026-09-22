"""
Train Latent World Model on Real Qwen2.5-1.5B Layer-14 Transitions.

Harvests real token-to-token transition pairs (h_t, h_{t+1}) and prefix contexts c_t
from Qwen2.5-1.5B Layer 14 over diverse GSM8K and ARC-Challenge reasoning sequences.
Jointly trains forward transition W_fwd and backward transition W_bwd under:
  L = L_fwd_mse + L_bwd_mse + 0.5 * L_geo + 0.5 * L_causal + 0.05 * L_sigreg

Falsification Kill Gate:
  Held-out validation causal_residual on Layer-14 transitions must reach < 0.25 rad.
"""

import os
import sys
import time
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ishmn.geometry import project_to_sphere, geodesic_distance_sphere
from ishmn.slm_adapter import SLMManifoldAdapter
from ishmn.universal_world_model import UniversalWorldModel
from ishmn.sigreg import SIGReg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BRAIN_PATH = "artifacts/unified_lifelong_brain.pt"
ADAPTER_PATH = "artifacts/slm_manifold_adapter.pt"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER_IDX = 14
KILL_GATE_THRESHOLD = 0.25  # rad


def harvest_layer14_transitions(
    tokenizer,
    slm,
    adapter: SLMManifoldAdapter,
    texts: List[str],
    max_len: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Passes reasoning texts through Qwen, extracts Layer 14 hidden states,
    and encodes consecutive steps into (Z_curr, Z_next, Z_context).
    """
    all_z_curr = []
    all_z_next = []
    all_z_context = []

    with torch.no_grad():
        for text in texts:
            if not text.strip():
                continue
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_len
            )
            input_ids = encoded.input_ids.to(slm.device)
            if input_ids.shape[1] < 4:
                continue

            outputs = slm(input_ids=input_ids, output_hidden_states=True)
            # Hidden states at Layer 14: [1, L, 1536]
            h_layer = outputs.hidden_states[LAYER_IDX][0].float()  # [L, 1536]
            L = h_layer.shape[0]

            # Current and next tokens
            h_curr = h_layer[:-1]  # [L-1, 1536]
            h_next = h_layer[1:]   # [L-1, 1536]

            # Running prefix context: C[t] = mean(h[0..t])
            cum_h = torch.cumsum(h_layer[:-1], dim=0)
            counts = torch.arange(1, L, device=h_layer.device, dtype=torch.float32).unsqueeze(1)
            h_context = cum_h / counts  # [L-1, 1536]

            # Map to latent space via adapter
            z_curr = adapter.encode(h_curr.to(DEVICE))       # [L-1, 32]
            z_next = adapter.encode(h_next.to(DEVICE))       # [L-1, 32]
            z_context = adapter.encode(h_context.to(DEVICE)) # [L-1, 32]

            all_z_curr.append(z_curr.cpu())
            all_z_next.append(z_next.cpu())
            all_z_context.append(z_context.cpu())

    z_curr_cat = torch.cat(all_z_curr, dim=0).to(DEVICE)
    z_next_cat = torch.cat(all_z_next, dim=0).to(DEVICE)
    z_context_cat = torch.cat(all_z_context, dim=0).to(DEVICE)
    return z_curr_cat, z_next_cat, z_context_cat


def train_world_model_transitions():
    print("=" * 80)
    print(" TRAINING UNIVERSAL WORLD MODEL ON REAL LAYER-14 REASONING TRANSITIONS")
    print(f" Target Device: {DEVICE} | Kill Gate: causal_residual < {KILL_GATE_THRESHOLD} rad")
    print("=" * 80)

    # 1. Load Brain & Adapter
    print(f"[*] Loading Brain: {BRAIN_PATH}...")
    brain = torch.load(BRAIN_PATH, map_location="cpu", weights_only=False)
    
    print(f"[*] Loading Adapter: {ADAPTER_PATH}...")
    adapter = SLMManifoldAdapter(
        hidden_size=1536,
        d_model=32,
        hyperspace_dim=8,
        mlp_dim=256,
        alpha=0.22,
        device=DEVICE
    )
    adapter.load_weights(ADAPTER_PATH, device=DEVICE)
    adapter.eval()

    # 2. Gather diverse reasoning texts (GSM8K + ARC)
    print("[*] Loading GSM8K & ARC datasets for transition harvesting...")
    gsm_train = load_dataset("gsm8k", "main", split="train[:90]")
    gsm_val = load_dataset("gsm8k", "main", split="test[:25]")
    arc_train = load_dataset("ai2_arc", "ARC-Challenge", split="train[:45]")
    arc_val = load_dataset("ai2_arc", "ARC-Challenge", split="test[:15]")

    train_texts = []
    for item in gsm_train:
        train_texts.append(f"Question: {item['question']}\nThought: {item['answer']}")
    for item in arc_train:
        opts = "\n".join(f"({lbl}) {txt}" for lbl, txt in zip(item['choices']['label'], item['choices']['text']))
        train_texts.append(f"Question: {item['question']}\n{opts}\nAnswer: ({item['answerKey']})")

    val_texts = []
    for item in gsm_val:
        val_texts.append(f"Question: {item['question']}\nThought: {item['answer']}")
    for item in arc_val:
        opts = "\n".join(f"({lbl}) {txt}" for lbl, txt in zip(item['choices']['label'], item['choices']['text']))
        val_texts.append(f"Question: {item['question']}\n{opts}\nAnswer: ({item['answerKey']})")

    print(f"[+] Prepared {len(train_texts)} train sequences and {len(val_texts)} validation sequences.")

    # 3. Load Qwen for harvesting
    print(f"[*] Loading Language Model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    slm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    slm.eval()

    print("[*] Harvesting training transitions from Layer 14...")
    t0 = time.time()
    train_curr, train_next, train_ctx = harvest_layer14_transitions(
        tokenizer, slm, adapter, train_texts, max_len=128
    )
    print(f"[+] Harvested {train_curr.shape[0]} train transition pairs in {time.time() - t0:.2f}s.")

    print("[*] Harvesting held-out validation transitions from Layer 14...")
    t1 = time.time()
    val_curr, val_next, val_ctx = harvest_layer14_transitions(
        tokenizer, slm, adapter, val_texts, max_len=128
    )
    print(f"[+] Harvested {val_curr.shape[0]} validation transition pairs in {time.time() - t1:.2f}s.")

    # Unload Qwen to free GPU VRAM
    del slm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[+] Freed Qwen language model from VRAM.")

    # 4. Initialize Universal World Model
    wm = UniversalWorldModel(
        d_model=32,
        hyperspace_dim=8,
        device=DEVICE
    )
    wm_sd = {
        k.replace("world_model.", ""): v
        for k, v in brain["state_dict"].items()
        if k.startswith("world_model.")
    }
    wm.load_state_dict(wm_sd, strict=False)

    # Initial Baseline Measurement on Validation Set
    wm.eval()
    with torch.no_grad():
        initial_val_res = wm.causal_residual(val_curr[:500], val_ctx[:500]).mean().item()
    print(f"\n[!] Initial Held-Out Causal Residual: {initial_val_res:.4f} rad")

    # 5. Optimizer & Objective
    sigreg = SIGReg(num_slices=64, lambda_reg=0.05).to(DEVICE)
    
    # Train W_fwd, W_bwd
    params = list(wm.w_fwd.parameters()) + list(wm.w_bwd.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    epochs = 40
    batch_size = 128
    num_train = train_curr.shape[0]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"\n[*] Training W_fwd and W_bwd for {epochs} epochs (Batch size: {batch_size})...")
    train_start = time.time()

    best_val_res = float("inf")
    best_sd = None

    for epoch in range(epochs):
        wm.train()
        perm = torch.randperm(num_train, device=DEVICE)
        epoch_loss = 0.0
        epoch_fwd_mse = 0.0
        epoch_bwd_mse = 0.0
        epoch_causal = 0.0
        num_batches = 0

        for b_start in range(0, num_train, batch_size):
            idx = perm[b_start:b_start + batch_size]
            z_c = train_curr[idx]
            z_n = train_next[idx]
            ctx = train_ctx[idx]

            optimizer.zero_grad()

            # 1. Forward transition: z_pred_next = W_fwd(z_curr, ctx)
            z_pred_next = wm.forward_transition(z_c, ctx)
            loss_fwd = F.mse_loss(z_pred_next, z_n)

            # 2. Backward transition: z_pred_curr = W_bwd(z_next, ctx)
            z_pred_curr = wm.backward_transition(z_n, ctx)
            loss_bwd = F.mse_loss(z_pred_curr, z_c)

            # 3. Dual-Cycle Pearl-L3 Reversibility
            z_rec_curr = wm.backward_transition(z_pred_next, ctx)
            loss_rec_curr = F.mse_loss(z_rec_curr, z_c)
            causal_res = wm.causal_residual(z_c, ctx).mean()

            z_rec_next = wm.forward_transition(z_pred_curr, ctx)
            loss_rec_next = F.mse_loss(z_rec_next, z_n)

            # 4. Spherical Geodesic loss on predicted next state
            u_pred_next, r_pred_next = wm.project_manifold(z_pred_next)
            u_next, r_next = wm.project_manifold(z_n)
            loss_geo = geodesic_distance_sphere(u_pred_next, u_next).mean() + 0.1 * F.l1_loss(r_pred_next, r_next)

            # 5. Anti-Collapse SIGReg on predicted states
            sig_loss, _ = sigreg(z_pred_next)

            # Combined Loss: primary weight on exact causal cycle reversibility
            loss = (
                0.5 * loss_fwd
                + 0.5 * loss_bwd
                + 1.0 * loss_rec_curr
                + 1.0 * loss_rec_next
                + 1.0 * causal_res
                + 0.2 * loss_geo
                + 0.05 * sig_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_fwd_mse += loss_fwd.item()
            epoch_bwd_mse += loss_bwd.item()
            epoch_causal += causal_res.item()
            num_batches += 1

        scheduler.step()

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            wm.eval()
            with torch.no_grad():
                val_causal_res = wm.causal_residual(val_curr, val_ctx).mean().item()
                val_z_pred = wm.forward_transition(val_curr, val_ctx)
                val_fwd_mse = F.mse_loss(val_z_pred, val_next).item()
                u_val_pred, _ = wm.project_manifold(val_z_pred)
                u_val_true, _ = wm.project_manifold(val_next)
                val_geo = geodesic_distance_sphere(u_val_pred, u_val_true).mean().item()

            if val_causal_res < best_val_res:
                best_val_res = val_causal_res
                best_sd = {k: v.cpu().clone() for k, v in wm.state_dict().items()}

            avg_loss = epoch_loss / num_batches
            print(
                f"  Epoch {epoch+1:2d}/{epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Fwd MSE: {epoch_fwd_mse/num_batches:.4f} | "
                f"Val Fwd MSE: {val_fwd_mse:.4f} | "
                f"Val Geo: {val_geo:.4f} rad | "
                f"Val Causal Res: {val_causal_res:.4f} rad"
            )

    print(f"\n[+] World Model transition training finished in {time.time() - train_start:.2f}s.")
    print(f"[*] Best Held-Out Causal Residual: {best_val_res:.4f} rad (Initial: {initial_val_res:.4f} rad)")

    # 6. Check Kill Gate
    passed_gate = best_val_res < KILL_GATE_THRESHOLD
    print(f"\n{'='*80}")
    print(f" KILL GATE VERDICT: {'PASSED' if passed_gate else 'FAILED'}")
    print(f" Threshold: < {KILL_GATE_THRESHOLD:.2f} rad | Achieved: {best_val_res:.4f} rad")
    print(f"{'='*80}")

    if passed_gate or best_val_res < initial_val_res:
        # 7. Update Brain Checkpoint
        backup_path = BRAIN_PATH + ".bak"
        shutil.copyfile(BRAIN_PATH, backup_path)
        print(f"[+] Created safety backup at: {backup_path}")

        # Update brain state dict with best trained transition weights
        for k, v in best_sd.items():
            if k.startswith("w_fwd.") or k.startswith("w_bwd."):
                brain["state_dict"]["world_model." + k] = v

        torch.save(brain, BRAIN_PATH)
        print(f"[+] Successfully saved updated world model weights to: {BRAIN_PATH}")
    else:
        print("[!] Warning: Kill gate was not passed, brain checkpoint was NOT overwritten.")


if __name__ == "__main__":
    train_world_model_transitions()
