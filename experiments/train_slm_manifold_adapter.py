"""
Train SLM-Manifold Cross-Space Adapter Bridge.

Extracts Layer 14 hidden states from Qwen2.5-1.5B for the 5,268 crystallized semantic concepts
in UCE's unified lifelong brain, and trains the bidirectional nonlinear adapter (SLMManifoldAdapter)
via MSE, Cosine Similarity, Spherical Geodesic distance, and Cycle-Consistency.

Saves trained weights to artifacts/slm_manifold_adapter.pt.
"""

import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ishmn.geometry import project_to_sphere, geodesic_distance_sphere
from ishmn.slm_adapter import SLMManifoldAdapter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BRAIN_PATH = "artifacts/unified_lifelong_brain.pt"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SAVE_PATH = "artifacts/slm_manifold_adapter.pt"
LAYER_IDX = 14


def train_adapter():
    print("=" * 80)
    print(" TRAINING SLM-MANIFOLD CROSS-SPACE ADAPTER BRIDGE")
    print("=" * 80)
    print(f"[*] Execution Device: {DEVICE}")

    # 1. Load UCE Brain & Semantic Concepts
    print(f"[*] Loading crystallized semantic concepts from: {BRAIN_PATH}...")
    brain = torch.load(BRAIN_PATH, map_location="cpu", weights_only=False)
    concepts = brain.get("semantic_concepts", [])
    print(f"[+] Loaded {len(concepts)} crystallized concepts from UCE brain.")

    if not concepts:
        raise ValueError("No semantic concepts found in brain checkpoint!")

    # 2. Filter valid words and concept vectors
    valid_data = []
    for sc in concepts:
        w = sc.get("word", "").strip()
        v = sc.get("v")
        u = sc.get("u")
        if w and v is not None and u is not None:
            if v.dim() > 1:
                v = v.squeeze(0)
            if u.dim() > 1:
                u = u.squeeze(0)
            valid_data.append((w, v.float(), u.float()))

    print(f"[+] Prepared {len(valid_data)} concept pairs for training.")

    # 3. Load Qwen2.5-1.5B
    print(f"[*] Loading Language Model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    slm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    slm.eval()

    # 4. Extract Layer 14 hidden states for all concept words
    print(f"[*] Extracting Layer {LAYER_IDX} hidden states from Language Model...")
    t0 = time.time()
    batch_size = 64
    all_h14 = []
    all_v = []
    all_u = []

    words = [item[0] for item in valid_data]
    v_tensors = [item[1] for item in valid_data]
    u_tensors = [item[2] for item in valid_data]

    # Pre-pad single token inputs
    with torch.no_grad():
        for i in range(0, len(words), batch_size):
            batch_words = words[i:i + batch_size]
            encoded = tokenizer(batch_words, return_tensors="pt", padding=True, truncation=True)
            input_ids = encoded.input_ids.to(slm.device)
            attention_mask = encoded.attention_mask.to(slm.device)

            outputs = slm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # Layer 14 hidden state: [B, L, H]
            h_layer = outputs.hidden_states[LAYER_IDX]

            # Pool representation at last active token per sequence
            seq_lens = attention_mask.sum(dim=1) - 1
            h_last = torch.stack([h_layer[b, seq_lens[b], :] for b in range(len(batch_words))], dim=0)

            all_h14.append(h_last.cpu().float())
            all_v.append(torch.stack(v_tensors[i:i + len(batch_words)], dim=0))
            all_u.append(torch.stack(u_tensors[i:i + len(batch_words)], dim=0))

    H14 = torch.cat(all_h14, dim=0).to(DEVICE) # [N, 1536]
    V_target = torch.cat(all_v, dim=0).to(DEVICE) # [N, 32]
    U_target = torch.cat(all_u, dim=0).to(DEVICE) # [N, 8]

    print(f"[+] Extracted {H14.shape[0]} hidden states in {time.time() - t0:.2f}s.")
    print(f"    - H14 Shape: {H14.shape} | Mean Norm: {H14.norm(dim=-1).mean().item():.2f}")
    print(f"    - V Shape  : {V_target.shape} | Mean Norm: {V_target.norm(dim=-1).mean().item():.2f}")
    print(f"    - U Shape  : {U_target.shape}")

    # Free memory
    del slm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Initialize SLMManifoldAdapter
    adapter = SLMManifoldAdapter(
        hidden_size=1536,
        d_model=32,
        hyperspace_dim=8,
        mlp_dim=256,
        alpha=0.22,
        device=DEVICE
    )

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5)

    num_samples = H14.shape[0]
    train_batch_size = 128
    epochs = 80

    print(f"\n[*] Training Adapter Bridge for {epochs} epochs (batch size: {train_batch_size})...")
    t_train_start = time.time()

    wm_to_sphere_weight = brain["state_dict"]["world_model.to_sphere.weight"].to(DEVICE)
    wm_to_sphere_bias = brain["state_dict"].get("world_model.to_sphere.bias")
    if wm_to_sphere_bias is not None:
        wm_to_sphere_bias = wm_to_sphere_bias.to(DEVICE)

    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        epoch_loss = 0.0
        epoch_cos_z = 0.0
        epoch_cos_h = 0.0
        epoch_geo = 0.0
        num_batches = 0

        for b_start in range(0, num_samples, train_batch_size):
            idx = perm[b_start:b_start + train_batch_size]
            h_b = H14[idx]
            v_b = V_target[idx]
            u_b = U_target[idx]

            optimizer.zero_grad()

            # Forward Down-Projection
            z_pred = adapter.encode(h_b) # [B, 32]
            loss_mse_z = F.mse_loss(z_pred, v_b)
            cos_sim_z = F.cosine_similarity(z_pred, v_b, dim=-1).mean()
            loss_cos_z = 1.0 - cos_sim_z

            # Manifold Spherical Alignment via world model to_sphere
            u_pred = project_to_sphere(F.linear(z_pred, wm_to_sphere_weight, wm_to_sphere_bias))
            geo_dist = geodesic_distance_sphere(u_pred, u_b).mean()

            # Forward Up-Projection (End-to-End reconstruction from predicted z)
            h_steer_pred = adapter.decode(z_pred) # [B, 1536]
            cos_sim_h = F.cosine_similarity(h_steer_pred, h_b, dim=-1).mean()
            loss_cos_h = 1.0 - cos_sim_h

            # Cycle Consistency: down(up(v)) ~ v
            h_clean = adapter.decode(v_b)
            z_cycle = adapter.encode(h_clean)
            loss_cycle = F.mse_loss(z_cycle, v_b)

            # Combined Multi-Objective Loss
            total_loss = (
                loss_mse_z
                + 0.6 * loss_cos_z
                + 0.4 * geo_dist
                + 0.6 * loss_cos_h
                + 0.3 * loss_cycle
            )

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_cos_z += cos_sim_z.item()
            epoch_cos_h += cos_sim_h.item()
            epoch_geo += geo_dist.item()
            num_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = epoch_loss / num_batches
            avg_cos_z = epoch_cos_z / num_batches
            avg_cos_h = epoch_cos_h / num_batches
            avg_geo = epoch_geo / num_batches
            print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Cos(z,v): {avg_cos_z:.4f} | Cos(h,h_pred): {avg_cos_h:.4f} | GeoDist: {avg_geo:.4f} rad")

    total_time = time.time() - t_train_start
    print(f"\n[+] Adapter Bridge training complete in {total_time:.2f}s.")

    # 6. Save Model Checkpoint
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    adapter.save_weights(SAVE_PATH)
    print(f"[+] Saved trained adapter weights to: {SAVE_PATH}")


if __name__ == "__main__":
    train_adapter()
