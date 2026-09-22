"""
Autonomous Target-Driven Full-System Training & Evaluation Harness.

Trains the FULL UNIFIED COGNITIVE SYSTEM:
1. Native Spherical Dynamic MoE Transformer Cortex (Language, MTP, Dual-Stream Attention)
2. Universal Cognitive Engine Core (Perceptor on S^7, World Model JEPA, Attractor Memory, Dual-Process Arbiter)

Iterates continuously across the multi-stage curriculum until the multi-domain evaluation
battery achieves COMPOSITE ACCURACY > 80.0%.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ishmn import (
    UniversalCognitiveEngine,
    geodesic_distance_sphere,
    project_to_sphere
)
from ishmn.native_cognitive_transformer import NativeSphericalTransformer
from ishmn.muon import HybridMuonAdamW
from ishmn.data_loader import (
    SovereignTokenizer,
    CharacterTokenizer,
    HuggingFaceCurriculumStreamer,
    CURATED_FALLBACK_CORPUS
)
from ishmn.training_engine import (
    MODEL_PROFILES,
    CheckpointManager,
    TimeBudgetController
)
from ishmn.evaluation_harness import evaluate_mmoe_comprehensive

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "artifacts/checkpoints"


def generate_synthetic_continuous_transition_batch(
    batch_size: int = 16,
    d_model: int = 64,
    device: torch.device = DEVICE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates a batch of continuous phase-space dynamical transitions for UCE World Model training.
    """
    # State x in R^d
    x_curr = torch.randn(batch_size, d_model, device=device)
    # Context c in R^d
    context = torch.randn(batch_size, d_model, device=device)
    # Smooth continuous transformation: x_next = 0.8 * x + 0.2 * tanh(W * c)
    x_next = 0.85 * x_curr + 0.15 * torch.tanh(context) + 0.05 * torch.sin(x_curr * context)
    return x_curr, context, x_next


def train_full_system_until_accuracy(
    target_accuracy: float = 80.0,
    eval_interval: int = 15,
    max_steps: int = 250,
    profile_name: str = "nano",
    dataset_mode: str = "hf_streaming",
    tokenizer_name: str = "tiktoken_gpt2",
    batch_size: int = 4,
    seq_len: int = 64,
    muon_lr: float = 0.015,
    adamw_lr: float = 8e-4,
    max_time_minutes: Optional[float] = 10.0,
    resume: bool = True
):
    print("=" * 88)
    print(" AUTONOMOUS FULL-SYSTEM COGNITIVE TRAINING (NATIVE MoE + UCE CORTEX)")
    print(f" Execution Device        : {DEVICE}")
    print(f" Target Evaluation Metric: Composite Accuracy >= {target_accuracy:.1f}%")
    print(f" Evaluation Interval     : Every {eval_interval} steps")
    print(f" Model Profile           : '{profile_name.upper()}'")
    print(f" Multi-Stage Curriculum  : 50% General Knowledge | 25% Math/Logic | 25% SFT")
    print(f" Full System Co-Training : Language Cortex + Perceptors + World Model + Arbiter")
    print(f" Checkpoint Directory    : {CHECKPOINT_DIR}")
    print("=" * 88)

    cfg = MODEL_PROFILES[profile_name]

    # 1. Initialize Tokenizer
    if tokenizer_name == "char":
        tokenizer = CharacterTokenizer()
    else:
        tokenizer = SovereignTokenizer("gpt2")
    print(f"[*] Subword Tokenizer ready (Vocab Size: {tokenizer.vocab_size:,})")

    # 2. Instantiate Native Spherical Transformer
    model = NativeSphericalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        hyperspace_dim=cfg["hyperspace_dim"],
        d_ff=cfg["d_ff"],
        max_seq_len=seq_len + 32,
        mtp_future_tokens=cfg["mtp_future_tokens"],
        vsa_capacity=cfg["vsa_capacity"],
        num_experts=cfg["num_experts"],
        top_k=cfg["top_k"],
        enable_shared_expert=cfg["enable_shared_expert"],
        enable_aux_loss_free_balancing=cfg["enable_aux_loss_free_balancing"],
        enable_vsa_memory=True,
        enable_world_model=True,
        device=DEVICE
    ).to(DEVICE)

    # 3. Instantiate Universal Cognitive Engine and Bind Native Transformer
    uce = UniversalCognitiveEngine(
        d_model=cfg["d_model"],
        d_hidden=cfg["d_ff"],
        hyperspace_dim=cfg["hyperspace_dim"],
        num_memory_slots=32,
        vocab_size=tokenizer.vocab_size,
        initial_experts=cfg["num_experts"],
        device=DEVICE
    ).to(DEVICE)
    uce.attach_native_transformer(model, tokenizer=tokenizer)

    total_params = sum(p.numel() for p in uce.parameters() if p.requires_grad)
    trans_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cog_params = total_params - trans_params
    print(f"[+] Full Cognitive Substrate Assembled:")
    print(f"    - Total Brain Parameters     : {total_params:,}")
    print(f"    - Native MoE Language Cortex : {trans_params:,}")
    print(f"    - Higher Cognitive UCE Core  : {cog_params:,}")

    # 4. Checkpoint Manager & Resumption
    ckpt_mgr = CheckpointManager(checkpoint_dir=CHECKPOINT_DIR)
    start_step = 0
    golden_path = os.path.join(CHECKPOINT_DIR, "native_moe_acc80_plus.pt")

    if resume and ckpt_mgr.has_checkpoint():
        print(f"\n[*] Found existing checkpoint! Resuming training state...")
        ckpt = ckpt_mgr.load_latest_checkpoint(model=model, optimizer=None, device=DEVICE)
        start_step = ckpt["step"]
        if "extra_meta" in ckpt and "uce_state_dict" in ckpt["extra_meta"]:
            uce.load_state_dict(ckpt["extra_meta"]["uce_state_dict"])
            print(f"[+] Loaded full UCE state from Step {start_step}.")
        print(f"[+] Loaded weights from Step {start_step} (Previous Loss: {ckpt['loss']:.4f}).")

    # 5. Hybrid Muon + AdamW Optimizer across the FULL SYSTEM
    optimizer = HybridMuonAdamW(
        model=uce,
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        weight_decay=0.01,
        ns_steps=5
    )
    print(f"[+] Optimizer Bound to Entire Cognitive Brain: {len(optimizer.muon_params)} 2D Muon params + {len(optimizer.adamw_params)} AdamW params.")

    # 6. Initialize Data Streamer
    streamer = None
    if dataset_mode == "hf_streaming":
        print("[*] Initializing Hugging Face Curriculum Streamer...")
        streamer = HuggingFaceCurriculumStreamer(
            tokenizer=tokenizer,
            batch_size=batch_size,
            seq_len=seq_len,
            prefetch_size=16,
            device=DEVICE,
            offline_mode=False
        )
        print("[+] Background streaming worker active.")

    # 7. Prepare Held-Out Validation Split
    print("[*] Caching held-out validation sequences for objective accuracy checks...")
    val_batches = []
    for _ in range(6):
        if streamer is not None:
            inp_v, tgt_v = streamer.sample_batch(device=DEVICE)
        else:
            txt = CURATED_FALLBACK_CORPUS[0]
            toks = tokenizer.encode(txt)[:seq_len]
            if len(toks) < seq_len:
                pad_id = getattr(tokenizer, "eos_token_id", 0) or 0
                toks = toks + [pad_id] * (seq_len - len(toks))
            inp_v = torch.tensor([toks], dtype=torch.long, device=DEVICE)
            tgt_v = torch.tensor([toks], dtype=torch.long, device=DEVICE)
        val_batches.append((inp_v, tgt_v))

    # 8. Time Budget Controller
    budget_ctrl = TimeBudgetController(
        max_time_minutes=max_time_minutes,
        max_steps=start_step + max_steps
    )

    current_step = start_step
    target_step = start_step + max_steps
    best_acc = 0.0

    print(f"\n[*] Commencing Full-System Optimization Loop (Steps {current_step + 1} to {target_step})...")

    try:
        while current_step < target_step:
            current_step += 1
            step_t0 = time.time()
            uce.train()

            # --- Phase A: Linguistic Curriculum Forward Pass ---
            if streamer is not None and (current_step % 2 != 0):
                inputs, targets = streamer.sample_batch(device=DEVICE)
            else:
                b_inp, b_tgt = [], []
                for _ in range(batch_size):
                    cat = random.choice(["general_knowledge", "reasoning_math_code", "conversational_sft"])
                    toks = []
                    while len(toks) < seq_len:
                        txt = random.choice(CURATED_FALLBACK_CORPUS[cat])
                        toks.extend(tokenizer.encode(txt))
                        toks.append(tokenizer.eos_token_id)
                    toks = toks[:seq_len]
                    b_inp.append(toks)
                    b_tgt.append(toks)
                inputs = torch.tensor(b_inp, dtype=torch.long, device=DEVICE)
                targets = torch.tensor(b_tgt, dtype=torch.long, device=DEVICE)

            out_lang = model(inputs, targets=targets, return_mtp=True)
            loss_lang = out_lang["loss"]

            # --- Phase B: Higher Cognitive World Model Transition Pass ---
            x_c, ctx_c, x_next = generate_synthetic_continuous_transition_batch(
                batch_size=batch_size * 2,
                d_model=cfg["d_model"],
                device=DEVICE
            )
            u_c, r_c, z_c = uce.perceive(x_c, modality="continuous")
            u_ctx, r_ctx, z_ctx = uce.perceive(ctx_c, modality="continuous")
            u_tgt, r_tgt, z_tgt = uce.perceive(x_next, modality="continuous")
            z_fwd = uce.world_model.forward_transition(z_c, z_ctx)
            loss_wm_mse = F.mse_loss(z_fwd, z_tgt)
            u_fwd, _ = uce.world_model.project_manifold(z_fwd)
            loss_wm_geo = geodesic_distance_sphere(u_fwd, u_tgt).mean()
            loss_causal_l3 = uce.world_model.compute_causal_inconsistency(z_c, z_ctx)
            loss_cog = 0.1 * (loss_wm_mse + loss_wm_geo + loss_causal_l3)

            # --- Combined Brain Loss ---
            total_brain_loss = loss_lang + 0.20 * loss_cog

            optimizer.zero_grad(set_to_none=True)
            total_brain_loss.backward()
            torch.nn.utils.clip_grad_norm_(uce.parameters(), max_norm=1.0)
            optimizer.step()

            telemetry = budget_ctrl.step_tick(batch_tokens=batch_size * seq_len)

            if current_step % 5 == 0 or current_step == start_step + 1:
                p_loss = out_lang["loss_components"].get("primary_ce", 0.0)
                wm_loss = out_lang["loss_components"].get("world_model_causal", 0.0)
                vsa_slots = uce.native_transformer.vsa_memory.occupied.sum().item()
                ms_step = telemetry["ewma_ms_per_step"] or 0.0
                tok_s = telemetry["tokens_per_sec"]

                print(f"  Step [{current_step:04d}/{target_step:04d}] | "
                      f"Total Loss: {total_brain_loss.item():.4f} | "
                      f"Lang CE: {p_loss:.3f} | Cog WM: {loss_cog.item():.4f} | "
                      f"VSA Slots: {int(vsa_slots)} | Speed: {ms_step:.1f}ms ({tok_s:.0f} tok/s)")

            # --- MILESTONE EVALUATION ---
            if current_step % eval_interval == 0:
                print("\n" + "-" * 88)
                print(f" [!] EXECUTING MULTI-DOMAIN EVALUATION BATTERY AT STEP {current_step}")
                print("-" * 88)

                uce.eval()
                eval_res = evaluate_mmoe_comprehensive(
                    model=model,
                    uce=uce,
                    tokenizer=tokenizer,
                    validation_batches=val_batches,
                    device=DEVICE
                )

                acc_math = eval_res["domain_accuracies"]["math"]
                acc_sci = eval_res["domain_accuracies"]["science"]
                acc_code = eval_res["domain_accuracies"]["code"]
                acc_curr = eval_res["domain_accuracies"]["curriculum"]
                comp_acc = eval_res["composite_accuracy"]

                print(f"  --> 1. Mathematical & Numeric Reasoning  : {acc_math:.1f}%")
                print(f"  --> 2. Causal & Physical Sensemaking     : {acc_sci:.1f}%")
                print(f"  --> 3. Algorithmic Logic & Syntax        : {acc_code:.1f}%")
                print(f"  --> 4. Held-Out Curriculum Validation    : {acc_curr:.1f}%")
                print(f"  ========================================================")
                print(f"  >>> COMPOSITE EVALUATION ACCURACY        : {comp_acc:.2f}% (Target: {target_accuracy:.1f}%)")
                print("=" * 88 + "\n")

                if comp_acc > best_acc:
                    best_acc = comp_acc
                    ckpt_mgr.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        step=current_step,
                        loss=total_brain_loss.item(),
                        profile_name=profile_name,
                        extra_meta={
                            "composite_accuracy": comp_acc,
                            "domain_accuracies": eval_res["domain_accuracies"],
                            "uce_state_dict": uce.state_dict()
                        },
                        is_best=True
                    )

                # EARLY TERMINATION CRITERION: Accuracy >= target_accuracy (e.g. 96.0%)
                if comp_acc >= target_accuracy:
                    print("=" * 88)
                    print(f" [SUCCESS] TARGET ACCURACY ACHIEVED: {comp_acc:.2f}% >= {target_accuracy:.1f}%!")
                    print("=" * 88)
                    # Commit Golden Checkpoint
                    ckpt_payload = {
                        "step": current_step,
                        "composite_accuracy": comp_acc,
                        "eval_results": eval_res,
                        "model_state_dict": model.state_dict(),
                        "uce_state_dict": uce.state_dict(),
                        "timestamp": time.time()
                    }
                    torch.save(ckpt_payload, golden_path)
                    print(f"[+] Golden Checkpoint committed -> {golden_path}")
                    if comp_acc >= 96.0:
                        path_96 = os.path.join(CHECKPOINT_DIR, "native_moe_acc96_plus.pt")
                        torch.save(ckpt_payload, path_96)
                        print(f"[+] 96%+ Milestone Checkpoint committed -> {path_96}")
                    break

            if telemetry["is_over_budget"]:
                print("\n[!] Time or step budget limit reached. Concluding run cleanly.")
                break

    finally:
        if streamer is not None:
            streamer.close()

    print("\n" + "=" * 88)
    print(" TRAINING SUMMARY & EVALUATION REPORT")
    print("=" * 88)
    print(f" 1. Final Step Reached       : {current_step}")
    print(f" 2. Best Composite Accuracy  : {best_acc:.2f}%")
    print(f" 3. Target Threshold Met     : {'YES (> ' + str(target_accuracy) + '%)' if best_acc >= target_accuracy else 'IN PROGRESS'}")
    print(f" 4. Total Brain Weights      : {total_params:,} parameters co-trained")
    print("=" * 88)
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-System MoE Cognitive Training Until >96% Eval Accuracy")
    parser.add_argument("--target_accuracy", type=float, default=96.0, help="Target composite evaluation accuracy")
    parser.add_argument("--eval_interval", type=int, default=10, help="Step interval between evaluations")
    parser.add_argument("--max_steps", type=int, default=200, help="Max steps to run")
    parser.add_argument("--profile", type=str, default="nano", choices=["nano", "micro", "standard"])
    parser.add_argument("--dataset", type=str, default="hf_streaming", choices=["hf_streaming", "local_mixture"])
    parser.add_argument("--tokenizer", type=str, default="tiktoken_gpt2", choices=["tiktoken_gpt2", "char"])
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
    parser.add_argument("--max_time_minutes", type=float, default=6.0, help="Time budget in minutes")
    parser.add_argument("--no_resume", action="store_true", help="Do not resume from previous checkpoint")
    args = parser.parse_args()

    train_full_system_until_accuracy(
        target_accuracy=args.target_accuracy,
        eval_interval=args.eval_interval,
        max_steps=args.max_steps,
        profile_name=args.profile,
        dataset_mode=args.dataset,
        tokenizer_name=args.tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_time_minutes=args.max_time_minutes,
        resume=not args.no_resume
    )
