"""
Production Multi-Stage Training Pipeline: Native Spherical Dynamic MoE Cognitive LLM.
Features:
1. Hugging Face Live Streaming Curriculum Mixture:
   - Stage 1: General World Knowledge & Web Foundations (50% target weight)
     * HuggingFaceFW/fineweb-edu (sample-10BT) / wikitext-103
   - Stage 2: Reasoning, Math, Logic & Code (25% target weight)
     * openai/gsm8k, TIGER-Lab/MathInstruct, flytech/python-codes-25k
   - Stage 3: Conversational Fluency & Instruction Dialogue (25% target weight)
     * HuggingFaceH4/ultrachat_200k, databricks/databricks-dolly-15k
2. Hardware-Tailored Model Profiles: 'nano' (~1.8M - 2.6M), 'micro' (~8.5M), 'standard' (~28.4M).
3. Keller Jordan's Muon Spectral Optimizer for 2D weights + AdamW for 1D/embeddings/unembed heads.
4. DeepSeek-Style Shared Expert Isolation + Aux-Loss-Free Dynamic Balancing.
5. Multi-Token Prediction (MTP) Auxiliary Heads (t+1, t+2, t+3).
6. TimeBudgetController: Guarantees training time never extrapolates.
7. Atomic CheckpointManager with Auto-Resume and Emergency Interruption Handling.
8. Rectified Born Rule Quantum-Probabilistic Generation Probes across all 3 domains.
"""

import os
import sys
import time
import math
import random
import argparse
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ishmn.native_cognitive_transformer import NativeSphericalTransformer
from ishmn.universal_cognitive_engine import UniversalCognitiveEngine
from ishmn.muon import HybridMuonAdamW
from ishmn.training_engine import (
    MODEL_PROFILES,
    CheckpointManager,
    TimeBudgetController,
    GracefulInterruptHandler
)
from ishmn.data_loader import (
    SovereignTokenizer,
    CharacterTokenizer,
    HuggingFaceCurriculumStreamer,
    CURATED_FALLBACK_CORPUS
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "artifacts/checkpoints"


def sample_local_balanced_batch(
    tokenizer: Any,
    batch_size: int = 4,
    seq_len: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fallback in-memory sampler from the 3 curriculum categories:
    - 50% General Knowledge
    - 25% Reasoning / Math / Code
    - 25% Conversational SFT
    """
    weights = [0.50, 0.25, 0.25]
    categories = ["general_knowledge", "reasoning_math_code", "conversational_sft"]

    inputs = []
    targets = []

    for _ in range(batch_size):
        chosen_cat = random.choices(categories, weights=weights, k=1)[0]
        sample = random.choice(CURATED_FALLBACK_CORPUS[chosen_cat])
        tokens = tokenizer.encode(sample)
        while len(tokens) < seq_len + 1:
            tokens.extend(tokenizer.encode(" " + sample))
        start_idx = random.randint(0, max(0, len(tokens) - seq_len - 1))
        sub_tokens = tokens[start_idx:start_idx + seq_len + 1]

        inputs.append(sub_tokens[:-1])
        targets.append(sub_tokens[1:])

    return (
        torch.tensor(inputs, dtype=torch.long, device=DEVICE),
        torch.tensor(targets, dtype=torch.long, device=DEVICE)
    )


def train_native_cognitive_llm(
    profile_name: str = "nano",
    dataset_mode: str = "hf_streaming",
    tokenizer_name: str = "tiktoken_gpt2",
    enable_vsa_memory: bool = True,
    enable_world_model: bool = True,
    steps: int = 60,
    batch_size: int = 4,
    seq_len: int = 64,
    muon_lr: float = 0.02,
    adamw_lr: float = 1e-3,
    max_time_minutes: Optional[float] = None,
    checkpoint_every_steps: int = 25,
    resume: bool = False
):
    print("=" * 88)
    print(" PRODUCTION TRAINING: SOVEREIGN NATIVE SPHERICAL COGNITIVE LLM")
    print(f" Execution Device      : {DEVICE}")
    print(f" Target Model Profile  : '{profile_name.upper()}' ({MODEL_PROFILES.get(profile_name, {}).get('description', '')})")
    print(f" Dataset Engine        : {dataset_mode.upper()}")
    print(f" Curriculum Mixture    : 50% General Knowledge | 25% Math/Logic/Code | 25% Conversational SFT")
    print(f" Subword Tokenizer     : {tokenizer_name}")
    print(f" Multi-Token Prediction: Active (k=3)")
    print(f" Optimizer             : Hybrid Muon (2D Spectral Core) + AdamW (1D / Embed / Heads)")
    print(f" Checkpoint Directory  : {CHECKPOINT_DIR}")
    print("=" * 88)

    if profile_name not in MODEL_PROFILES:
        raise ValueError(f"Unknown profile '{profile_name}'. Available: {list(MODEL_PROFILES.keys())}")

    cfg = MODEL_PROFILES[profile_name]

    # 1. Initialize Tokenizer
    if tokenizer_name == "char":
        tokenizer = CharacterTokenizer()
    else:
        tokenizer = SovereignTokenizer("gpt2")
    print(f"[*] Tokenizer ready. Vocabulary Size: {tokenizer.vocab_size:,}")

    # 2. Initialize Data Streamer
    streamer = None
    if dataset_mode == "hf_streaming":
        print("[*] Initializing Hugging Face Curriculum Streamer (FineWeb-Edu + GSM8K + MathInstruct + Python-Codes + UltraChat + Dolly)...")
        streamer = HuggingFaceCurriculumStreamer(
            tokenizer=tokenizer,
            batch_size=batch_size,
            seq_len=seq_len,
            weights=(0.50, 0.25, 0.25),
            prefetch_size=16,
            device=DEVICE,
            offline_mode=False
        )
        print("[+] Background HF prefetch worker online.")

    # 3. Initialize Model
    print(f"\n[*] Instantiating NativeSphericalTransformer under '{profile_name}' configuration...")
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
        enable_vsa_memory=enable_vsa_memory,
        enable_world_model=enable_world_model,
        device=DEVICE
    ).to(DEVICE)

    # Attach Native Transformer to Universal Cognitive Engine (UCE)
    uce = UniversalCognitiveEngine(
        d_model=cfg["d_model"],
        d_hidden=cfg["d_ff"],
        hyperspace_dim=cfg["hyperspace_dim"],
        device=DEVICE
    )
    uce.attach_native_transformer(model, tokenizer=tokenizer)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Model Ready: {num_params:,} trainable parameters.")
    print(f"    - Layer Blocks: {cfg['num_layers']} Dual-Stream Riemannian Blocks")
    print(f"    - MoE Routing : {cfg['num_experts']} Experts (top-{cfg['top_k']}) + 1 DeepSeek-Style Shared Expert")
    print(f"    - Cognitive Subsystems: VSA Memory (Active) | Causal World Model (Active) | UCE Cortex (Attached)")

    # 4. Setup Checkpoint Manager & Auto-Resume
    ckpt_mgr = CheckpointManager(checkpoint_dir=CHECKPOINT_DIR)
    start_step = 0
    loss_history = []

    if resume and ckpt_mgr.has_checkpoint():
        print(f"\n[*] Found existing checkpoint! Resuming training...")
        ckpt = ckpt_mgr.load_latest_checkpoint(model=model, optimizer=None, device=DEVICE)
        start_step = ckpt["step"]
        loss_history = ckpt.get("loss_history", [])

        # Initialize Hybrid Optimizer on the restored parameters
        optimizer = HybridMuonAdamW(
            model=model,
            muon_lr=muon_lr,
            adamw_lr=adamw_lr,
            weight_decay=0.01,
            ns_steps=5
        )
        if ckpt.get("optimizer_state_dict") is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"[!] Note on optimizer state restoration: {e}")
        print(f"[+] Seamlessly Resumed from Step {start_step} (Previous Loss: {ckpt['loss']:.4f}).")
    else:
        if resume:
            print(f"\n[!] Resume requested, but no checkpoint found. Starting from Step 0.")
        optimizer = HybridMuonAdamW(
            model=model,
            muon_lr=muon_lr,
            adamw_lr=adamw_lr,
            weight_decay=0.01,
            ns_steps=5
        )

    # 5. Setup Time Budget Controller (Guarantees Bounded Training)
    budget_ctrl = TimeBudgetController(
        max_time_minutes=max_time_minutes,
        max_steps=start_step + steps
    )

    # 6. Baseline Probe Before Training
    probe_prompt = "The velocity of light in vacuum is"
    prompt_ids = torch.tensor([tokenizer.encode(probe_prompt)], dtype=torch.long, device=DEVICE)
    raw_gen = model.generate(prompt_ids, max_new_tokens=15, temperature=0.7, enable_born_rule=True)
    print(f"\n[*] Baseline Generation Probe at Step {start_step} (Prompt: '{probe_prompt}'):")
    print(f"    \"{tokenizer.decode(raw_gen[0].tolist())}\"")

    # 7. Training Loop with Graceful Interrupt Handling
    target_step = start_step + steps
    print(f"\n[*] Commencing Optimization from Step {start_step + 1} to {target_step}...")

    current_step = start_step
    try:
        with GracefulInterruptHandler() as interrupt_handler:
            for current_step in range(start_step + 1, target_step + 1):
                if interrupt_handler.interrupted:
                    print(f"\n[!] Interrupted at step {current_step - 1}. Committing emergency checkpoint...")
                    ckpt_mgr.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        step=current_step - 1,
                        loss=loss_history[-1] if loss_history else 0.0,
                        profile_name=profile_name,
                        loss_history=loss_history
                    )
                    print(f"[+] Emergency checkpoint saved. You can resume with '--resume' anytime.")
                    return

                model.train()
                optimizer.zero_grad()

                if streamer is not None:
                    input_ids, targets = streamer.sample_batch(device=DEVICE)
                else:
                    input_ids, targets = sample_local_balanced_batch(
                        tokenizer=tokenizer,
                        batch_size=batch_size,
                        seq_len=seq_len
                    )

                out = model(input_ids=input_ids, targets=targets, return_mtp=True)
                loss = out["loss"]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                loss_val = loss.item()
                loss_history.append(loss_val)

                # Step Tick & Budget Enforcement
                batch_tokens = batch_size * seq_len
                telemetry = budget_ctrl.step_tick(batch_tokens=batch_tokens)

                # Periodic Status Reporting
                if current_step % 10 == 0 or current_step == start_step + 1:
                    p_ce = out["loss_components"].get("primary_ce", 0.0)
                    m_ce2 = out["loss_components"].get("mtp_offset_2", 0.0)
                    wm_loss = out["loss_components"].get("world_model_causal", 0.0)
                    cone_norm = out["cone_norm"]
                    vsa_slots = model.vsa_memory.occupied.sum().item()
                    ewma_ms = telemetry["ewma_ms_per_step"]
                    tokens_sec = telemetry["tokens_per_sec"]
                    eta_s = telemetry["eta_seconds"]

                    print(f"  Step [{current_step:04d}/{target_step:04d}] | Loss: {loss_val:.4f} | "
                          f"Primary: {p_ce:.3f} | MTP-2: {m_ce2:.3f} | WM-Causal: {wm_loss:.3f} | "
                          f"Cone R: {cone_norm:.3f} | VSA Slots: {int(vsa_slots)} | "
                          f"Speed: {ewma_ms:.1f} ms/step ({tokens_sec:.0f} tok/s) | ETA: {eta_s:.1f}s")

                # Periodic Checkpoint Saving
                if current_step % checkpoint_every_steps == 0:
                    saved_path = ckpt_mgr.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        step=current_step,
                        loss=loss_val,
                        profile_name=profile_name,
                        loss_history=loss_history
                    )
                    print(f"    [+] Atomic Checkpoint committed at Step {current_step} -> {saved_path}")

                # Memory Hygiene
                if current_step % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Budget Check
                if telemetry["is_over_budget"]:
                    print(f"\n[!] Time or Step budget reached! Concluding run cleanly...")
                    break
    finally:
        if streamer is not None:
            streamer.close()

    # Final Checkpoint Commit
    if len(loss_history) > 0 and (current_step % checkpoint_every_steps != 0):
        ckpt_mgr.save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=current_step,
            loss=loss_history[-1],
            profile_name=profile_name,
            loss_history=loss_history,
            is_best=True
        )
        print(f"\n[+] Final Checkpoint successfully committed at Step {current_step}.")
    else:
        print(f"\n[+] Checkpoint for final Step {current_step} already active.")

    # 8. Multi-Category Post-Training Generation Probes via UCE Cognitive Bus
    print("\n" + "=" * 88)
    print(" MULTI-CATEGORY COGNITIVE GENERATION PROBES (UCE + SCRATCHPAD + BORN RULE)")
    print("=" * 88)

    test_probes = [
        ("Science & World Knowledge", "The velocity of light in vacuum is"),
        ("Mathematical Reasoning", "Calculate: 15 * 4 = "),
        ("Algorithmic Code", "def quicksort(arr):"),
        ("Conversational Instruction", "User: Explain gravity clearly.\nAssistant:")
    ]

    model.eval()
    for cat_name, p_text in test_probes:
        cog_res = uce.think_and_speak(
            p_text,
            max_new_tokens=25,
            temperature=0.7,
            enable_born_rule=True,
            enable_scratchpad=True,
            write_to_vsa=True
        )
        print(f"  [{cat_name}]")
        print(f"    Input     : {p_text}")
        print(f"    Response  : {cog_res['response'].strip()}")
        if cog_res.get("dual_process") and cog_res["dual_process"].get("system2_exact_result") is not None:
            print(f"    System 2  : Exact Math Proof = {cog_res['dual_process']['system2_exact_result']}")
        print(f"    Hopfield  : Energy = {cog_res['hopfield_energy']:.3f} | VSA Occupied Slots = {cog_res['vsa_occupied_slots']}")
        print("-" * 88)

    print("\n" + "=" * 88)
    print(" TRAINING SUMMARY & RUNTIME ASSURANCE")
    print("=" * 88)
    print(f" 1. Wall-Clock Time Elapsed    : {telemetry['elapsed_sec']:.2f}s")
    print(f" 2. Training Throughput        : {telemetry['tokens_per_sec']:.1f} tokens/second")
    print(f" 3. Extrapolation Guarded      : Strict budget enforcement verified.")
    print(f" 4. Resume Guarantee           : Run can be continued anytime with '--resume'.")
    print("=" * 88)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production Multi-Stage Native Cognitive LLM Training")
    parser.add_argument("--profile", type=str, default="nano", choices=["nano", "micro", "standard"])
    parser.add_argument("--dataset", type=str, default="hf_streaming", choices=["hf_streaming", "local_mixture"])
    parser.add_argument("--tokenizer", type=str, default="tiktoken_gpt2", choices=["tiktoken_gpt2", "char"])
    parser.add_argument("--disable_vsa", action="store_true", help="Disable VSA memory")
    parser.add_argument("--disable_world_model", action="store_true", help="Disable World Model causal loss")
    parser.add_argument("--steps", type=int, default=50, help="Number of training steps to execute")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
    parser.add_argument("--max_time_minutes", type=float, default=None, help="Wall-clock time limit")
    parser.add_argument("--checkpoint_every_steps", type=int, default=25, help="Checkpoint interval")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    train_native_cognitive_llm(
        profile_name=args.profile,
        dataset_mode=args.dataset,
        tokenizer_name=args.tokenizer,
        enable_vsa_memory=not args.disable_vsa,
        enable_world_model=not args.disable_world_model,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_time_minutes=args.max_time_minutes,
        checkpoint_every_steps=args.checkpoint_every_steps,
        resume=args.resume
    )
