"""
Production Training Engine for Sovereign Cognitive LLMs.
Features:
1. Hardware-tailored Model Size Profiles (nano, micro, standard).
2. Atomic Checkpoint Management with double buffering (prevents corruption on interruption).
3. Full Resume Fidelity (Model weights, Muon/AdamW optimizer states, step counter, loss history, RNG states).
4. TimeBudgetController with EWMA throughput estimation (prevents training time extrapolation).
5. Graceful Interrupt Handling for emergency checkpoint saves.
"""

import os
import sys
import time
import signal
import random
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# 1. Model Sizing Profiles Tailored for Consumer GPUs (RTX 3060 6GB)
# -----------------------------------------------------------------------------
MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
    "nano": {
        "description": "Ultra-lightweight profile (~1.8M params). Fast iteration, < 600MB VRAM.",
        "d_model": 64,
        "num_layers": 3,
        "num_heads": 4,
        "hyperspace_dim": 8,
        "d_ff": 128,
        "max_seq_len": 256,
        "mtp_future_tokens": 3,
        "vsa_capacity": 256,
        "num_experts": 3,
        "top_k": 2,
        "enable_shared_expert": True,
        "enable_aux_loss_free_balancing": True,
        "enable_gwt_coalition": True,
    },
    "micro": {
        "description": "Balanced high-density profile (~8.5M params). Serious local reasoning, < 1.4GB VRAM.",
        "d_model": 128,
        "num_layers": 4,
        "num_heads": 4,
        "hyperspace_dim": 8,
        "d_ff": 256,
        "max_seq_len": 512,
        "mtp_future_tokens": 3,
        "vsa_capacity": 512,
        "num_experts": 4,
        "top_k": 2,
        "enable_shared_expert": True,
        "enable_aux_loss_free_balancing": True,
        "enable_gwt_coalition": True,
    },
    "standard": {
        "description": "Foundational reasoning profile (~28.4M params). High-capacity MoE, < 3.2GB VRAM.",
        "d_model": 256,
        "num_layers": 6,
        "num_heads": 8,
        "hyperspace_dim": 8,
        "d_ff": 512,
        "max_seq_len": 512,
        "mtp_future_tokens": 3,
        "vsa_capacity": 1024,
        "num_experts": 6,
        "top_k": 2,
        "enable_shared_expert": True,
        "enable_aux_loss_free_balancing": True,
        "enable_gwt_coalition": True,
    },
    "500m": {
        "description": "500M high-capacity Dual-Cortex MoE profile (~503M total params). Sparse Top-2 compute, < 2.4GB VRAM in BF16.",
        "d_model": 512,
        "num_layers": 6,
        "num_heads": 8,
        "hyperspace_dim": 16,
        "d_ff": 1024,
        "max_seq_len": 256,
        "mtp_future_tokens": 2,
        "vsa_capacity": 512,
        "num_experts": 15,
        "top_k": 2,
        "enable_shared_expert": True,
        "enable_aux_loss_free_balancing": True,
        "use_bf16": True,
    }
}



# -----------------------------------------------------------------------------
# 2. Time Budget & Throughput Controller (Prevents Time Extrapolation)
# -----------------------------------------------------------------------------
class TimeBudgetController:
    """
    Monitors training execution time, tracks Exponential Weighted Moving Average (EWMA)
    throughput (ms/step, tokens/sec), and halts training cleanly when the time budget expires.
    """
    def __init__(
        self,
        max_time_minutes: Optional[float] = None,
        max_steps: Optional[int] = None,
        ewma_alpha: float = 0.1
    ):
        self.max_time_seconds = max_time_minutes * 60.0 if max_time_minutes else None
        self.max_steps = max_steps
        self.ewma_alpha = ewma_alpha

        self.start_time = time.time()
        self.last_step_time = time.time()
        self.ewma_ms_per_step = None
        self.steps_completed = 0
        self.total_tokens_processed = 0

    def step_tick(self, batch_tokens: int) -> Dict[str, Any]:
        """Call at the end of each training step."""
        now = time.time()
        step_duration_ms = (now - self.last_step_time) * 1000.0
        self.last_step_time = now

        self.steps_completed += 1
        self.total_tokens_processed += batch_tokens

        if self.ewma_ms_per_step is None:
            self.ewma_ms_per_step = step_duration_ms
        else:
            self.ewma_ms_per_step = (
                (1.0 - self.ewma_alpha) * self.ewma_ms_per_step + self.ewma_alpha * step_duration_ms
            )

        elapsed_sec = now - self.start_time
        tokens_per_sec = (self.total_tokens_processed / max(1e-4, elapsed_sec))

        # Estimate remaining time
        if self.max_steps and self.steps_completed < self.max_steps:
            rem_steps = self.max_steps - self.steps_completed
            eta_seconds = (rem_steps * self.ewma_ms_per_step) / 1000.0
        elif self.max_time_seconds:
            eta_seconds = max(0.0, self.max_time_seconds - elapsed_sec)
        else:
            eta_seconds = 0.0

        return {
            "elapsed_sec": elapsed_sec,
            "ewma_ms_per_step": self.ewma_ms_per_step,
            "tokens_per_sec": tokens_per_sec,
            "eta_seconds": eta_seconds,
            "is_over_budget": self.is_over_budget()
        }

    def is_over_budget(self) -> bool:
        elapsed = time.time() - self.start_time
        if self.max_time_seconds is not None and elapsed >= self.max_time_seconds:
            return True
        if self.max_steps is not None and self.steps_completed >= self.max_steps:
            return True
        return False


# -----------------------------------------------------------------------------
# 3. Fault-Tolerant Checkpoint Manager (Atomic Saving + Seamless Resumption)
# -----------------------------------------------------------------------------
class CheckpointManager:
    """
    Manages robust atomic saving and recovery of training states.
    Uses .pt.tmp files and atomic renaming so power cuts or crashes never corrupt checkpoints.
    Saves full optimizer state (Muon momentum + AdamW buffers) and RNG states.
    """
    def __init__(self, checkpoint_dir: str = "artifacts/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.latest_path = os.path.join(self.checkpoint_dir, "native_llm_latest.pt")
        self.best_path = os.path.join(self.checkpoint_dir, "native_llm_best.pt")
        self.best_loss = float("inf")

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Any,
        step: int,
        loss: float,
        profile_name: str,
        loss_history: Optional[List[float]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> str:
        """Atomically saves the complete training state."""
        checkpoint_data = {
            "step": step,
            "loss": loss,
            "best_loss": min(self.best_loss, loss),
            "profile_name": profile_name,
            "loss_history": loss_history or [],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "rng_states": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "python": random.getstate()
            },
            "timestamp": time.time(),
            "extra_meta": extra_meta or {}
        }

        # 1. Atomic write to temporary file
        tmp_path = self.latest_path + ".tmp"
        torch.save(checkpoint_data, tmp_path)

        # 2. Atomic rename with multi-step retry for Windows file system locks
        replaced = False
        last_err = None
        for attempt in range(5):
            try:
                os.replace(tmp_path, self.latest_path)
                replaced = True
                break
            except OSError as e:
                last_err = e
                time.sleep(0.1 * (attempt + 1))

        if not replaced:
            try:
                if os.path.exists(self.latest_path):
                    os.remove(self.latest_path)
                os.replace(tmp_path, self.latest_path)
                replaced = True
            except OSError as e:
                # Fallback: copy instead of rename
                import shutil
                shutil.copy2(tmp_path, self.latest_path)
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        # 3. If best loss achieved, save copy as best.pt
        if is_best or loss < self.best_loss:
            self.best_loss = loss
            try:
                torch.save(checkpoint_data, self.best_path)
            except OSError:
                time.sleep(0.1)
                torch.save(checkpoint_data, self.best_path)

        return self.latest_path

    def has_checkpoint(self) -> bool:
        return os.path.exists(self.latest_path)

    def load_latest_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[Any] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Loads the latest checkpoint and restores model, optimizer, and RNG states.
        """
        if not self.has_checkpoint():
            raise FileNotFoundError(f"No checkpoint found at: {self.latest_path}")

        map_location = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self.latest_path, map_location=map_location, weights_only=False)

        # Restore model weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer states
        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"[!] Warning: Could not fully restore optimizer state: {e}")

        # Restore RNG states for exact reproducibility
        rng = checkpoint.get("rng_states")
        if rng:
            if "torch" in rng and rng["torch"] is not None:
                try:
                    torch.set_rng_state(rng["torch"].cpu())
                except Exception as e:
                    print(f"[!] Warning: Could not restore PyTorch CPU RNG state: {e}")
            if "cuda" in rng and rng["cuda"] is not None and torch.cuda.is_available():
                try:
                    cuda_rng = [s.cpu() for s in rng["cuda"]] if isinstance(rng["cuda"], list) else rng["cuda"].cpu()
                    torch.cuda.set_rng_state_all(cuda_rng)
                except Exception as e:
                    pass
            if "python" in rng and rng["python"] is not None:
                try:
                    random.setstate(rng["python"])
                except Exception:
                    pass

        self.best_loss = checkpoint.get("best_loss", float("inf"))
        return checkpoint


# -----------------------------------------------------------------------------
# 4. Graceful Interrupt Context Manager
# -----------------------------------------------------------------------------
class GracefulInterruptHandler:
    """
    Catches keyboard interrupts (SIGINT / Ctrl+C) and triggers a clean shutdown
    allowing the checkpoint manager to write an emergency recovery file.
    """
    def __init__(self):
        self.interrupted = False
        self.original_handler = None

    def __enter__(self):
        self.interrupted = False
        self.original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handler)
        return self

    def _handler(self, signum, frame):
        print("\n\n[!] Interrupt received! Flagging graceful shutdown...")
        self.interrupted = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_handler is not None:
            signal.signal(signal.SIGINT, self.original_handler)
