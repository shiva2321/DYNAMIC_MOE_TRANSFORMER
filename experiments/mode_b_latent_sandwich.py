"""
Mode B: The Latent Reasoning Sandwich.
Deep Latent Feature Binding between Small Language Model (Qwen2.5-1.5B) and Universal Cognitive Engine.

Architecture:
1. Intercepts hidden states at intermediate layer (Layer 14 of 28).
2. Projects h in R^1536 -> z in R^32 onto Riemannian Manifold S^7 x R+.
3. Runs UCE deliberate_and_halting_loop with Adaptive Computation Time (ACT 1..6 ticks)
   leveraging 7 Dynamic MoE Specialists and HRPC Predictive Coding.
4. Projects settled state z* in R^32 -> Delta h in R^1536.
5. Injects residually into Layer 15: h_next = h + alpha * Delta h.
"""

import sys
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ishmn import UniversalCognitiveEngine

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
BRAIN_PATH = "artifacts/unified_lifelong_brain.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INTERMEDIATE_LAYER_IDX = 14  # Middle layer of 28 layers


class LatentManifoldSandwich(nn.Module):
    def __init__(
        self,
        uce: UniversalCognitiveEngine,
        hidden_size: int = 1536,
        d_model: int = 32,
        alpha: float = 0.20,
        device: torch.device = DEVICE
    ):
        super().__init__()
        self.uce = uce
        self.hidden_size = hidden_size
        self.d_model = d_model
        self.device = device
        self.alpha = alpha

        # Down-projection: LLM hidden space -> UCE manifold latent space
        self.down_proj = nn.Linear(hidden_size, d_model, bias=False, device=self.device)
        # Up-projection: UCE settled manifold state -> LLM hidden space
        self.up_proj = nn.Linear(d_model, hidden_size, bias=False, device=self.device)

        # Orthogonal initialization for stable projection
        nn.init.orthogonal_(self.down_proj.weight)
        nn.init.orthogonal_(self.up_proj.weight)
        self.up_proj.weight.data *= 0.15  # Controlled initial injection amplitude

        self.hook_handle = None
        self.deliberation_telemetry = []
        self.is_active = True

    def hook_fn(self, module, input_tuple, output):
        """Forward hook executed at Layer 14."""
        if not self.is_active:
            return output

        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output

        if h.dim() == 2:
            h_3d = h.unsqueeze(1)
            was_2d = True
        else:
            h_3d = h
            was_2d = False

        B, L, H = h_3d.shape

        # Focus deliberation on the current generation token (last token in sequence)
        h_last = h_3d[:, -1:, :].float().to(self.device)  # [B, 1, H]
        h_context = h_3d.mean(dim=1, keepdim=True).float().to(self.device)  # [B, 1, H]

        with torch.no_grad():
            # 1. Project into UCE manifold latent space
            z_state = self.down_proj(h_last).squeeze(1)       # [B, d_model]
            z_context = self.down_proj(h_context).squeeze(1)   # [B, d_model]

            # 2. Execute Adaptive Computation Time (ACT) on Riemannian Manifold
            delib = self.uce.world_model.deliberate_and_halting_loop(
                z_init=z_state,
                context=z_context,
                moe_layer=self.uce.moe
            )

            z_settled = delib["final_latent"]  # [B, d_model]
            steps = delib["steps_taken"]
            consensus = delib["consensus"]
            causal_err = delib["causal_error"]

            # Record cognitive telemetry
            self.deliberation_telemetry.append({
                "step_count": steps,
                "consensus": consensus,
                "causal_error": causal_err,
                "system": delib["system_used"]
            })

            # 3. Project settled manifold state back to LLM hidden space
            delta_h = self.up_proj(z_settled.unsqueeze(1)).to(h.dtype)  # [B, 1, H]

        # 4. Residual Injection into Layer 15
        h_modified = h_3d.clone()
        h_modified[:, -1:, :] = h_3d[:, -1:, :] + self.alpha * delta_h
        h_final = h_modified.squeeze(1) if was_2d else h_modified

        if is_tuple:
            return (h_final,) + output[1:]
        else:
            return h_final

    def attach(self, model):
        """Attaches the sandwich hook to the intermediate layer."""
        target_layer = model.model.layers[INTERMEDIATE_LAYER_IDX]
        self.hook_handle = target_layer.register_forward_hook(self.hook_fn)
        print(f"[*] Latent Manifold Sandwich attached to Layer {INTERMEDIATE_LAYER_IDX} of 28.")

    def detach(self):
        """Detaches the sandwich hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            print("[*] Latent Manifold Sandwich detached.")

    def reset_telemetry(self):
        self.deliberation_telemetry = []


def run_mode_b_evaluation():
    print("\n" + "="*80)
    print(" MODE B EMPIRICAL EVALUATION BATTERY: THE LATENT REASONING SANDWICH")
    print("="*80)

    # 1. Load SLM
    print(f"[*] Loading SLM: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    slm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    if not torch.cuda.is_available():
        slm.to(DEVICE)
    slm.eval()

    # 2. Load UCE
    print(f"[*] Loading UCE Brain from {BRAIN_PATH}...")
    uce = UniversalCognitiveEngine.load_brain(BRAIN_PATH, device=DEVICE)
    uce.eval()

    # 3. Initialize Sandwich
    sandwich = LatentManifoldSandwich(uce=uce, hidden_size=1536, d_model=32, alpha=0.25, device=DEVICE)

    test_prompts = [
        (
            "Causal Thermodynamics",
            "Explain in 2 precise sentences what happens when liquid nitrogen at -196 C is poured directly onto a white-hot glowing iron anvil."
        ),
        (
            "Topological Equivalence",
            "Why is a standard ceramic coffee mug topologically equivalent to a donut (torus)? Explain in 2 sentences."
        ),
        (
            "Contradiction Refutation",
            "Can a frictionless pendulum swing higher than its initial release height in a closed vacuum chamber without external energy? Explain."
        )
    ]

    for category, prompt in test_prompts:
        print(f"\n" + "-"*80)
        print(f"PROMPT CATEGORY: {category}")
        print(f"Prompt: \"{prompt}\"")

        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([formatted], return_tensors="pt").to(slm.device)

        # --- A. Baseline SLM (Without Manifold Deliberation) ---
        sandwich.is_active = False
        t0 = time.time()
        with torch.no_grad():
            out_base = slm.generate(**inputs, max_new_tokens=90, do_sample=False)
        t_base = time.time() - t0
        base_text = tokenizer.decode(out_base[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

        # --- B. Latent Manifold Sandwich Active ---
        sandwich.is_active = True
        sandwich.attach(slm)
        sandwich.reset_telemetry()
        t0 = time.time()
        with torch.no_grad():
            out_sand = slm.generate(**inputs, max_new_tokens=90, do_sample=False)
        t_sand = time.time() - t0
        sand_text = tokenizer.decode(out_sand[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
        sandwich.detach()

        # Analyze Manifold Thinking Depth
        telemetry = sandwich.deliberation_telemetry
        if telemetry:
            step_counts = [t["step_count"] for t in telemetry]
            causal_errors = [t["causal_error"] for t in telemetry]
            consensuses = [t["consensus"] for t in telemetry]
            
            s1_count = sum(1 for s in step_counts if s == 1)
            s2_count = sum(1 for s in step_counts if s > 1)
            avg_steps = np.mean(step_counts)
            avg_causal_err = np.mean(causal_errors)
            avg_consensus = np.mean(consensuses)
        else:
            avg_steps = 0
            s1_count = 0
            s2_count = 0
            avg_causal_err = 0.0
            avg_consensus = 0.0

        print(f"\n[Baseline SLM Output] (Elapsed: {t_base:.2f}s):")
        print(f"  \"{base_text}\"")

        print(f"\n[Latent Manifold Sandwich Output] (Elapsed: {t_sand:.2f}s):")
        print(f"  \"{sand_text}\"")

        print(f"\n[Manifold Deliberation Telemetry (Layer 14)]:")
        print(f"  Total Token Forward Passes : {len(telemetry)}")
        print(f"  System 1 Instant Reflex (t=1): {s1_count} tokens ({s1_count/max(len(telemetry),1)*100:.1f}%)")
        print(f"  System 2 Deliberations (t>1) : {s2_count} tokens ({s2_count/max(len(telemetry),1)*100:.1f}%)")
        print(f"  Average Thinking Ticks       : {avg_steps:.2f} ticks / token")
        print(f"  Mean Causal Rollback Error   : {avg_causal_err:.4f} rad")
        print(f"  Mean Phase Halting Consensus : {avg_consensus:.4f}")

    print("\n[+] Mode B Evaluation Completed Successfully.")


if __name__ == "__main__":
    run_mode_b_evaluation()
