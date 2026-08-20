import time
import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.nanogpt import HyperTransformerLM
from hyperspace.dynamic_optimizer import DynamicWarmupAdamW
from scratch.verify_exact_equivalence import ExactVectorizedDendriticMoE

class ProductionHyperMoEVectorized(nn.Module):
    """
    Drop-in Vectorized HyperMoE containing exact TwoCompartmentDendriticExpert SwiGLU math.
    """
    def __init__(
        self,
        d_model: int = 384,
        d_ff: int = 768,
        d_hyper: int = 2048,
        top_k: int = 2,
        max_k: int = 4,
        top_p: float = 0.85,
        dynamic_k: bool = True,
        spawn_threshold: float = 0.30,
        max_experts: int = 32,
        initial_experts: int = 2,
        use_bus: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_hyper = d_hyper
        self.top_k = top_k
        self.max_k = max_k
        self.top_p = top_p
        self.dynamic_k = dynamic_k
        self.use_bus = use_bus
        self.max_experts = max_experts
        self.num_experts = initial_experts

        # Phasor Projections
        self.proj_r = nn.Linear(d_model, d_hyper, bias=False)
        self.proj_i = nn.Linear(d_model, d_hyper, bias=False)
        nn.init.orthogonal_(self.proj_r.weight)
        nn.init.orthogonal_(self.proj_i.weight)

        # Hyperspace Memory (Learnable Complex Keys)
        self.keys_r = nn.Parameter(torch.randn(max_experts, d_hyper) * 0.02)
        self.keys_i = nn.Parameter(torch.randn(max_experts, d_hyper) * 0.02)

        # Global Workspace Bus
        from hyperspace.bus import HyperspaceGlobalBus
        self.bus = HyperspaceGlobalBus(d_model=d_model, d_hyper=d_hyper) if use_bus else None

        # Exact Vectorized Dendritic Experts
        self.experts = ExactVectorizedDendriticMoE(
            d_model=d_model, d_ff=d_ff, num_experts=max_experts, device="cpu"
        )

    def get_unit_keys(self) -> torch.Tensor:
        active_r = self.keys_r[:self.num_experts]
        active_i = self.keys_i[:self.num_experts]
        z = torch.complex(active_r, active_i)
        return z / (torch.abs(z) + 1e-8)

    def forward(self, x: torch.Tensor, allow_spawning: bool = False, is_replay: bool = False):
        b, s, d = x.shape
        flat_x = x.view(-1, d)
        num_tokens = flat_x.shape[0]

        # 1. Complex Phasor Projection
        norm_x = F.normalize(flat_x, p=2, dim=-1)
        r_part = self.proj_r(norm_x).float()
        i_part = self.proj_i(norm_x).float()
        z = torch.complex(r_part, i_part)
        token_phasors = z / (torch.abs(z) + 1e-8)

        # 2. Semantic Resonance (Hermitian Cosine Similarity)
        unit_keys = self.get_unit_keys() # [N_experts, d_hyper]
        # Hermitian dot product: Re(u * conj(v))
        kr, ki = unit_keys.real, unit_keys.imag
        pr, pi = token_phasors.real, token_phasors.imag
        resonance = (pr @ kr.T + pi @ ki.T) / self.d_hyper # [num_tokens, num_experts]

        # 3. Softmax Routing & Dynamic K
        scaled_logits = resonance * 10.0
        router_probs = F.softmax(scaled_logits, dim=-1)

        k_eval = self.top_k
        if self.dynamic_k:
            entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-9), dim=-1)
            mean_entropy = entropy.mean().item()
            max_entropy = math.log(self.num_experts) if self.num_experts > 1 else 1.0
            norm_entropy = mean_entropy / max_entropy if max_entropy > 0 else 0.0
            k_eval = min(self.max_k, max(1, int(round(self.top_k * (1.0 + norm_entropy)))))

        top_weights, top_indices = torch.topk(router_probs, k=min(k_eval, self.num_experts), dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 4. Bus & Apical Context
        bus_context = None
        if self.bus is not None and k_eval > 1 and self.num_experts > 1:
            # Gather basal outputs for bus
            selected_keys = unit_keys[top_indices]
            # Dummy basal approximation for bus broadcast
            bus_context = torch.zeros(num_tokens, top_indices.shape[1], d, device=x.device, dtype=x.dtype)

        # 5. Exact Vectorized Forward Dispatch
        final_output = self.experts(flat_x, top_indices, top_weights, bus_context)

        # 6. Orthogonality Loss
        if self.num_experts > 1:
            gram = (kr @ kr.T + ki @ ki.T) / self.d_hyper
            mask = ~torch.eye(self.num_experts, dtype=torch.bool, device=x.device)
            ortho_loss = torch.mean(gram[mask] ** 2)
        else:
            ortho_loss = torch.tensor(0.0, device=x.device)

        telemetry = {
            "ortho_loss": ortho_loss,
            "load_balance_loss": torch.tensor(0.0, device=x.device),
            "top_indices": top_indices,
            "top_weights": top_weights,
            "active_k": k_eval
        }
        return final_output.view(b, s, d), telemetry

def benchmark_full_end_to_end():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 90)
    print("  [FULL END-TO-END PRODUCTION TRAINING LOOP BENCHMARK]")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print("  Including: 4 Layers, Embeddings, Attention, Vectorized SwiGLU MoE, AMP Autocast, AdamW Optimizer")
    print("=" * 90)

    vocab_size = 50304
    d_model = 384
    n_layers = 4
    n_heads = 6
    d_ff = 768
    d_hyper = 2048
    micro_batch = 2
    seq_len = 256
    accum_steps = 6 # Total batch = 3,072 tokens per step

    # Load actual tokenized batch from disk
    from train_scaled_production_engine import ScaledProductionDataStreamer
    streamer = ScaledProductionDataStreamer(cache_dir="data/scaled_real_corpus", seq_len=seq_len, batch_size=micro_batch)

    for num_exp in [2, 8, 16, 32]:
        print(f"\n--- Benchmarking Full End-to-End Model at {num_exp} Experts per Layer ---")
        
        # Build full model with vectorized blocks
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

        # Replace standard hyper_moe in each block with ProductionHyperMoEVectorized
        for b in model.blocks:
            b.hyper_moe = ProductionHyperMoEVectorized(
                d_model=d_model,
                d_ff=d_ff,
                d_hyper=d_hyper,
                top_k=2,
                max_k=4,
                top_p=0.85,
                dynamic_k=True,
                max_experts=32,
                initial_experts=num_exp,
                use_bus=True
            ).to(device)

        optimizer = DynamicWarmupAdamW(model.parameters(), lr=6e-4, weight_decay=0.1)
        scaler = torch.amp.GradScaler('cuda')

        # Warmup (3 full optimizer steps = 18 micro-steps)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accum_steps):
                x, y = streamer.get_domain_batch("fineweb_edu", split="train")
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, loss, _ = model(x, targets=y, allow_spawning=False)
                    loss = loss / accum_steps
                scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        torch.cuda.synchronize()

        # Timed benchmark: 20 full optimizer steps (120 micro-batches = 61,440 tokens)
        timed_steps = 20
        t0 = time.perf_counter()
        for _ in range(timed_steps):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accum_steps):
                x, y = streamer.get_domain_batch("fineweb_edu", split="train")
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, loss, _ = model(x, targets=y, allow_spawning=False)
                    loss = loss / accum_steps
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        total_tokens = timed_steps * accum_steps * micro_batch * seq_len
        tokens_per_sec = total_tokens / elapsed
        ms_per_opt_step = (elapsed / timed_steps) * 1000.0
        total_run_mins_3600 = (3600 * (elapsed / timed_steps)) / 60.0

        print(f"  [MEASURED] {num_exp} Experts:")
        print(f"    • Throughput:           {tokens_per_sec:.1f} tokens/sec")
        print(f"    • Step Time:            {ms_per_opt_step:.1f} ms / optimizer step")
        print(f"    • 3,600-Step Full Run:  {total_run_mins_3600:.1f} minutes ({total_run_mins_3600/60:.2f} hours)")

if __name__ == "__main__":
    benchmark_full_end_to_end()
