import random
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from dmoe.encoder import UniversalMultimodalEncoder
from dmoe.experts import first_triggered_reason
from dmoe.manifold import GeometricCorrectiveManifold
from dmoe.layer import DynamicMorphicTransformerLayer
from dmoe.decoder import UniversalProjectiveDecoder
from dmoe import checkpoint as checkpoint_module


class SelfStabilizingDynamicMorphicMoE(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        d_ff: int = 128,
        initial_layers: int = 2,
        initial_experts: int = 3,
        theta_up: float = 0.45,
        theta_down: float = 0.05,
        theta_novelty_vertical: float = 2.5,
        min_layers: int = 1,
        max_layers: int = 8,
        enable_simulation_loop: bool = False,
        max_simulation_steps: int = 4,
        theta_converge: float = 0.1,
        min_improvement: float = 0.01,
        skip_threshold: float = 0.0,
        tokenizer_free: bool = False,
        chunk_size: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta_up = theta_up
        self.theta_down = theta_down
        self.theta_novelty_vertical = theta_novelty_vertical
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.enable_simulation_loop = enable_simulation_loop
        self.max_simulation_steps = max_simulation_steps
        self.theta_converge = theta_converge
        self.min_improvement = min_improvement
        self.skip_threshold = skip_threshold
        self.tokenizer_free = tokenizer_free
        self.chunk_size = chunk_size

        self.encoder = UniversalMultimodalEncoder(
            d_model, tokenizer_free=tokenizer_free, chunk_size=chunk_size
        )
        self.manifold_corr = GeometricCorrectiveManifold(d_model)

        self.layers = nn.ModuleList(
            [
                DynamicMorphicTransformerLayer(d_model, num_heads, d_ff, initial_experts=initial_experts)
                for _ in range(initial_layers)
            ]
        )

        self.decoder = UniversalProjectiveDecoder(
            d_model, tokenizer_free=tokenizer_free, chunk_size=chunk_size
        )

        if self.enable_simulation_loop:
            self.loop_damping_logit = nn.Parameter(torch.tensor(0.0))

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @staticmethod
    def _complex_unit_norm(
        z_r: torch.Tensor, z_i: torch.Tensor, eps: float = 1e-5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_r = z_r.mean(dim=-1, keepdim=True)
        mean_i = z_i.mean(dim=-1, keepdim=True)
        cent_r = z_r - mean_r
        cent_i = z_i - mean_i
        std = torch.sqrt((cent_r ** 2 + cent_i ** 2).mean(dim=-1, keepdim=True) + eps)
        return cent_r / std, cent_i / std

    def _run_simulation_loop(
        self, z_r: torch.Tensor, z_i: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.manifold_corr.last_novelty_score.item() <= self.skip_threshold:
            return z_r, z_i, 0

        damping = torch.sigmoid(self.loop_damping_logit)
        steps_used = 0
        max_steps = (
            random.randint(1, self.max_simulation_steps)
            if self.training
            else self.max_simulation_steps
        )
        previous_error = None

        for _ in range(max_steps):
            norm_r, norm_i = self._complex_unit_norm(z_r, z_i)
            corrected_r, corrected_i, structural_error, _ = self.manifold_corr(
                norm_r, norm_i, update_stats=False
            )
            candidate_r, candidate_i, _ = self.layers[-1](
                corrected_r, corrected_i, update_stats=False
            )
            z_r = z_r + damping * (candidate_r - z_r)
            z_i = z_i + damping * (candidate_i - z_i)
            steps_used += 1

            if not self.training:
                current_error = structural_error.item()
                if current_error < self.theta_converge:
                    break
                if previous_error is not None and (previous_error - current_error) < self.min_improvement:
                    break
                previous_error = current_error

        return z_r, z_i, steps_used

    def forward(
        self, x: torch.Tensor, output_head: str = "token", update_stats: bool = True
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor], int]]:
        z_r, z_i = self.encoder(x)
        z_r, z_i, structural_error, novelty_score = self.manifold_corr(
            z_r, z_i, update_stats=update_stats
        )

        entropy_traces = []
        for layer in self.layers:
            z_r, z_i, entropy = layer(z_r, z_i, update_stats=update_stats)
            entropy_traces.append(entropy)

        simulation_steps_used = 0
        if self.enable_simulation_loop:
            z_r, z_i, simulation_steps_used = self._run_simulation_loop(z_r, z_i)

        output, decoder_latent = self.decoder.forward_with_latent(z_r, z_i, head=output_head)

        return {
            "output": output,
            "decoder_latent": decoder_latent,
            "structural_error": structural_error,
            "novelty_score": novelty_score,
            "routing_entropies": entropy_traces,
            "z_real": z_r,
            "z_imag": z_i,
            "simulation_steps_used": simulation_steps_used,
        }

    def execute_graph_mutation(
        self, novelty_score: Optional[float] = None
    ) -> Dict[str, Union[int, List[int], List[str]]]:
        mutation_logs = []
        current_error = self.manifold_corr.running_error_magnitude.item()
        current_novelty = (
            novelty_score if novelty_score is not None else self.manifold_corr.last_novelty_score.item()
        )
        device = next(self.parameters()).device

        vertical_reason = first_triggered_reason(
            [
                (
                    current_error > self.theta_up,
                    f"Structural Error={current_error:.4f} > {self.theta_up:.4f}",
                ),
                (
                    current_novelty > self.theta_novelty_vertical,
                    f"Novelty Z-Score={current_novelty:.4f} > {self.theta_novelty_vertical:.4f}",
                ),
            ]
        )

        if vertical_reason is not None and len(self.layers) < self.max_layers:
            new_layer = DynamicMorphicTransformerLayer(
                self.d_model,
                self.num_heads,
                self.d_ff,
                initial_experts=self.layers[-1].moe.num_experts,
            ).to(device)
            self.layers.append(new_layer)
            mutation_logs.append(
                f"[Vertical Growth] {vertical_reason}. Appended Layer #{len(self.layers)}."
            )

        elif len(self.layers) > self.min_layers:
            salience_values = [l.layer_salience.item() for l in self.layers]
            min_salience, min_idx = min((val, idx) for idx, val in enumerate(salience_values))

            if min_salience < self.theta_down:
                del self.layers[min_idx]
                mutation_logs.append(
                    f"[Vertical Pruning] Layer #{min_idx} salience={min_salience:.4f} < "
                    f"{self.theta_down:.4f}. Collapsed layer."
                )

        for idx, layer in enumerate(self.layers):
            spawned, pruned, log_msg = layer.moe.mutate_topology(novelty_score=current_novelty)
            if spawned or pruned:
                mutation_logs.append(f"[Horizontal Morph Layer {idx}] {log_msg}")

        return {
            "total_layers": len(self.layers),
            "experts_per_layer": [l.moe.num_experts for l in self.layers],
            "mutation_logs": mutation_logs,
        }

    def save_checkpoint(self, path: str) -> None:
        checkpoint_module.save_checkpoint(self, path)

    @classmethod
    def load_checkpoint(cls, path: str, device=None) -> "SelfStabilizingDynamicMorphicMoE":
        return checkpoint_module.load_checkpoint(path, cls, device=device)


def rebuild_optimizer_preserving_state(
    old_optimizer: torch.optim.Optimizer, model: nn.Module, **adamw_kwargs
) -> torch.optim.AdamW:
    id_to_state = {}
    for group in old_optimizer.param_groups:
        for p in group["params"]:
            if p in old_optimizer.state:
                id_to_state[id(p)] = old_optimizer.state[p]

    new_optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    for group in new_optimizer.param_groups:
        for p in group["params"]:
            saved_state = id_to_state.get(id(p))
            if saved_state is not None:
                new_optimizer.state[p] = saved_state

    return new_optimizer
