from typing import Dict, List, Union

import torch
import torch.nn as nn

from dmoe.encoder import UniversalMultimodalEncoder
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
        min_layers: int = 1,
        max_layers: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta_up = theta_up
        self.theta_down = theta_down
        self.min_layers = min_layers
        self.max_layers = max_layers

        self.encoder = UniversalMultimodalEncoder(d_model)
        self.manifold_corr = GeometricCorrectiveManifold(d_model)

        self.layers = nn.ModuleList(
            [
                DynamicMorphicTransformerLayer(d_model, num_heads, d_ff, initial_experts=initial_experts)
                for _ in range(initial_layers)
            ]
        )

        self.decoder = UniversalProjectiveDecoder(d_model)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def forward(self, x: torch.Tensor, output_head: str = "token") -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        z_r, z_i = self.encoder(x)
        z_r, z_i, structural_error = self.manifold_corr(z_r, z_i)

        entropy_traces = []
        for layer in self.layers:
            z_r, z_i, entropy = layer(z_r, z_i)
            entropy_traces.append(entropy)

        output = self.decoder(z_r, z_i, head=output_head)

        return {
            "output": output,
            "structural_error": structural_error,
            "routing_entropies": entropy_traces,
            "z_real": z_r,
            "z_imag": z_i,
        }

    def execute_graph_mutation(self) -> Dict[str, Union[int, List[int], List[str]]]:
        mutation_logs = []
        current_error = self.manifold_corr.running_error_magnitude.item()
        device = next(self.parameters()).device

        if current_error > self.theta_up and len(self.layers) < self.max_layers:
            new_layer = DynamicMorphicTransformerLayer(
                self.d_model,
                self.num_heads,
                self.d_ff,
                initial_experts=self.layers[-1].moe.num_experts,
            ).to(device)
            self.layers.append(new_layer)
            mutation_logs.append(
                f"[Vertical Growth] Structural Error={current_error:.4f} > {self.theta_up:.4f}. "
                f"Appended Layer #{len(self.layers)}."
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
            spawned, pruned, log_msg = layer.moe.mutate_topology()
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
