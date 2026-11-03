from typing import Optional

import torch
import torch.nn as nn

from dmoe.layer import DynamicMorphicTransformerLayer


def save_checkpoint(model: nn.Module, path: str) -> None:
    config = {
        "d_model": model.d_model,
        "num_heads": model.num_heads,
        "d_ff": model.d_ff,
        "theta_up": model.theta_up,
        "theta_down": model.theta_down,
        "min_layers": model.min_layers,
        "max_layers": model.max_layers,
    }
    experts_per_layer = [layer.moe.num_experts for layer in model.layers]
    numeric_projector_dims = [int(k) for k in model.encoder.numeric_projectors.keys()]
    torch.save(
        {
            "config": config,
            "experts_per_layer": experts_per_layer,
            "numeric_projector_dims": numeric_projector_dims,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str, model_cls, device: Optional[torch.device] = None) -> nn.Module:
    payload = torch.load(path, map_location=device or "cpu", weights_only=True)
    config = payload["config"]
    experts_per_layer = payload["experts_per_layer"]
    numeric_projector_dims = payload.get("numeric_projector_dims", [])

    model = model_cls(
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        initial_layers=len(experts_per_layer),
        initial_experts=experts_per_layer[0] if experts_per_layer else 1,
        theta_up=config["theta_up"],
        theta_down=config["theta_down"],
        min_layers=config["min_layers"],
        max_layers=config["max_layers"],
    )

    rebuilt_layers = nn.ModuleList(
        [
            DynamicMorphicTransformerLayer(config["d_model"], config["num_heads"], config["d_ff"], initial_experts=n)
            for n in experts_per_layer
        ]
    )
    model.layers = rebuilt_layers
    for f_dim in numeric_projector_dims:
        model.encoder._get_numeric_projector(f_dim, torch.device("cpu"))
    model.load_state_dict(payload["state_dict"])

    if device is not None:
        model = model.to(device)

    return model

