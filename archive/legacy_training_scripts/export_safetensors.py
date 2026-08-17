"""
Model Exporter: Converts Hyperspace 2.0 Checkpoints to HuggingFace SafeTensors / Config Bundle.
Generates:
1. model.safetensors (or pytorch_model.bin)
2. config.json (Full model hyperparameter specifications)
3. generation_config.json
"""

import os
import sys
import json
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def export_model_package(ckpt_path: str, output_dir: str = "exported_model"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return False

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = state.get("config", {})
    state_dict = state.get("model_state", state)

    # 1. Export config.json
    hf_config = {
        "architectures": ["HyperTransformerLM"],
        "model_type": "hyperspace_dynamic_moe",
        "vocab_size": 50304,
        "d_model": config.get("d_model", 384),
        "n_layers": config.get("n_layers", 6),
        "n_heads": config.get("n_heads", 6),
        "d_ff": config.get("d_ff", 1024),
        "d_hyper": config.get("d_hyper", 2048),
        "top_k": config.get("top_k", 2),
        "max_experts": config.get("max_experts", 16),
        "spawn_threshold": config.get("spawn_threshold", 0.25),
        "max_position_embeddings": config.get("seq_len", 256) + 32,
        "torch_dtype": "float32",
        "transformers_version": "4.40.0",
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)
    print(f"[SAVED] Exported {config_path}")

    # 2. Export generation_config.json
    gen_config = {
        "temperature": 0.75,
        "top_k": 40,
        "top_p": 0.9,
        "max_new_tokens": 128,
        "do_sample": True,
        "pad_token_id": 50256,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
    }
    gen_config_path = os.path.join(output_dir, "generation_config.json")
    with open(gen_config_path, "w", encoding="utf-8") as f:
        json.dump(gen_config, f, indent=2)
    print(f"[SAVED] Exported {gen_config_path}")

    # 3. Export weights (SafeTensors if available, fallback to pytorch_model.bin)
    try:
        from safetensors.torch import save_file
        st_path = os.path.join(output_dir, "model.safetensors")
        # Ensure tensors are contiguous and on CPU
        cleaned_dict = {}
        for k, v in state_dict.items():
            if torch.is_tensor(v):
                # If complex tensor, split real and imag for safetensors compatibility
                if v.is_complex():
                    cleaned_dict[f"{k}.real"] = v.real.clone().contiguous().cpu()
                    cleaned_dict[f"{k}.imag"] = v.imag.clone().contiguous().cpu()
                else:
                    cleaned_dict[k] = v.clone().contiguous().cpu()
        save_file(cleaned_dict, st_path)
        print(f"[SAVED] Exported SafeTensors weights to {st_path}")
    except ImportError:
        bin_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(state_dict, bin_path)
        print(f"[SAVED] Exported PyTorch binary weights to {bin_path}")

    print(f"\n[EXPORT COMPLETE] Complete model bundle exported to {output_dir}/")
    return True

if __name__ == "__main__":
    ckpt = "experiments/checkpoints/scaled_production_hyperspace.pt"
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    export_model_package(ckpt)
