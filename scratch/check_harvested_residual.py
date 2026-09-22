import sys
sys.path.insert(0, ".")
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from ishmn.slm_adapter import SLMManifoldAdapter
from ishmn.universal_world_model import UniversalWorldModel
from ishmn.geometry import geodesic_distance_sphere
from experiments.train_world_model_on_layer14 import harvest_layer14_transitions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
slm = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-1.5B-Instruct',
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map='auto' if torch.cuda.is_available() else None
)
adapter = SLMManifoldAdapter(hidden_size=1536, d_model=32, hyperspace_dim=8, device=device)
adapter.load_weights('artifacts/slm_manifold_adapter.pt', device=device)
adapter.eval()

brain = torch.load('artifacts/unified_lifelong_brain.pt', map_location='cpu', weights_only=False)
wm = UniversalWorldModel(d_model=32, hyperspace_dim=8, device=device)
wm_sd = {k.replace('world_model.', ''): v for k, v in brain['state_dict'].items() if k.startswith('world_model.')}
wm.load_state_dict(wm_sd, strict=False)
wm.eval()

gsm_val = load_dataset('gsm8k', 'main', split='test[:5]')
val_texts = [f"Question: {item['question']}\nThought: {item['answer']}" for item in gsm_val]

val_curr, val_next, val_ctx = harvest_layer14_transitions(tokenizer, slm, adapter, val_texts, max_len=128)

with torch.no_grad():
    res = wm.causal_residual(val_curr, val_ctx)
    print(f"Harvested {val_curr.shape[0]} tokens.")
    print(f"Mean causal residual on harvested validation: {res.mean().item():.4f} rad")
    print(f"Median causal residual: {res.median().item():.4f} rad")
    print(f"Min causal residual: {res.min().item():.4f} rad")
    print(f"Max causal residual: {res.max().item():.4f} rad")
    print(f"Fraction < 0.35 rad: {(res < 0.35).float().mean().item() * 100:.2f}%")
