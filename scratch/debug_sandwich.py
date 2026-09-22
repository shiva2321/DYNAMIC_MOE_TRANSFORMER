import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ishmn import GroundedHybridEngine

engine = GroundedHybridEngine(brain_path="artifacts/unified_lifelong_brain.pt")
prompt = "In physics, what happens when light passes from air into water?"
messages = [{"role": "user", "content": prompt}]
formatted = engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = engine.tokenizer([formatted], return_tensors="pt").to(engine.slm.device)

# Check 1 step of generation
with torch.no_grad():
    outputs = engine.slm(**inputs, output_hidden_states=True)
    h14 = outputs.hidden_states[14]
    print(f"h14 shape: {h14.shape}, min: {h14.min()}, max: {h14.max()}, has_nan: {torch.isnan(h14).any()}")
    
    z = engine.sandwich.down_proj(h14[:, -1:, :].float()).squeeze(1)
    print(f"z shape: {z.shape}, min: {z.min()}, max: {z.max()}, has_nan: {torch.isnan(z).any()}")
    
    delib = engine.uce.world_model.deliberate_and_halting_loop(z_init=z, context=z, moe_layer=engine.uce.moe)
    print(f"delib: steps={delib['steps_taken']}, err={delib['causal_error']}, z_final has_nan={torch.isnan(delib['final_latent']).any()}")
    
    delta_h = engine.sandwich.up_proj(delib['final_latent'].unsqueeze(1)).to(h14.dtype)
    print(f"delta_h shape: {delta_h.shape}, min: {delta_h.min()}, max: {delta_h.max()}, has_nan={torch.isnan(delta_h).any()}")
    
    h_mod = h14[:, -1:, :] + engine.sandwich.alpha * delta_h
    print(f"h_mod min: {h_mod.min()}, max: {h_mod.max()}")
