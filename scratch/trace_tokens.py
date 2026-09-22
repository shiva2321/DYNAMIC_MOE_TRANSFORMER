import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ishmn import GroundedHybridEngine

engine = GroundedHybridEngine(brain_path="artifacts/unified_lifelong_brain.pt")
prompt = "In physics, what happens when light passes from air into water?"
messages = [{"role": "user", "content": prompt}]
formatted = engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = engine.tokenizer([formatted], return_tensors="pt").to(engine.slm.device)

print("Testing ungrounded generation (baseline):")
engine.sandwich.is_active = False
out_base = engine.slm.generate(**inputs, max_new_tokens=40, do_sample=False)
print("Baseline text:")
print(engine.tokenizer.decode(out_base[0][len(inputs.input_ids[0]):], skip_special_tokens=True))

print("\nTesting grounded generation token by token:")
engine.sandwich.is_active = True
engine.sandwich.reset_telemetry()

# Let's inspect each step of generation
curr_ids = inputs.input_ids
past_key_values = None

for step in range(10):
    with torch.no_grad():
        if past_key_values is None:
            model_inputs = {"input_ids": curr_ids, "use_cache": True}
        else:
            model_inputs = {"input_ids": curr_ids[:, -1:], "past_key_values": past_key_values, "use_cache": True}
        
        out = engine.slm(**model_inputs)
        next_token_logits = out.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        past_key_values = out.past_key_values
        curr_ids = torch.cat([curr_ids, next_token], dim=1)
        token_str = engine.tokenizer.decode(next_token[0])
        telemetry = engine.sandwich.deliberation_telemetry[-1] if engine.sandwich.deliberation_telemetry else {}
        print(f"Step {step}: token='{token_str}' (id={next_token.item()}), steps={telemetry.get('step_count')}, err={telemetry.get('causal_error')}, surprise={telemetry.get('surprise'):.3f}")
        if telemetry.get('causal_error') != telemetry.get('causal_error'): # is nan
            print("NaN detected!")
            break
