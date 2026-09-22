"""
Live demonstration of native sequence generation using the Looped Riemannian Transformer.
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ishmn import UniversalCognitiveEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize native UCE
uce = UniversalCognitiveEngine(
    d_model=32,
    d_hidden=64,
    hyperspace_dim=8,
    num_memory_slots=32,
    vocab_size=128,
    initial_experts=2,
    device=device
)

# Prompt with ASCII token bytes for "Sun orbits"
prompt_str = "Sun orbits"
prompt_tokens = torch.tensor([[ord(c) for c in prompt_str]], device=device)

print(f"[*] Prompt string : '{prompt_str}'")
print(f"[*] Prompt tokens : {prompt_tokens.tolist()[0]}")

# Generate 8 tokens autoregressively with Looped Riemannian Transformer
res = uce.solve_sequence_autoregressive(
    prompt_tokens=prompt_tokens,
    max_gen_tokens=8,
    max_loops=4,
    temperature=0.8,
    sample=True
)

gen_tokens = res["generated_tokens"][0].tolist()
full_seq = res["full_sequence"][0].tolist()

print(f"[+] Generated token IDs : {gen_tokens}")
gen_chars = [chr(t) if 32 <= t < 127 else f"<{t}>" for t in gen_tokens]
print(f"[+] Generated characters: {''.join(gen_chars)}")
print(f"[+] Average thinking loops per token: {res['avg_steps_per_token']:.2f}")
