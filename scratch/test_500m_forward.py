import sys
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
from ishmn.native_cognitive_transformer import NativeSphericalTransformer
from ishmn.universal_cognitive_engine import UniversalCognitiveEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = {
    'd_model': 512,
    'num_layers': 6,
    'num_heads': 8,
    'd_ff': 1024,
    'num_experts': 15,
    'top_k': 2,
    'hyperspace_dim': 16,
    'max_seq_len': 256
}

model = NativeSphericalTransformer(
    vocab_size=50257,
    d_model=cfg['d_model'],
    num_layers=cfg['num_layers'],
    num_heads=cfg['num_heads'],
    hyperspace_dim=cfg['hyperspace_dim'],
    d_ff=cfg['d_ff'],
    max_seq_len=cfg['max_seq_len'],
    mtp_future_tokens=2,
    vsa_capacity=512,
    num_experts=cfg['num_experts'],
    top_k=cfg['top_k'],
    enable_shared_expert=True,
    enable_aux_loss_free_balancing=True,
    enable_vsa_memory=True,
    enable_world_model=True,
    device=device
).to(device)

uce = UniversalCognitiveEngine(
    d_model=cfg['d_model'],
    d_hidden=cfg['d_ff'],
    hyperspace_dim=cfg['hyperspace_dim'],
    num_memory_slots=32,
    vocab_size=50257,
    initial_experts=cfg['num_experts'],
    device=device
).to(device)
uce.attach_native_transformer(model)

inp = torch.randint(0, 50257, (2, 64), device=device)
tgt = torch.randint(0, 50257, (2, 64), device=device)

# Forward pass with torch.autocast
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out = model(inp, targets=tgt, return_mtp=True)
    loss = out["loss"]

print(f"BF16 Autocast Forward Pass Successful! Loss: {loss.item():.4f}")
print(f"Peak VRAM during forward pass: {torch.cuda.max_memory_allocated() / (1024 ** 2):.1f} MB")
