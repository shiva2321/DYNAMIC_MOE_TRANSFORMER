import sys
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
from ishmn.native_cognitive_transformer import NativeSphericalTransformer
from ishmn.universal_cognitive_engine import UniversalCognitiveEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

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

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
mem_start = torch.cuda.memory_allocated() / (1024 ** 2)

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

total_p = sum(p.numel() for p in uce.parameters())
mem_fp32 = torch.cuda.memory_allocated() / (1024 ** 2)
print(f"Total Parameters: {total_p:,} ({total_p/1e6:.1f}M)")
print(f"FP32 Allocated VRAM: {mem_fp32:.1f} MB")

# Now convert to BF16 or FP16
uce.to(dtype=torch.bfloat16)
model.to(dtype=torch.bfloat16)
mem_bf16 = torch.cuda.memory_allocated() / (1024 ** 2)
print(f"BF16 Allocated VRAM: {mem_bf16:.1f} MB (Saved {mem_fp32 - mem_bf16:.1f} MB!)")
