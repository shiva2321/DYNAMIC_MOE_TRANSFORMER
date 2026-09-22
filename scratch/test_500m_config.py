import sys
sys.path.insert(0, ".")
import torch
from ishmn.native_cognitive_transformer import NativeSphericalTransformer
from ishmn.universal_cognitive_engine import UniversalCognitiveEngine

configs = [
    {'d_model': 384, 'num_layers': 6, 'num_heads': 6, 'd_ff': 768, 'num_experts': 16, 'hyperspace_dim': 16},
    {'d_model': 384, 'num_layers': 8, 'num_heads': 6, 'd_ff': 768, 'num_experts': 16, 'hyperspace_dim': 16},
    {'d_model': 512, 'num_layers': 6, 'num_heads': 8, 'd_ff': 1024, 'num_experts': 12, 'hyperspace_dim': 16},
    {'d_model': 512, 'num_layers': 6, 'num_heads': 8, 'd_ff': 1024, 'num_experts': 14, 'hyperspace_dim': 16},
    {'d_model': 384, 'num_layers': 8, 'num_heads': 6, 'd_ff': 768, 'num_experts': 20, 'hyperspace_dim': 16},
]


for cfg in configs:
    model = NativeSphericalTransformer(
        vocab_size=50257,
        d_model=cfg['d_model'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        hyperspace_dim=cfg['hyperspace_dim'],
        d_ff=cfg['d_ff'],
        max_seq_len=256,
        mtp_future_tokens=2,
        vsa_capacity=512,
        num_experts=cfg['num_experts'],
        top_k=2,
        enable_shared_expert=True,
        enable_aux_loss_free_balancing=True,
        enable_vsa_memory=True,
        enable_world_model=True
    )
    uce = UniversalCognitiveEngine(
        d_model=cfg['d_model'],
        d_hidden=cfg['d_ff'],
        hyperspace_dim=cfg['hyperspace_dim'],
        num_memory_slots=32,
        vocab_size=50257,
        initial_experts=cfg['num_experts']
    )
    uce.attach_native_transformer(model)
    total = sum(p.numel() for p in uce.parameters())
    trans = sum(p.numel() for p in model.parameters())
    cog = total - trans
    print(f"L={cfg['num_layers']}, D={cfg['d_model']}, E={cfg['num_experts']} -> Total: {total/1e6:.1f}M (Trans: {trans/1e6:.1f}M, UCE: {cog/1e6:.1f}M)")
    del uce, model
