import sys
sys.path.insert(0, ".")
import torch
from ishmn.universal_cognitive_engine import UniversalCognitiveEngine
from ishmn.native_cognitive_transformer import NativeSphericalTransformer
from ishmn.data_loader import SovereignTokenizer
from ishmn.training_engine import MODEL_PROFILES

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load('artifacts/checkpoints/native_moe_acc100.pt', map_location=device, weights_only=False)
cfg = MODEL_PROFILES['nano']
tokenizer = SovereignTokenizer('gpt2')

model = NativeSphericalTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=cfg['d_model'],
    num_layers=cfg['num_layers'],
    num_heads=cfg['num_heads'],
    hyperspace_dim=cfg['hyperspace_dim'],
    d_ff=cfg['d_ff'],
    max_seq_len=96,
    mtp_future_tokens=cfg['mtp_future_tokens'],
    vsa_capacity=cfg['vsa_capacity'],
    num_experts=cfg['num_experts'],
    top_k=cfg['top_k'],
    enable_shared_expert=cfg['enable_shared_expert'],
    enable_aux_loss_free_balancing=cfg['enable_aux_loss_free_balancing'],
    enable_vsa_memory=True,
    enable_world_model=True,
    device=device
).to(device)

uce = UniversalCognitiveEngine(
    d_model=cfg['d_model'],
    d_hidden=cfg['d_ff'],
    hyperspace_dim=cfg['hyperspace_dim'],
    num_memory_slots=32,
    vocab_size=tokenizer.vocab_size,
    initial_experts=cfg['num_experts'],
    device=device
).to(device)
uce.attach_native_transformer(model, tokenizer=tokenizer)
uce.load_state_dict(ckpt['uce_state_dict'])
model.load_state_dict(ckpt['model_state_dict'])

prompt = 'Calculate: 37 * 6 = '
res = uce.think_and_speak(prompt, max_new_tokens=10, temperature=0.1)
print('Response:', repr(res['response']))
print('System2 proof:', res['system2_proof'])
