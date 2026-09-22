import sys
sys.path.insert(0, ".")
import torch
from ishmn.data_loader import SovereignTokenizer, HuggingFaceCurriculumStreamer
from ishmn.honest_eval_harness import run_honest_evaluation_suite
from ishmn import UniversalCognitiveEngine
from ishmn.native_cognitive_transformer import NativeSphericalTransformer
from ishmn.training_engine import MODEL_PROFILES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = MODEL_PROFILES["micro"]
tok = SovereignTokenizer("gpt2")

m = NativeSphericalTransformer(
    vocab_size=tok.vocab_size,
    d_model=cfg["d_model"],
    num_layers=cfg["num_layers"],
    num_heads=cfg["num_heads"],
    hyperspace_dim=cfg["hyperspace_dim"],
    d_ff=cfg["d_ff"],
    max_seq_len=96,
    mtp_future_tokens=cfg["mtp_future_tokens"],
    vsa_capacity=cfg["vsa_capacity"],
    num_experts=cfg["num_experts"],
    top_k=cfg["top_k"],
    max_experts=cfg["num_experts"],
    enable_shared_expert=cfg["enable_shared_expert"],
    enable_aux_loss_free_balancing=cfg["enable_aux_loss_free_balancing"],
    enable_gwt_coalition=cfg.get("enable_gwt_coalition", True),
    enable_vsa_memory=True,
    enable_world_model=False,
    tie_weights=True,
    device=DEVICE
).to(DEVICE)

u = UniversalCognitiveEngine(
    d_model=cfg["d_model"],
    d_hidden=cfg["d_ff"],
    hyperspace_dim=cfg["hyperspace_dim"],
    num_memory_slots=32,
    vocab_size=tok.vocab_size,
    initial_experts=cfg["num_experts"],
    device=DEVICE
).to(DEVICE)
u.attach_native_transformer(m, tokenizer=tok)

ckpt = torch.load("artifacts/checkpoints/native_moe_micro_latest.pt", map_location=DEVICE)
m.load_state_dict(ckpt["model_state_dict"])
u.load_state_dict(ckpt["uce_state_dict"])
print("Loaded step from checkpoint:", ckpt["step"])

s = HuggingFaceCurriculumStreamer(tok, batch_size=8, seq_len=64, device=DEVICE)

for trial in range(4):
    vlist = []
    for _ in range(8):
        inp_v, _ = s.sample_batch(device=DEVICE)
        vlist.extend(inp_v[0].tolist())
    vt = torch.tensor(vlist, dtype=torch.long, device=DEVICE)
    ev = run_honest_evaluation_suite(m, u, tok, held_out_tokens=vt, device=DEVICE)
    loss = ev["held_out_language_perplexity"]["val_loss"]
    ppl = ev["held_out_language_perplexity"]["perplexity"]
    print(f"Trial {trial}: loss={loss:.4f}, perplexity={ppl:.2f}")

s.close()
