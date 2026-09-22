import sys
import math
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ishmn import UniversalCognitiveEngine

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# PATH B: LOG-RADIAL SCALE-EQUIVARIANT CONTINUOUS HYPERSPACE
# =========================================================================
print("=" * 70)
print("TESTING PATH B: SCALE-EQUIVARIANT LOG-RADIAL CONTINUOUS MANIFOLD")
print("=" * 70)

def encode_log(x_val):
    s = 1.0 if x_val > 0 else (-1.0 if x_val < 0 else 0.0)
    log_m = math.log1p(abs(x_val))
    return [s, log_m]

def decode_log(s, log_m):
    return s * math.expm1(max(min(log_m, 70.0), 0.0))

# Test encode/decode on user numbers:
user_cases = [
    (23146484561521.0, "+", 45132030.0, 23146529693551.0),
    (23164.0, "-", 45.0, 23119.0),
    (36.0, "*", 8.0, 288.0),
    (24.0, "*", 2.0, 48.0),
    (9.5, "-", 4.5, 5.0),
    (2.0, "*", 3.0, 6.0),
    (12.0, "+", 7.0, 19.0),
]

print("\n" + "=" * 70)
print("TRAINING NEURAL ENGINE IN LOG-RADIAL SPACE ACROSS VAST DYNAMIC RANGE")
print("=" * 70)

# Generate diverse dataset spanning 1 to 10^6
tr_a, tr_op, tr_b, tr_y = [], [], [], []

for _ in range(600):
    # Multi-scale sampling: uniform in log space (from 1 to 1,000,000)
    mag_a = 10 ** (random.uniform(0.0, 5.5))
    mag_b = 10 ** (random.uniform(0.0, 5.5))
    op = random.choice([0, 1, 2])  # +, -, *
    if op == 0:
        y = mag_a + mag_b
    elif op == 1:
        if mag_a < mag_b:
            mag_a, mag_b = mag_b, mag_a
        y = mag_a - mag_b
    else:
        # Small multiplication so products don't exceed 10^7
        mag_a = 10 ** (random.uniform(0.0, 3.5))
        mag_b = 10 ** (random.uniform(0.0, 3.5))
        y = mag_a * mag_b
    tr_a.append(mag_a)
    tr_op.append(op)
    tr_b.append(mag_b)
    tr_y.append(y)

N = len(tr_a)
x_in = torch.zeros(N, 2, device=DEVICE)
ctx_in = torch.zeros(N, 6, device=DEVICE)
y_tgt = torch.zeros(N, 2, device=DEVICE)

for i in range(N):
    sa, lma = encode_log(tr_a[i])
    sb, lmb = encode_log(tr_b[i])
    sy, lmy = encode_log(tr_y[i])
    x_in[i, 0] = sa
    x_in[i, 1] = lma / 20.0  # Normalize log-magnitude by 20.0 (covers up to e^20 ~ 500 million!)
    ctx_in[i, tr_op[i]] = 1.0
    ctx_in[i, 4] = sb
    ctx_in[i, 5] = lmb / 20.0
    y_tgt[i, 0] = sy
    y_tgt[i, 1] = lmy / 20.0

# 2-layer MLP model
model = nn.Sequential(
    nn.Linear(2 + 6, 64),
    nn.SiLU(),
    nn.Linear(64, 64),
    nn.SiLU(),
    nn.Linear(64, 2)
).to(DEVICE)

opt = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
for ep in range(1, 41):
    perm = torch.randperm(N)
    for b in range(0, N, 32):
        idx = perm[b:b+32]
        pred = model(torch.cat([x_in[idx], ctx_in[idx]], dim=-1))
        loss = F.mse_loss(pred, y_tgt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    if ep % 10 == 0:
        print(f"Ep {ep:2d} | Loss: {loss.item():.5f}")

model.eval()
print("\nTesting on User Test Cases in Log-Radial Space:")
for a, op, b, y_true in user_cases:
    op_code = 0 if op == "+" else (1 if op == "-" else 2)
    sa, lma = encode_log(a)
    sb, lmb = encode_log(b)
    inp = torch.tensor([[sa, lma / 20.0, 1.0 if op_code==0 else 0.0, 1.0 if op_code==1 else 0.0, 1.0 if op_code==2 else 0.0, 0.0, sb, lmb / 20.0]], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        out = model(inp)[0]
        pred_s = 1.0 if out[0].item() >= 0 else -1.0
        pred_lm = max(out[1].item() * 20.0, 0.0)
        pred_y = decode_log(pred_s, pred_lm)
        rel_err = abs(pred_y - y_true) / max(abs(y_true), 1.0) * 100.0
print("\n" + "=" * 70)
print("TESTING PATH A: UNIVERSAL SEQUENCE DELIBERATION (TOKEN REASONER)")
print("=" * 70)

# Build character vocabulary
chars = list("0123456789+-*= \n")
char2id = {c: i for i, c in enumerate(chars)}
id2char = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)

def text_to_tensor(text):
    return torch.tensor([[char2id[c] for c in text]], dtype=torch.long, device=DEVICE)

def tensor_to_text(tensor):
    return "".join([id2char[int(i)] for i in tensor.flatten().cpu().tolist()])

print(f"Token Vocab: {vocab_size} ASCII characters: {chars}")

# Create small sequence transformer
uce_seq = UniversalCognitiveEngine(
    d_model=32,
    d_hidden=64,
    hyperspace_dim=8,
    num_memory_slots=16,
    vocab_size=vocab_size,
    initial_experts=1,
    max_experts=4,
    lr=0.005,
    device=DEVICE
)

# Generate algorithmic training strings: reversed digits for natural carry flow
# e.g., "5+3=8\n", "12+7=19\n"
train_seqs = []
for a in range(1, 30):
    for b in range(1, 30):
        train_seqs.append(f"{a}+{b}={a+b}\n")
        if a >= b:
            train_seqs.append(f"{a}-{b}={a-b}\n")
for a in range(1, 10):
    for b in range(1, 10):
        train_seqs.append(f"{a}*{b}={a*b}\n")

print(f"Training on {len(train_seqs)} algorithmic equation sequences...")

uce_seq.train()
optimizer = torch.optim.AdamW(
    list(uce_seq.perceptor.parameters()) +
    list(uce_seq.token_to_dmodel.parameters()) +
    list(uce_seq.world_model.deliberation_loop.parameters()) +
    list(uce_seq.symbolic_head.parameters()),
    lr=0.005
)

# Train on padded batches
max_len = max(len(s) for s in train_seqs)
pad_id = char2id[" "]

for ep in range(1, 21):
    random.shuffle(train_seqs)
    total_loss = 0.0
    steps = 0
    batch_size = 32
    for start in range(0, len(train_seqs), batch_size):
        batch = train_seqs[start:start+batch_size]
        padded = []
        for s in batch:
            ids = [char2id[c] for c in s]
            ids = ids + [pad_id] * (max_len - len(ids))
            padded.append(ids)
        tokens = torch.tensor(padded, dtype=torch.long, device=DEVICE)
        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]

        optimizer.zero_grad()
        embeds = uce_seq.perceptor.token_embed(inp)
        h = uce_seq.token_to_dmodel(embeds)
        h_out, _ = uce_seq.world_model.deliberation_loop(h, max_steps=2)
        logits = uce_seq.symbolic_head(h_out)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1), ignore_index=pad_id)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        steps += 1
    if ep % 5 == 0:
        print(f"Epoch {ep:2d} | CrossEntropy: {total_loss/steps:.4f}")

uce_seq.eval()
print("\nTesting Sequence Generation:")
test_samples = ["12+7=", "18-6=", "5*4=", "24+15="]
for prompt in test_samples:
    curr = [char2id[c] for c in prompt]
    with torch.no_grad():
        for _ in range(6):
            t_in = torch.tensor([curr], dtype=torch.long, device=DEVICE)
            embeds = uce_seq.perceptor.token_embed(t_in)
            h = uce_seq.token_to_dmodel(embeds)
            h_out, _ = uce_seq.world_model.deliberation_loop(h, max_steps=2)
            logits = uce_seq.symbolic_head(h_out)[:, -1, :]
            next_t = int(torch.argmax(logits, dim=-1).item())
            curr.append(next_t)
            if id2char[next_t] == "\n":
                break
    gen_text = "".join([id2char[i] for i in curr]).strip()
    print(f"  Prompt: {prompt:<8s} -> Generated: '{gen_text}'")

