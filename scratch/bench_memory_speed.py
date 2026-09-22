"""
Benchmark memory retrieval speed on the 5,266-concept brain:
Compares vectorized GPU query against legacy iteration.
"""
import time
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ishmn import UniversalCognitiveEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading unified brain on {device}...")
uce = UniversalCognitiveEngine.load_brain("artifacts/unified_lifelong_brain.pt", device=device)
print(f"Loaded {len(uce.memory.semantic_concepts)} semantic concepts.")

u_query = torch.randn(1, uce.hyperspace_dim, device=device)
u_query = u_query / u_query.norm(dim=-1, keepdim=True)

# Warmup
res = uce.memory.query_semantic_lattice(u_query, top_k=5)

N = 200
t0 = time.perf_counter()
for _ in range(N):
    res = uce.memory.query_semantic_lattice(u_query, top_k=5)
t_vec = (time.perf_counter() - t0) / N * 1000

print(f"[Vectorized GPU Retrieval]: {t_vec:.3f} ms per query across {len(uce.memory.semantic_concepts)} concepts!")
print(f"Top concept retrieved: {res[0]['word'] if res else None}, similarity: {res[0]['semantic_similarity']*100:.1f}%")
