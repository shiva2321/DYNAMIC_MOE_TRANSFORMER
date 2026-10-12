import sys
sys.path.insert(0, ".")
import math
import numpy as np
import torch
import torch.nn.functional as F

from growmoe.config import PhyloCfg
from growmoe.pools.phylo import PhyloMoE
from growmoe.experts import MLPExpert
from growmoe.stack import RepExpert, RepLayer, StackMoE

def _units(n: int, d: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    return torch.from_numpy(x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).float()

print("Testing StackMoE learning loop mechanics...")
