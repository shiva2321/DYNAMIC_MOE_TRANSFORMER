"""
Arithmetic Dataset Engine & Spherical Feature Generator.

Generates thousands of arithmetic operations across diverse operators:
- Addition: a + b
- Subtraction: a - b
- Multiplication: a * b
- Modulo: a % 5

Splits generated:
1. Train Split: 4,000 problems with operands a, b in [1, 50]
2. In-Distribution (ID) Holdout: 500 unseen problems with a, b in [1, 50]
3. Out-Of-Distribution (OOD) Extrapolation: 500 unseen problems with a, b in [51, 100]
4. Unseen Composite Operations: 500 chained two-step operations (e.g. a * b + c, a + b - c)
"""

import random
import numpy as np
import torch
from geometry import project_to_sphere

OPERATOR_MAP = {
    "+": 0,
    "-": 1,
    "*": 2,
    "%": 3,
    "composite": 4
}

NORMALIZATION_SCALE = 50.0

def build_dataset_vectorized(
    samples: list,
    d_model: int = 32,
    hyperspace_dim: int = 8,
    device: torch.device = None
) -> dict:
    """
    Vectorized tensor construction for arithmetic problems in infinite spherical hyperspace.
    Constructs all input embeddings, S^{D-1} unit vectors, and radial anchors in parallel.
    """
    if device is None:
        device = torch.device("cpu")

    N = len(samples)
    a_vals = torch.tensor([s[0] for s in samples], dtype=torch.float32, device=device)
    op_ids = torch.tensor([OPERATOR_MAP[s[1]] for s in samples], dtype=torch.long, device=device)
    b_vals = torch.tensor([s[2] for s in samples], dtype=torch.float32, device=device)
    res_vals = torch.tensor([float(s[3]) for s in samples], dtype=torch.float32, device=device)

    # 1. Continuous input feature tensor [N, d_model]
    X = torch.zeros(N, d_model, device=device)
    X[:, 0] = a_vals / NORMALIZATION_SCALE
    X[:, 1] = b_vals / NORMALIZATION_SCALE

    # One-hot operator encoding (indices 2..6)
    X.scatter_(1, (2 + op_ids).unsqueeze(1), 1.0)

    # Sinusoidal positional harmonics for numbers
    freqs = [1.0, 2.0, 4.0, 8.0, 16.0]
    for idx, f in enumerate(freqs):
        X[:, 7 + idx * 2] = torch.sin(f * a_vals / NORMALIZATION_SCALE)
        X[:, 7 + idx * 2 + 1] = torch.cos(f * a_vals / NORMALIZATION_SCALE)
        X[:, 17 + idx * 2] = torch.sin(f * b_vals / NORMALIZATION_SCALE)
        X[:, 17 + idx * 2 + 1] = torch.cos(f * b_vals / NORMALIZATION_SCALE)

    # 2. Canonical Hyperspace Coordinates (u in S^{D-1}, r in R+)
    raw_u = torch.zeros(N, hyperspace_dim, device=device)
    raw_u.scatter_(1, (op_ids % hyperspace_dim).unsqueeze(1), 2.0)
    raw_u.scatter_(1, ((op_ids + 2) % hyperspace_dim).unsqueeze(1), 1.0)
    raw_u[:, 4] = torch.sin(a_vals / 10.0)
    raw_u[:, 5] = torch.cos(b_vals / 10.0)

    U = project_to_sphere(raw_u)
    R = (1.0 + torch.log1p(torch.abs(a_vals) + torch.abs(b_vals))).unsqueeze(1)
    Y = (res_vals / NORMALIZATION_SCALE).unsqueeze(1)
    ops = [s[1] for s in samples]

    return {"X": X, "U": U, "R": R, "Y": Y, "ops": ops, "samples": samples}

def generate_arithmetic_splits(
    num_train: int = 4000,
    num_id_val: int = 500,
    num_ood: int = 500,
    num_composite: int = 500,
    d_model: int = 32,
    hyperspace_dim: int = 8,
    device: torch.device = None
):
    if device is None:
        device = torch.device("cpu")

    # 1. Enumerate complete candidate universe for a, b in [1, 50] (10,000 unique problems)
    all_pairs = []
    for op in ["+", "-", "*", "%"]:
        for a in range(1, 51):
            for b in range(1, 51):
                if op == "+":
                    res = a + b
                elif op == "-":
                    res = a - b
                elif op == "*":
                    res = a * b
                elif op == "%":
                    res = a % 5
                all_pairs.append((a, op, b, res))

    random.shuffle(all_pairs)

    # Completely disjoint train and unseen in-distribution splits
    train_samples = all_pairs[:num_train]
    id_samples = all_pairs[num_train:num_train + num_id_val]

    # 2. Out-of-Distribution (OOD) Extrapolation: operands in [51, 100]
    ood_samples = []
    for _ in range(num_ood):
        op = random.choice(["+", "-", "*", "%"])
        a = random.randint(51, 100)
        b = random.randint(51, 100)
        if op == "+":
            res = a + b
        elif op == "-":
            res = a - b
        elif op == "*":
            res = a * b
        elif op == "%":
            res = a % 5
        ood_samples.append((a, op, b, res))

    # 3. Unseen Composite Operations: chained two-step reasoning (a * b + c) or (a + b - c)
    composite_samples = []
    for _ in range(num_composite):
        subtype = random.choice(["mult_add", "add_sub"])
        if subtype == "mult_add":
            a = random.randint(1, 15)
            b = random.randint(1, 10)
            c = random.randint(1, 20)
            res = (a * b) + c
            composite_samples.append((a, "composite", b + c * 0.1, res))
        else:
            a = random.randint(10, 40)
            b = random.randint(5, 25)
            c = random.randint(1, 15)
            res = (a + b) - c
            composite_samples.append((a, "composite", b - c * 0.1, res))

    return {
        "train": build_dataset_vectorized(train_samples, d_model=d_model, hyperspace_dim=hyperspace_dim, device=device),
        "val_id": build_dataset_vectorized(id_samples, d_model=d_model, hyperspace_dim=hyperspace_dim, device=device),
        "val_ood": build_dataset_vectorized(ood_samples, d_model=d_model, hyperspace_dim=hyperspace_dim, device=device),
        "val_composite": build_dataset_vectorized(composite_samples, d_model=d_model, hyperspace_dim=hyperspace_dim, device=device)
    }
