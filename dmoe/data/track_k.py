"""
Track K & T1 Benchmark Task Generator.
Header-Free, Context-Requiring Rule Learning and Belief Revision.

Task Specifications from §6 and §14:
- Alphabet: ASCII 32..47 (16 distinct symbols).
- No header tags, no prefix bytes, no slot-0 giveaways.
- T1: y_t = pi_k(x_t, x_{t-1}) (Domain-specific bigram rules requiring contextual sequence memory).
- K1: New-rule acquisition with n in {1, 3, 10} exposures.
- K2: Structured burst resistance (temporally clustered contradictory burst from single source).
- K3: Genuine belief revision (persistent cross-source counter-evidence).
- K4: Curation under capacity (Zipfian multi-rule frequency distribution).
"""

from typing import Any, Dict, List, Optional, Tuple
import torch

ALPHABET_START = 32
ALPHABET_SIZE = 16
ALPHABET_END = ALPHABET_START + ALPHABET_SIZE  # 48
NOISE_RATE = 0.05


def make_markov_matrix(
    domain_id: int,
    num_states: int = ALPHABET_SIZE,
    rho: float = 0.5,
    seed: int = 0,
) -> torch.Tensor:
    """
    Generate a sparse first-order Markov transition matrix for domain_id.
    rho in [0, 1] controls support overlap between domains:
    - rho = 0.0: Disjoint transition support between domains.
    - rho = 0.5: Partial overlap.
    - rho = 1.0: Full overlap (pure context-dependent rule distinction).
    """
    gen = torch.Generator().manual_seed(int(seed) * 1000 + int(domain_id) * 37 + 7)
    logits = torch.randn(num_states, num_states, generator=gen)

    # Bias transitions based on domain ID to control overlap rho
    if rho < 1.0:
        domain_bias = torch.zeros(num_states, num_states)
        for s in range(num_states):
            preferred = (s + domain_id * 3) % num_states
            domain_bias[s, preferred] += 3.0 * (1.0 - rho)
        logits = logits + domain_bias

    # Keep top-4 sparse transitions per state
    topk_vals, topk_idx = torch.topk(logits, k=4, dim=-1)
    mask = torch.zeros_like(logits).scatter_(-1, topk_idx, 1.0)
    probs = torch.softmax(topk_vals, dim=-1)
    P = torch.zeros_like(logits)
    P.scatter_(-1, topk_idx, probs)
    return P


def make_bigram_rule_table(
    domain_id: int,
    alphabet_size: int = ALPHABET_SIZE,
    seed: int = 0,
) -> torch.Tensor:
    """
    Generate deterministic bigram mapping pi_k(x_t, x_{t-1}) -> y_t in [0..alphabet_size-1].
    Guarantees that y_t cannot be predicted from x_t alone without context x_{t-1}.
    """
    gen = torch.Generator().manual_seed(int(seed) * 2000 + int(domain_id) * 53 + 13)
    table = torch.zeros(alphabet_size, alphabet_size, dtype=torch.long)
    for prev_s in range(alphabet_size):
        # Permutation for current symbol given previous symbol
        perm = torch.randperm(alphabet_size, generator=gen)
        table[prev_s] = perm
    return table


def generate_t1_batch(
    domain_id: int,
    batch_size: int = 16,
    seq_len: int = 16,
    rho: float = 0.5,
    noise_rate: float = NOISE_RATE,
    seed: int = 0,
    rule_table_override: Optional[torch.Tensor] = None,
    domain_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of header-free, context-dependent sequences for domain_id:
    x: [B, seq_len] in [32..47]
    y: [B, seq_len] in [32..47]
    y_t = pi_k(x_t, x_{t-1}) + 32, with 5% label noise.
    """
    gen = torch.Generator().manual_seed(int(seed))
    d_seed = domain_seed if domain_seed is not None else (int(domain_id) * 37 + 7)
    P = make_markov_matrix(domain_id, num_states=ALPHABET_SIZE, rho=rho, seed=d_seed)
    rule_table = (
        rule_table_override
        if rule_table_override is not None
        else make_bigram_rule_table(domain_id, alphabet_size=ALPHABET_SIZE, seed=d_seed)
    )

    x_idx = torch.zeros(batch_size, seq_len, dtype=torch.long)
    # Initial token sampled uniformly from alphabet
    x_idx[:, 0] = torch.randint(0, ALPHABET_SIZE, (batch_size,), generator=gen)

    # Autoregressive Markov transition for inputs
    for t in range(1, seq_len):
        prev = x_idx[:, t - 1]
        step_probs = P[prev]  # [B, ALPHABET_SIZE]
        # Sample next tokens using generator
        sampled = torch.multinomial(step_probs, num_samples=1, generator=gen).squeeze(-1)
        x_idx[:, t] = sampled

    # Compute target y_t = pi_k(x_t, x_{t-1})
    y_idx = torch.zeros(batch_size, seq_len, dtype=torch.long)
    # For t = 0, wrap around to last token or use self
    y_idx[:, 0] = rule_table[x_idx[:, -1], x_idx[:, 0]]
    for t in range(1, seq_len):
        y_idx[:, t] = rule_table[x_idx[:, t - 1], x_idx[:, t]]

    # Apply symmetric label noise
    if noise_rate > 0.0:
        noise_mask = torch.rand((batch_size, seq_len), generator=gen) < noise_rate
        rand_y = torch.randint(0, ALPHABET_SIZE, (batch_size, seq_len), generator=gen)
        y_idx = torch.where(noise_mask, rand_y, y_idx)

    # Shift indices to ASCII alphabet [32..47]
    x_bytes = x_idx + ALPHABET_START
    y_bytes = y_idx + ALPHABET_START
    return x_bytes, y_bytes


def generate_k2_burst_stream(
    domain_id: int,
    total_steps: int = 120,
    burst_start: int = 40,
    burst_length: int = 5,
    batch_size: int = 16,
    seq_len: int = 16,
    seed: int = 0,
    domain_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Task K2 Stream:
    Rule is established during steps 0..burst_start-1.
    Steps burst_start .. burst_start+burst_length-1 receive contradictory labels
    originating from a single unverified source ID ('adversarial_source').
    At burst_start + burst_length, normal distribution resumes.
    """
    stream = []
    d_seed = domain_seed if domain_seed is not None else (int(domain_id) * 37 + 7)
    normal_rule = make_bigram_rule_table(domain_id, seed=d_seed)
    # Contradictory rule (inverted permutation)
    contra_rule = (normal_rule + 7) % ALPHABET_SIZE

    for st in range(total_steps):
        is_burst = burst_start <= st < (burst_start + burst_length)
        source_id = "noisy_burst_source" if is_burst else "ground_truth_stream"
        active_table = contra_rule if is_burst else normal_rule
        x, y = generate_t1_batch(
            domain_id=domain_id,
            batch_size=batch_size,
            seq_len=seq_len,
            seed=seed * 10000 + st,
            rule_table_override=active_table,
            domain_seed=d_seed,
        )
        stream.append({
            "step": st,
            "domain_id": domain_id,
            "x": x,
            "y": y,
            "is_contradiction": is_burst,
            "source_id": source_id,
        })
    return stream


def generate_k3_revision_stream(
    domain_id: int,
    total_steps: int = 120,
    revision_start: int = 50,
    batch_size: int = 16,
    seq_len: int = 16,
    seed: int = 0,
    domain_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Task K3 Stream:
    Rule is established during steps 0..revision_start-1.
    From revision_start onward, the rule permanently changes across multiple diverse source IDs.
    """
    stream = []
    d_seed = domain_seed if domain_seed is not None else (int(domain_id) * 37 + 7)
    old_rule = make_bigram_rule_table(domain_id, seed=d_seed)
    revised_rule = (old_rule * 3 + 5) % ALPHABET_SIZE

    for st in range(total_steps):
        is_revised = st >= revision_start
        source_id = f"verified_source_{st % 4}" if is_revised else "original_source"
        active_table = revised_rule if is_revised else old_rule
        x, y = generate_t1_batch(
            domain_id=domain_id,
            batch_size=batch_size,
            seq_len=seq_len,
            seed=seed * 10000 + st,
            rule_table_override=active_table,
            domain_seed=d_seed,
        )
        stream.append({
            "step": st,
            "domain_id": domain_id,
            "x": x,
            "y": y,
            "is_revised": is_revised,
            "source_id": source_id,
        })
    return stream
