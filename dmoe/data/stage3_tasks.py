"""
Header-free, shared-vocabulary Stage 3 continual learning, belief revision,
and compositional generalization task generators.

Key anti-cheat invariants enforced across all tasks:
1. All domains share the exact ASCII token alphabet [32..47] (vocab size 16).
2. No domain header tokens, no slot-0 domain indicators, no disjoint token ranges.
3. Identical near-uniform unigram token marginals across domains (1/16 per token);
   domains differ in directed transition structure (controlled by overlap `rho`)
   and causal relational target mapping y_t = pi_k(x_{t-1}, x_t).
4. Held-out compositional pairs share underlying isomorphic structure with seen
   pairs so neural generalization above 1/16 chance is mathematically learnable
   while pure key-value dictionaries drop to chance (~1/16) on unseen pairs.
"""

from typing import Dict, List, Optional, Tuple
import torch

ALPHABET_START = 32
ALPHABET_SIZE = 16
SEP_TOKEN = 48
UNK_BASE = 49


def make_shared_vocab_markov(
    domain_id: int,
    rho: float = 0.0,
    num_states: int = ALPHABET_SIZE,
    seed: int = 1000,
) -> torch.Tensor:
    """
    Construct a doubly-balanced transition matrix P[s, s'] over [0..num_states-1]
    whose stationary unigram distribution is exact uniform 1/num_states across ALL
    domains, while directed transition edges depend on `domain_id` and overlap `rho`.
    """
    gen = torch.Generator().manual_seed(int(seed) * 101 + int(domain_id) * 37 + 19)
    # Disjoint offset groups (excluding 0 so Markov chains mix rapidly; gcd / coprime spread)
    domain_offsets = [
        (1, 2, 5, 9),
        (3, 4, 8, 12),
        (6, 7, 11, 14),
        (10, 13, 15, 2),
    ]
    shared_offsets = (1, 3, 6, 10)
    dom_offs = domain_offsets[int(domain_id) % len(domain_offsets)]

    P = torch.zeros(num_states, num_states, dtype=torch.float32)
    # Build cyclic transition kernel biased toward domain offsets vs shared pool
    w_dom = float(max(0.0, 1.0 - float(rho)))
    w_shared = float(max(0.0, min(1.0, float(rho))))
    # Deterministic minor weights per offset slot preserve translation uniformity
    slot_w = torch.tensor([0.34, 0.26, 0.22, 0.18], dtype=torch.float32)
    perm_slots = torch.randperm(4, generator=gen)
    slot_w = slot_w[perm_slots]

    for s in range(num_states):
        for j, off in enumerate(dom_offs):
            P[s, (s + off) % num_states] += w_dom * float(slot_w[j].item())
        for j, off in enumerate(shared_offsets):
            P[s, (s + off) % num_states] += w_shared * float(slot_w[j].item())

    row_sums = P.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return P / row_sums


def make_relational_rule_table(
    domain_id: int,
    alphabet_size: int = ALPHABET_SIZE,
    seed: int = 1000,
) -> torch.Tensor:
    """
    Construct a non-linear context-requiring rule table pi_k(a, b) -> y in [0..15]
    where every marginal H(Y|a) = H(Y|b) = log(16), making unigram guessing chance 1/16.
    """
    gen = torch.Generator().manual_seed(int(seed) * 313 + int(domain_id) * 79 + 11)
    perm_a = torch.randperm(alphabet_size, generator=gen)
    perm_b = torch.randperm(alphabet_size, generator=gen)
    perm_out = torch.randperm(alphabet_size, generator=gen)
    muls = [1, 3, 5, 7]
    mul = muls[int(domain_id) % len(muls)]

    grid_a = perm_a.unsqueeze(1).expand(alphabet_size, alphabet_size)
    grid_b = perm_b.unsqueeze(0).expand(alphabet_size, alphabet_size)
    idx = (grid_a + mul * grid_b) % alphabet_size
    return perm_out[idx]


def generate_stage3a_batch(
    domain_id: int,
    batch_size: int = 32,
    seq_len: int = 16,
    rho: float = 0.0,
    seed: int = 0,
    domain_seed: int = 1000,
    noise_rate: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate header-free, shared-vocab [32..47] causal sequence task batch for Stage 3A.
    x: [B, L] in [32..47], y[:, t] = pi_k(x[:, t-1], x[:, t]) + 32 for t >= 1.
    """
    gen = torch.Generator().manual_seed(int(seed))
    P = make_shared_vocab_markov(domain_id, rho=rho, seed=domain_seed)
    rule = make_relational_rule_table(domain_id, seed=domain_seed)

    x_idx = torch.zeros(batch_size, seq_len, dtype=torch.long)
    x_idx[:, 0] = torch.randint(0, ALPHABET_SIZE, (batch_size,), generator=gen)
    for t in range(1, seq_len):
        probs = P[x_idx[:, t - 1]]
        x_idx[:, t] = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)

    y_idx = torch.zeros_like(x_idx)
    y_idx[:, 0] = x_idx[:, 0]
    y_idx[:, 1:] = rule[x_idx[:, :-1], x_idx[:, 1:]]

    if noise_rate > 0.0:
        n_mask = torch.rand((batch_size, seq_len), generator=gen) < noise_rate
        n_mask[:, 0] = False
        rand_y = torch.randint(0, ALPHABET_SIZE, (batch_size, seq_len), generator=gen)
        y_idx = torch.where(n_mask, rand_y, y_idx)

    return x_idx + ALPHABET_START, y_idx + ALPHABET_START


def build_compositional_belief_tables(
    seed: int = 0,
    num_exceptions: int = 12,
    split_ratio: float = 0.75,
) -> Dict[str, object]:
    """
    Build structured compositional law + curated idiosyncratic exception pairs
    for Stage 3B, plus a stratified 75/25 (seen / held-out) context split where
    every individual symbol appears frequently in `seen_pairs`.
    """
    gen = torch.Generator().manual_seed(int(seed) * 911 + 42)
    phi = torch.randperm(ALPHABET_SIZE, generator=gen)
    inv_phi = torch.argsort(phi)

    grid_a = phi.unsqueeze(1).expand(ALPHABET_SIZE, ALPHABET_SIZE)
    grid_b = phi.unsqueeze(0).expand(ALPHABET_SIZE, ALPHABET_SIZE)

    base_law = inv_phi[(grid_a + 3 * grid_b) % ALPHABET_SIZE]
    rev_law = inv_phi[(grid_a + 3 * grid_b + 5) % ALPHABET_SIZE]
    burst_law = inv_phi[(grid_a + 5 * grid_b + 9) % ALPHABET_SIZE]

    # Balanced stratified split: hold out exact `4` targets per `prev_s` row -> 192 seen / 64 held-out
    seen_mask = torch.ones(ALPHABET_SIZE, ALPHABET_SIZE, dtype=torch.bool)
    seen_pairs: List[Tuple[int, int]] = []
    heldout_pairs: List[Tuple[int, int]] = []
    per_row_hold = max(1, int(round(ALPHABET_SIZE * (1.0 - split_ratio))))
    for a in range(ALPHABET_SIZE):
        row_perm = torch.randperm(ALPHABET_SIZE, generator=gen)
        hold_cols = row_perm[:per_row_hold].tolist()
        for b in hold_cols:
            seen_mask[a, b] = False
        for b in range(ALPHABET_SIZE):
            if seen_mask[a, b]:
                seen_pairs.append((a, b))
            else:
                heldout_pairs.append((a, b))

    # Inject fixed idiosyncratic exceptions inside seen_pairs (testing episodic Lane E exact memory!)
    exc_perm = torch.randperm(len(seen_pairs), generator=gen)[:num_exceptions].tolist()
    exc_pairs = [seen_pairs[i] for i in exc_perm]
    init_table = base_law.clone()
    rev_table = rev_law.clone()
    burst_table = burst_law.clone()
    for a, b in exc_pairs:
        init_table[a, b] = (base_law[a, b] + 7) % ALPHABET_SIZE
        rev_table[a, b] = (rev_law[a, b] + 11) % ALPHABET_SIZE
        burst_table[a, b] = (burst_law[a, b] + 3) % ALPHABET_SIZE

    return {
        "init_table": init_table,
        "burst_table": burst_table,
        "rev_table": rev_table,
        "seen_mask": seen_mask,
        "seen_pairs": seen_pairs,
        "heldout_pairs": heldout_pairs,
        "exc_pairs": exc_pairs,
    }


def generate_3b_seen_batch(
    rule_table: torch.Tensor,
    seen_mask: torch.Tensor,
    batch_size: int = 32,
    seq_len: int = 16,
    seed: int = 0,
    focus_pairs: Optional[List[Tuple[int, int]]] = None,
    focus_prob: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample continuous autoregressive sequences whose transitions (x_{t-1}, x_t)
    lie strictly inside `seen_mask` (zero contamination of held-out contexts).
    """
    gen = torch.Generator().manual_seed(int(seed))
    trans_probs = seen_mask.float()
    row_sums = trans_probs.sum(dim=-1, keepdim=True).clamp_min(1.0)
    P = trans_probs / row_sums

    x_idx = torch.zeros(batch_size, seq_len, dtype=torch.long)
    x_idx[:, 0] = torch.randint(0, ALPHABET_SIZE, (batch_size,), generator=gen)
    for t in range(1, seq_len):
        prev = x_idx[:, t - 1]
        x_idx[:, t] = torch.multinomial(P[prev], num_samples=1, generator=gen).squeeze(-1)

    if focus_pairs and focus_prob > 0.0:
        for b in range(batch_size):
            if float(torch.rand(1, generator=gen).item()) < focus_prob and seq_len >= 4:
                pa, pb = focus_pairs[int(torch.randint(0, len(focus_pairs), (1,), generator=gen).item())]
                pos = int(torch.randint(0, seq_len - 2, (1,), generator=gen).item())
                x_idx[b, pos] = pa
                x_idx[b, pos + 1] = pb
                # Re-anchor next token strictly inside seen_mask[pb]
                x_idx[b, pos + 2] = int(torch.multinomial(P[pb], num_samples=1, generator=gen).item())

    y_idx = torch.zeros_like(x_idx)
    y_idx[:, 0] = x_idx[:, 0]
    y_idx[:, 1:] = rule_table[x_idx[:, :-1], x_idx[:, 1:]]
    return x_idx + ALPHABET_START, y_idx + ALPHABET_START


def generate_3b_pair_eval_batch(
    rule_table: torch.Tensor,
    pairs: List[Tuple[int, int]],
    batch_size: int = 64,
    seq_len: int = 16,
    seed: int = 999,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate exact pair subsets (e.g. `heldout_pairs` or `exc_pairs`) at odd steps t=1,3,...,L-1.
    Returns (x [B, L], y [B, L], eval_mask [B, L] bool).
    """
    gen = torch.Generator().manual_seed(int(seed))
    x_idx = torch.zeros(batch_size, seq_len, dtype=torch.long)
    y_idx = torch.zeros(batch_size, seq_len, dtype=torch.long)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    n_pairs = len(pairs)
    idx_draw = torch.randint(0, n_pairs, (batch_size, seq_len // 2), generator=gen)
    for b in range(batch_size):
        for slot in range(seq_len // 2):
            t = slot * 2
            a, b_tok = pairs[int(idx_draw[b, slot].item())]
            x_idx[b, t] = a
            x_idx[b, t + 1] = b_tok
            y_idx[b, t + 1] = rule_table[a, b_tok]
            mask[b, t + 1] = True
    return x_idx + ALPHABET_START, y_idx + ALPHABET_START, mask


def generate_3b_ic_episodes(
    batch_size: int,
    num_facts: int = 6,
    seed: int = 0,
    include_contradiction: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    In-context causal fact binding + contradicting observation benchmark (Stage 3B-IC):
    Prompt: [e_0, v_0, ..., e_{F-1}, v_{F-1}, obs_e, obs_v, e_q, SEP] -> answer = v_q.
    When `include_contradiction=True`, `obs_e = e_q` and `obs_v != v_q` (corrupted noise
    observation after clean establishment); when False, `obs_e, obs_v` is an extra consistent fact.
    Returns (x [B, L], y_target [B], corr_mask [B, L] bool marking the corrupted obs value step).
    """
    gen = torch.Generator().manual_seed(int(seed))
    total_facts = num_facts + 1  # +1 trailing obs pair before query
    L = total_facts * 2 + 2
    x = torch.zeros(batch_size, L, dtype=torch.long)
    y = torch.zeros(batch_size, dtype=torch.long)
    corr_mask = torch.zeros(batch_size, L, dtype=torch.bool)

    for b in range(batch_size):
        ents = torch.randperm(ALPHABET_SIZE, generator=gen)[: total_facts] + ALPHABET_START
        vals = torch.randperm(ALPHABET_SIZE, generator=gen)[: total_facts] + ALPHABET_START
        q_idx = int(torch.randint(0, num_facts, (1,), generator=gen).item())
        for i in range(num_facts):
            x[b, 2 * i] = ents[i]
            x[b, 2 * i + 1] = vals[i]

        obs_pos = 2 * num_facts
        if include_contradiction:
            wrong_v = ((vals[q_idx] - ALPHABET_START + 1 + int(torch.randint(0, ALPHABET_SIZE - 1, (1,), generator=gen).item())) % ALPHABET_SIZE) + ALPHABET_START
            x[b, obs_pos] = ents[q_idx]
            x[b, obs_pos + 1] = wrong_v
            corr_mask[b, obs_pos + 1] = True
        else:
            x[b, obs_pos] = ents[num_facts]
            x[b, obs_pos + 1] = vals[num_facts]

        x[b, 2 * total_facts] = ents[q_idx]
        x[b, 2 * total_facts + 1] = SEP_TOKEN
        y[b] = vals[q_idx]

    return x, y, corr_mask
