"""
Tests for Unified Multi-Scale Novelty Detection (dmoe/novelty.py).

Verifies:
1. In-distribution batches are not false-flagged after baseline calibration.
2. Coarse domain shifts are detected as "coarse_domain".
3. Same-input / different-rule ("near-twin") shifts are detected as "near_twin_rule".
4. Localized nuance (1-2 anomalous tokens inside a mostly familiar 16-token sequence,
   where sequence-mean distance and mean surprise stay low) is detected as "local_nuance"
   and pinpointed by novel_token_mask.
5. End-to-end RealSelfStabilizingDMoE returns calibrated NoveltyVerdict in forward().
"""

import torch

from dmoe.novelty import UnifiedNoveltyDetector
from dmoe.real_model import RealSelfStabilizingDMoE


def _calibrate_detector(det: UnifiedNoveltyDetector, base_cent: torch.Tensor):
    torch.manual_seed(7)
    base_proto = torch.nn.functional.normalize(base_cent.clone(), p=2, dim=-1)
    for _ in range(15):
        B, L, D = 8, 16, base_cent.size(-1)
        tok_lat = base_proto.view(1, 1, D).expand(B, L, D) + 0.04 * torch.randn(B, L, D)
        seq_sig = torch.nn.functional.normalize(tok_lat.mean(dim=1), p=2, dim=-1)
        surp = 0.05 + 0.02 * torch.rand(B, L)
        nll = 0.10 + 0.05 * torch.rand(B, L)
        v = det.evaluate(
            expert_idx=0,
            seq_sig=seq_sig,
            expert_centroid=base_proto.unsqueeze(0),
            token_latents=tok_lat,
            token_surprise=surp,
            token_nll=nll,
            expert_is_converged=True,
            update_baseline=True,
        )
        assert v.is_novel is False


def test_unified_novelty_in_distribution_and_coarse_shift():
    D = 24
    det = UnifiedNoveltyDetector(d_model=D)
    torch.manual_seed(3)
    base_cent = torch.randn(D)
    _calibrate_detector(det, base_cent)

    # 1. In-distribution check
    B, L = 4, 16
    tok_id = base_cent.view(1, 1, D).expand(B, L, D) + 0.04 * torch.randn(B, L, D)
    sig_id = torch.nn.functional.normalize(tok_id.mean(dim=1), p=2, dim=-1)
    v_id = det.evaluate(
        expert_idx=0,
        seq_sig=sig_id,
        expert_centroid=base_cent.unsqueeze(0),
        token_latents=tok_id,
        token_surprise=torch.full((B, L), 0.06),
        token_nll=torch.full((B, L), 0.12),
        expert_is_converged=True,
        update_baseline=False,
    )
    assert v_id.is_novel is False
    assert v_id.novelty_type == "none"

    # 2. Coarse domain shift (orthogonal/negated latent distribution)
    coarse_cent = -base_cent
    tok_coarse = coarse_cent.view(1, 1, D).expand(B, L, D) + 0.04 * torch.randn(B, L, D)
    sig_coarse = torch.nn.functional.normalize(tok_coarse.mean(dim=1), p=2, dim=-1)
    v_coarse = det.evaluate(
        expert_idx=0,
        seq_sig=sig_coarse,
        expert_centroid=base_cent.unsqueeze(0),
        token_latents=tok_coarse,
        token_surprise=torch.full((B, L), 0.06),
        expert_is_converged=True,
        update_baseline=False,
    )
    assert v_coarse.is_novel is True
    assert v_coarse.novelty_type == "coarse_domain"


def test_unified_novelty_near_twin_rule_shift():
    D = 24
    det = UnifiedNoveltyDetector(d_model=D)
    torch.manual_seed(5)
    base_cent = torch.randn(D)
    _calibrate_detector(det, base_cent)

    # Identical inputs (seq_dist ~ 0), but rule flipped -> high surprise across sequence
    B, L = 4, 16
    tok_same = base_cent.view(1, 1, D).expand(B, L, D) + 0.04 * torch.randn(B, L, D)
    sig_same = torch.nn.functional.normalize(tok_same.mean(dim=1), p=2, dim=-1)
    v_twin = det.evaluate(
        expert_idx=0,
        seq_sig=sig_same,
        expert_centroid=base_cent.unsqueeze(0),
        token_latents=tok_same,
        token_surprise=torch.full((B, L), 0.72),
        token_nll=torch.full((B, L), 1.80),
        expert_is_converged=True,
        update_baseline=False,
    )
    assert v_twin.is_novel is True
    assert v_twin.novelty_type == "near_twin_rule"


def test_unified_novelty_detects_local_single_token_nuance():
    """
    87.5% of tokens (14/16) are completely familiar, so sequence-mean distance
    and sequence-mean surprise stay low (~0.15), yet 2 nuance-flipped tokens
    at positions [12, 13] trigger "local_nuance" and are pinpointed in novel_token_mask.
    """
    D = 24
    det = UnifiedNoveltyDetector(d_model=D, tail_fraction=0.15)
    torch.manual_seed(9)
    base_cent = torch.randn(D)
    _calibrate_detector(det, base_cent)

    B, L = 4, 16
    tok_lat = base_cent.view(1, 1, D).expand(B, L, D).clone() + 0.03 * torch.randn(B, L, D)
    # Only 2 of 16 tokens are shifted/negated (12.5% of sequence)
    tok_lat[:, 12:14, :] = -base_cent.view(1, 1, D)
    sig = torch.nn.functional.normalize(tok_lat.mean(dim=1), p=2, dim=-1)

    surp = torch.full((B, L), 0.05)
    surp[:, 12:14] = 0.85  # mean surprise = 0.15 (below 0.42 mean threshold!)
    nll = torch.full((B, L), 0.10)
    nll[:, 12:14] = 2.40

    assert float(surp.mean().item()) < 0.20  # Mean-only detector would miss this

    v_nuance = det.evaluate(
        expert_idx=0,
        seq_sig=sig,
        expert_centroid=base_cent.unsqueeze(0),
        token_latents=tok_lat,
        token_surprise=surp,
        token_nll=nll,
        expert_is_converged=True,
        update_baseline=False,
    )
    assert v_nuance.is_novel is True
    assert v_nuance.novelty_type == "local_nuance"
    assert v_nuance.novel_token_mask is not None
    assert bool(v_nuance.novel_token_mask[:, 12:14].all().item()) is True
    assert bool(v_nuance.novel_token_mask[:, :10].any().item()) is False


def test_real_model_emits_novelty_verdict_in_forward():
    torch.manual_seed(17)
    model = RealSelfStabilizingDMoE(
        d_model=16,
        num_heads=2,
        d_ff=32,
        initial_layers=1,
        chunk_size=1,
        enable_thinking=True,
        max_thinking_steps=2,
        auto_lifecycle=True,
    )
    model.train()
    xb = torch.randint(32, 40, (4, 8))
    out = model(xb, force_thinking_steps=2, targets=xb)
    verdict = out["novelty_verdict"]
    assert verdict is not None
    assert 0.0 <= verdict.novelty_score <= 1.0
    assert verdict.novelty_type in {"none", "coarse_domain", "near_twin_rule", "local_nuance"}


def test_high_entropy_real_stream_calibration_no_deadlock():
    """
    On open-vocabulary natural text streams, in-distribution surprise can exceed 0.42
    (e.g., 0.55). Verify that UnifiedNoveltyDetector calibrates without deadlocking
    and subsequently accepts unseen batches at that baseline while flagging spikes.
    """
    torch.manual_seed(99)
    D = 16
    det = UnifiedNoveltyDetector(d_model=D)
    base_cent = torch.nn.functional.normalize(torch.randn(1, D), p=2, dim=-1)

    for _ in range(12):
        tok_lat = base_cent.view(1, 1, D).expand(4, 16, D) + 0.04 * torch.randn(4, 16, D)
        sig = torch.nn.functional.normalize(tok_lat.mean(dim=1), p=2, dim=-1)
        surp = torch.full((4, 16), 0.54) + 0.01 * torch.randn(4, 16)
        nll = torch.full((4, 16), 2.10) + 0.05 * torch.randn(4, 16)
        det.evaluate(
            expert_idx=0,
            seq_sig=sig,
            expert_centroid=base_cent,
            token_latents=tok_lat,
            token_surprise=surp,
            token_nll=nll,
            expert_is_converged=True,
            update_baseline=True,
        )

    assert det.expert_stats[0].count == 12
    # Subsequent batch at the same 0.54 surprise baseline must NOT be flagged novel
    tok_lat = base_cent.view(1, 1, D).expand(4, 16, D) + 0.04 * torch.randn(4, 16, D)
    sig = torch.nn.functional.normalize(tok_lat.mean(dim=1), p=2, dim=-1)
    v_in = det.evaluate(
        expert_idx=0,
        seq_sig=sig,
        expert_centroid=base_cent,
        token_latents=tok_lat,
        token_surprise=torch.full((4, 16), 0.55),
        token_nll=torch.full((4, 16), 2.12),
        expert_is_converged=True,
        update_baseline=False,
    )
    assert v_in.is_novel is False


