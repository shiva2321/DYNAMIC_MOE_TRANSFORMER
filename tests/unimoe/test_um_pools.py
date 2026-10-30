"""The four improvements as a pool subclass (unimoe/heads.py, unimoe/pools.py): a running LDA head, fusion of the answer with a domain head and with the per-expert LDA head, a growth statistic
that does not depend on the training recipe, and honest memory accounting. growmoe/ is not edited."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from growmoe.config import PhyloCfg
from growmoe.experts import MLPExpert
from growmoe.pools.phylo import PhyloMoE
from unimoe import density as DN
from unimoe import heads as HD
from unimoe import pools as PL

D = 128


def blobs(n_per=40, classes=(0, 1, 2), seed=0, noise=0.25, n_classes=12):
    centers = F.normalize(torch.randn(n_classes, D, generator=torch.Generator().manual_seed(1234)), dim=1)       # the classes are fixed; only the noise and the order depend on the seed
    g = torch.Generator().manual_seed(seed)
    ys = torch.tensor([c for c in classes for _ in range(n_per)])
    x = F.normalize(centers[ys] + noise * torch.randn(len(ys), D, generator=g), dim=1)
    perm = torch.randperm(len(ys), generator=g)
    return x[perm], ys[perm]


# ---------------------------------------------------------------------------------------------------------------------------------- the running LDA head
def test_the_lda_head_separates_classes_and_gives_no_mass_to_classes_it_has_not_seen():
    x, y = blobs()
    h = HD.LdaHead(D, 12)
    h.update(x, y)
    xt, yt = blobs(seed=1)
    p = h.proba(xt)
    assert float((p.argmax(1) == yt).float().mean()) > 0.95
    assert float(p[:, 3:].sum()) < 1e-6 and torch.allclose(p.sum(1), torch.ones(len(p)), atol=1e-5)
    assert h.seen().tolist() == [True, True, True] + [False] * 9


def test_the_lda_head_is_one_pass_and_incremental_the_same_statistics_whatever_the_chunking():
    x, y = blobs(seed=2)
    one, chunked = HD.LdaHead(D, 12), HD.LdaHead(D, 12)
    one.update(x, y)
    for lo in range(0, len(y), 17):
        chunked.update(x[lo:lo + 17], y[lo:lo + 17])
    xt, _ = blobs(seed=3)
    assert torch.allclose(one.logits(xt)[:, :3], chunked.logits(xt)[:, :3], atol=1e-3)


def test_the_lda_head_stays_finite_with_very_few_samples_and_counts_what_it_stores():
    h = HD.LdaHead(D, 12)
    x, y = blobs(n_per=1)
    h.update(x, y)
    assert torch.isfinite(h.logits(x)[:, :3]).all()
    assert h.n_scalars() == 12 * D + D * D + 12 + 1                                              # class sums, the pooled scatter, class counts, the total


# ---------------------------------------------------------------------------------------------------------------------------------- the fusion functions
def test_fusing_with_a_domain_head_restricts_or_reweights_the_answer_to_the_predicted_domain():
    blocks = ((0, 3), (3, 3))
    probs = torch.tensor([[0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.0, 0.0]])                           # class 3 (domain 1) is the pool's favourite
    dom = torch.tensor([[0.9, 0.1]])                                                              # the domain head says domain 0
    hard = PL.fuse_domain(probs, dom, blocks, "hard", 1.0)
    assert float(hard[0, 3:].sum()) == 0.0 and float(hard.sum()) == pytest.approx(1.0) and int(hard.argmax()) in (0, 1, 2)
    soft = PL.fuse_domain(probs, dom, blocks, "soft", 1.0)
    assert float(soft.sum()) == pytest.approx(1.0) and int(soft.argmax()) in (0, 1, 2) and float(soft[0, 3]) > 0.0
    assert torch.equal(PL.fuse_domain(probs, dom, blocks, "off", 1.0), probs)
    assert float(PL.fuse_domain(probs, torch.tensor([[0.0, 0.0]]), blocks, "hard", 1.0).sum()) == pytest.approx(1.0)      # no mass anywhere: the answer is kept, not zeroed
    padded = PL.fuse_domain(torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05, 0.0]]), dom, blocks, "soft", 1.0)        # labels outside every block keep a tiny factor
    assert torch.isfinite(padded).all()


def test_blending_the_expert_with_its_lda_head_interpolates_in_log_probability():
    p_mlp = torch.tensor([[0.7, 0.2, 0.1]])
    p_lda = torch.tensor([[0.1, 0.8, 0.1]])
    assert torch.allclose(PL.blend(p_mlp, p_lda, 0.0), p_mlp, atol=1e-6)
    assert int(PL.blend(p_mlp, p_lda, 1.0).argmax()) == 1 and int(PL.blend(p_mlp, p_lda, 0.5).argmax()) in (0, 1)
    assert float(PL.blend(p_mlp, p_lda, 0.5).sum()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------------------------------------------- the pool
def cfg(**kw):
    base = dict(d=D, n_classes=12, route_rule="accept_then_logf", seed=3, inner_steps=3)
    base.update(kw)
    return PL.UniCfg(**base)


def stream(seed=0, n_batches=12, noise=0.4):
    x, y = blobs(n_per=64, classes=(0, 1, 2, 3, 4, 5), seed=seed, noise=noise)
    return [(x[i * 64:(i + 1) * 64], y[i * 64:(i + 1) * 64]) for i in range(min(n_batches, len(y) // 64))]


def test_with_every_improvement_off_the_subclass_is_the_shipped_pool():
    a = PL.UniPool(cfg())
    b = PhyloMoE(PhyloCfg(d=D, n_classes=12, route_rule="accept_then_logf", seed=3, inner_steps=3))
    for x, y in stream():
        a.observe(x, y)
        b.observe(x, y)
    xt, _ = blobs(seed=9, classes=(0, 1, 2, 3, 4, 5))
    assert torch.allclose(a.predict(xt).probs, b.predict(xt).probs, atol=1e-6)
    assert a.summary()["n_experts"] == b.summary()["n_experts"]


def test_the_one_step_fit_gap_does_not_depend_on_the_training_recipe_and_grows_with_width():
    x, y = blobs(n_per=64, classes=tuple(range(8)), noise=0.5)
    gaps = {}
    for steps in (1, 20):
        pool = PL.UniPool(cfg(inner_steps=steps))
        for width in (8, 256):
            mod = MLPExpert(D, width, 1, 12, seed=1)
            rec = type("R", (), {"module": mod})()
            pre, post = pool._one_step_probe(rec, x[:64], y[:64])
            gaps[(steps, width)] = 1.0 - float(post.sum() / pre.sum())
    assert gaps[(1, 8)] == pytest.approx(gaps[(20, 8)]) and gaps[(1, 256)] == pytest.approx(gaps[(20, 256)])       # the recipe does not enter
    assert gaps[(1, 256)] > gaps[(1, 8)] > 0.0                                                                    # a wider expert removes more of a fresh batch in one step


def test_growth_by_the_one_step_gap_fires_for_a_starved_expert_when_allowed_and_not_when_the_threshold_forbids_it():
    def run(threshold):
        pool = PL.UniPool(cfg(hidden=4, inner_steps=20, growth_rule="gap", gap_mode="one_step", grow_gap=threshold, grow_min_age=20, grow_window=10, grow_cooldown=10, spawn_rule="probe"))
        x, y = blobs(n_per=300, classes=tuple(range(8)), noise=0.6)
        for lo in range(0, len(y) - 63, 64):
            pool.observe(x[lo:lo + 64], y[lo:lo + 64])
        return sum(1 for e in pool.events if e.kind == "grow"), pool
    fired, pool = run(0.99)
    assert fired >= 1 and max(r.module.hidden for r in pool.experts.values()) > 4
    assert run(0.0001)[0] == 0


def test_a_domain_head_is_trained_from_the_labels_and_fuses_into_the_answer():
    blocks = ((0, 3), (3, 3))
    pool = PL.UniPool(cfg(domain_blocks=blocks, domain_head="hard"))
    for x, y in stream(n_batches=10, noise=0.2):
        pool.observe(x, y)
    xt, yt = blobs(seed=11, classes=(0, 1, 2, 3, 4, 5), noise=0.2)
    p = pool.predict(xt)
    assert torch.allclose(p.probs.sum(1), torch.ones(len(p.probs)), atol=1e-4)
    assert pool.domain_lda.total > 0 and bool(((p.probs[:, :3].sum(1) < 1e-6) | (p.probs[:, 3:6].sum(1) < 1e-6)).all())      # hard fusion: every answer lies inside ONE domain's block
    dom_true = torch.where(yt < 3, 0, 1)
    dom_pred = torch.where(p.probs.argmax(1) < 3, 0, 1)
    assert float((dom_true == dom_pred).float().mean()) > 0.9


def test_the_lda_blend_changes_the_answer_and_its_storage_is_counted_in_the_pools_cost():
    base, lda = PL.UniPool(cfg()), PL.UniPool(cfg(lda_blend=0.5))
    for x, y in stream():
        base.observe(x, y)
        lda.observe(x, y)
    xt, _ = blobs(seed=9, classes=(0, 1, 2, 3, 4, 5), noise=0.4)
    assert not torch.allclose(base.predict(xt).probs, lda.predict(xt).probs, atol=1e-4)
    assert lda.extra_cost(D)[2] > base.extra_cost(D)[2] + D * D                                  # at least one expert's pooled scatter is stored


def test_bad_settings_are_refused():
    with pytest.raises(ValueError):
        cfg(gap_mode="sideways")
    with pytest.raises(ValueError):
        cfg(domain_head="maybe", domain_blocks=((0, 3),))
    with pytest.raises(ValueError):
        cfg(domain_head="hard")                                                                   # a domain head needs the label blocks
    with pytest.raises(ValueError):
        cfg(lda_blend=1.5)


# ---------------------------------------------------------------------------------------------------------------------------------- the head as a novelty reader
def test_the_density_head_distance_is_small_for_taught_classes_and_large_for_anything_else_and_zero_at_a_class_mean():
    x, y = blobs(n_per=300)                                                                          # many more samples than dimensions: a Mahalanobis distance needs a covariance that is estimated
    h = DN.DensityHead(D, 12)
    with pytest.raises(ValueError):
        h.distance(x)                                                                                # a head that has seen nothing has no notion of near
    h.update(x, y)
    xt, _ = blobs(n_per=60, seed=1)
    unseen, _ = blobs(seed=3, classes=(5, 6, 7))
    rand = F.normalize(torch.randn(120, D, generator=torch.Generator().manual_seed(9)), dim=1)
    d_known, d_unseen, d_rand = h.distance(xt), h.distance(unseen), h.distance(rand)
    assert d_known.shape == (len(xt),) and bool((d_known >= 0).all())
    ranks = lambda far: float((far[:, None] > d_known[None, :]).float().mean())                      # chance that a foreign point is farther than a taught-class point
    assert ranks(d_unseen) > 0.95 and ranks(d_rand) > 0.95
    mu = (h.s[h.seen()] / h.n[h.seen(), None]).float()
    assert float(h.distance(mu).max()) < 1e-2 * float(d_unseen.mean())                               # the nearest class mean is at distance zero (up to float rounding)
    h.update(unseen, torch.full((len(unseen),), 5))                                                  # teaching a class moves it from far to near: the cache is not stale
    assert float(h.distance(unseen).mean()) < float(d_unseen.mean()) - 10.0
