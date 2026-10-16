"""Stage 4.1, optional part: a signature that never holds the same point twice (`VmfSignature(dedupe=True)`, switched on in the pool by `PhyloCfg.sig_dedupe`, OFF by default).

Why it exists: when an expert is founded its cluster goes into its signature, then the cluster's samples are assigned to it and the training loop adds them again. The signature holds every founding
point twice, each calibration score becomes the "twin" value, and the expert's own region gets the smallest possible p-value. The x4 infant relaxation hides that for a few batches."""

import numpy as np
import torch
import torch.nn.functional as F

from growmoe.config import PhyloCfg
from growmoe.pools.phylo import PhyloMoE
from growmoe.signature import VmfSignature
from growmoe.vmf import sample_vmf

D, C = 16, 4


def cluster(n, seed, kappa=60.0, mu_seed=0):
    mu = F.normalize(torch.from_numpy(np.random.default_rng(mu_seed).standard_normal(D)).float(), dim=0)
    return sample_vmf(mu, kappa, n, np.random.default_rng(seed))


def test_a_signature_keeps_a_repeated_point_unless_asked_not_to():
    x = cluster(64, 1)
    plain = VmfSignature(D)
    plain.update(x)
    plain.update(x)
    assert plain.n == 128 and plain.seen == 128                              # the shipped behaviour


def test_with_dedupe_a_repeated_point_is_one_prototype():
    x = cluster(64, 1)
    sig = VmfSignature(D, dedupe=True)
    sig.update(x)
    sig.update(x)
    assert sig.n == 64 and sig.seen == 64


def test_dedupe_drops_only_exact_repeats():
    x, y = cluster(64, 1), cluster(64, 2, mu_seed=1)
    sig = VmfSignature(D, dedupe=True)
    sig.update(x)
    sig.update(torch.cat([x[:10], y]))                                         # 10 repeats among 64 new points
    assert sig.n == 128 and sig.seen == 128
    close = x[:5] + 1e-2 * torch.randn(5, D, generator=torch.Generator().manual_seed(3))
    sig.update(close)                                                          # close is not the same
    assert sig.n == 133


def test_a_doubled_cluster_ruins_the_calibration_and_dedupe_keeps_it():
    x, fresh = cluster(64, 1), cluster(200, 9)
    doubled, clean = VmfSignature(D), VmfSignature(D, dedupe=True)
    for sig in (doubled, clean):
        sig.update(x)
        sig.update(x)
    assert float(doubled.pvalue(fresh).mean()) < 0.02                        # every fresh point of the same region sits at the floor 1/(n+1)
    assert float(clean.pvalue(fresh).mean()) > 0.3                           # about 0.5 when the calibration is honest


def test_a_signature_pickled_before_the_flag_existed_still_updates():
    sig = VmfSignature(D)
    del sig.__dict__["dedupe"]
    sig.update(cluster(8, 1))
    assert sig.n == 8


# ---------------------------------------------------------------------------------------------------- the pool

def teacher(seed):
    return torch.randn(D, C, generator=torch.Generator().manual_seed(seed))


def region_batch(centre_seed, W, n, seed, scale=8.0):
    centre = F.normalize(torch.randn(1, D, generator=torch.Generator().manual_seed(1000 + centre_seed)), dim=1)
    noise = torch.randn(n, D, generator=torch.Generator().manual_seed(seed))
    return F.normalize(scale * centre + noise, dim=1), (F.normalize(noise, dim=1) @ W).argmax(1)


def make_pool(**kw):
    cfg = dict(d=D, hidden=32, n_classes=C, seed=3, q=0.02, alpha_spawn=0.01, m_min=32, inner_steps=5, route_rule="accept_then_logf", set_gate=False, group_vote=False, spawn_rule="transfer", max_experts=64)
    cfg.update(kw)
    return PhyloMoE(PhyloCfg(**cfg))


def learn(pool, centre, W, batches, start=0):
    for k in range(batches):
        pool.observe(*region_batch(centre, W, 64, 500 + 37 * centre + start + k))


def test_the_flag_is_off_by_default_and_reaches_every_signature_the_pool_builds():
    assert PhyloCfg().sig_dedupe is False
    pool = make_pool(sig_dedupe=True)
    learn(pool, 0, teacher(1), 6)
    pool.observe(*region_batch(1, teacher(2), 64, 2001))
    assert len(pool.experts) == 2 and all(r.meta["sig"].dedupe for r in pool.experts.values())


def test_the_founding_cluster_enters_the_new_experts_signature_once_with_the_flag_and_twice_without():
    sizes = {}
    for flag in (False, True):
        pool = make_pool(sig_dedupe=flag)
        learn(pool, 0, teacher(1), 6)
        pool.observe(*region_batch(1, teacher(2), 64, 2001))                   # the first batch of a new region founds expert 1 from its 64 novel samples
        assert len(pool.experts) == 2
        sizes[flag] = pool.experts[1].meta["sig"].n
    assert sizes == {False: 128, True: 64}


def test_without_twins_a_regime_does_not_found_an_expert_per_batch_even_with_no_infant_relaxation():
    counts = {}
    for flag in (False, True):
        pool = make_pool(sig_dedupe=flag, infant_relax=1.0)
        learn(pool, 0, teacher(1), 6)
        learn(pool, 1, teacher(2), 6)                                           # six batches of the same new region
        counts[flag] = len(pool.experts)
    assert counts[False] > 2                                                    # the defect: the region's own next batches look new and each founds an expert
    assert counts[True] == 2


def test_without_twins_and_without_relaxation_a_new_region_right_after_a_spawn_is_still_seen_as_new():
    for relax, expect_blind in ((4.0, True), (1.0, False)):
        pool = make_pool(sig_dedupe=True, infant_relax=relax)
        learn(pool, 0, teacher(1), 6)
        learn(pool, 1, teacher(2), 2)                                           # region 1's expert is an infant now
        pool.observe(*region_batch(2, teacher(3), 64, 2002))                    # a third region arrives while it is
        assert (len(pool.experts) == 2) is expect_blind                         # x4: the infant takes the batch; x1: the pool founds expert 2
