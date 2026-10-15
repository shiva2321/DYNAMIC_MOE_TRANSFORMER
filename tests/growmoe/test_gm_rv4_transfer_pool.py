"""RV4 stage 4b inside PhyloMoE: references, the verdict, extend without touching a weight, spawn with a pretrained child, merge (spec sections 5.2, 5.4, 5.5)."""

import copy
import math
from collections import deque

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from growmoe.accounting import pool_report
from growmoe.config import PhyloCfg
from growmoe.experts import MLPExpert
from growmoe.pools.phylo import PhyloMoE
from growmoe.signature import VmfSignature
from rv4_helpers import D, C, regime_batch, trained_pool, unit


@pytest.fixture(scope="module")
def base():
    pool, W = trained_pool()
    return pool, W


def _fresh(base):
    return copy.deepcopy(base[0]), base[1]


# ------------------------------------------------------------------------------------------------ the references
def test_the_reference_holds_honest_losses_of_a_learned_expert_and_is_capped(base):
    pool, _ = _fresh(base)
    ref = pool.experts[0].meta["ref"]
    assert len(ref) == 512 and ref.maxlen == 512
    assert all(math.isfinite(v) for v in ref) and sum(ref) / len(ref) < math.log(C)                     # below chance: it learned


def test_with_the_probe_rule_no_reference_is_kept_and_no_transfer_field_is_logged():
    pool, W = trained_pool(n_batches=6, spawn_rule="probe")
    assert all("ref" not in r.meta for r in pool.experts.values())
    pool._handle_cluster(*regime_batch(W, 48, 9001, perm=1), evidence=50.0)
    assert not any("p_conflict" in e.fields for e in pool.events)


def test_infants_do_not_feed_the_reference_and_actives_do():
    """Hand-built pool: expert 0 active on blob A, expert 1 an infant on blob B (untrained; only the bookkeeping is under test)."""
    d = 12
    g = np.random.default_rng(0)
    draw = lambda i, n: torch.from_numpy(unit(np.eye(d)[i] * 8 + g.standard_normal((n, d))))              # noqa: E731
    labels = lambda s: torch.randint(0, 4, (64,), generator=torch.Generator().manual_seed(s))
    pool = PhyloMoE(PhyloCfg(d=d, hidden=8, n_classes=4, seed=1, q=0.02, m_min=32, inner_steps=1, route_rule="accept_then_logf", set_gate=False, group_vote=False, spawn_rule="transfer"))
    for i, state in ((0, "active"), (1, "infant")):
        mod = MLPExpert(d_in=d, hidden=8, depth=1, n_classes=4, seed=i)
        sig = VmfSignature(d=d, max_protos=256, seed=i)
        sig.update(draw(i, 80))
        sig.fit()
        pool.add_expert(mod, parent=None if i == 0 else 0, state=state, meta={"sig": sig, "infant_calls_seen": 0 if state == "infant" else pool.cfg.infant_calls})
    for k in range(4):                                                  # infancy lasts infant_calls = 5 observe calls: at the start of the 5th the expert is active
        pool.observe(draw(1, 64), labels(k))
        assert pool.experts[1].state == "infant" and "ref" not in pool.experts[1].meta
    pool.observe(draw(1, 64), labels(9))
    assert pool.experts[1].state == "active" and len(pool.experts[1].meta["ref"]) > 0
    pool.observe(draw(0, 64), labels(10))
    assert len(pool.experts[0].meta["ref"]) > 0


# ------------------------------------------------------------------------------------------------ extend and spawn
def test_a_cluster_the_parent_already_predicts_is_absorbed_without_touching_a_weight(base):
    pool, W = _fresh(base)
    rec = pool.experts[0]
    before = {k: v.clone() for k, v in rec.module.state_dict().items()}
    seen_before, replay_before = rec.meta["sig"].seen, rec.replay.n
    x, y = regime_batch(W, 48, 9000)
    assert pool._handle_cluster(x, y, evidence=50.0) is None and sorted(pool.experts) == [0]
    ev = pool.events.of_kind("extend")[-1]
    assert ev.fields["p_conflict"] > 0.01 and ev.fields["n_ref"] == 512 and "loss_cluster" in ev.fields
    assert all(torch.equal(before[k], v) for k, v in rec.module.state_dict().items())
    assert rec.meta["sig"].seen == seen_before + 48 and rec.replay.n >= replay_before


def test_a_cluster_whose_labels_the_parent_gets_wrong_is_spawned_and_pretrained(base):
    pool, W = _fresh(base)
    x, y = regime_batch(W, 48, 9001, perm=1)
    assert pool._handle_cluster(x, y, evidence=50.0) == 1
    assert sorted(pool.experts) == [0, 1] and pool.experts[1].parent == 0
    f = pool.events.of_kind("spawn")[-1].fields
    assert f["p_conflict"] <= 0.01 and f["loss_cluster"] > f["loss_ref"] + 0.35 and f["margin"] == pytest.approx(0.25 * math.log(C))
    with torch.no_grad():
        child, parent = F.cross_entropy(pool.experts[1].module(x), y), F.cross_entropy(pool.experts[0].module(x), y)
    assert float(child) < float(parent)                                                                 # the pretraining steps taught the child the new labels


def test_the_novel_samples_near_the_parent_are_handed_to_it_on_an_extend(base):
    pool, W = _fresh(base)
    x, y = regime_batch(W, 64, 9002)
    assigned = torch.full((64,), -1, dtype=torch.long)
    pool._handle_cluster(x[:48], y[:48], evidence=50.0, novel_indices=torch.arange(64), x_all=x, assigned=assigned)
    assert pool.events.of_kind("extend") and int((assigned == 0).sum()) > 32


def test_an_unlabelled_cluster_is_handled_by_the_path_of_today(base):
    pool, W = _fresh(base)
    twin = copy.deepcopy(pool)
    twin.cfg.spawn_rule = "probe"
    x, _ = regime_batch(W, 48, 9003)
    y = torch.full((48,), -1)
    assert pool._handle_cluster(x, y, evidence=50.0) == twin._handle_cluster(x, y, evidence=50.0)
    assert [e.kind for e in pool.events] == [e.kind for e in twin.events]


def test_a_parent_with_too_few_reference_losses_falls_back_to_the_probe():
    pool, W = trained_pool(n_batches=1)                                  # the founding batch never feeds the reference
    assert "ref" not in pool.experts[0].meta
    twin = copy.deepcopy(pool)
    twin.cfg.spawn_rule = "probe"
    x, y = regime_batch(W, 48, 9004)
    assert pool._handle_cluster(x, y, evidence=50.0) == twin._handle_cluster(x, y, evidence=50.0)
    assert [e.kind for e in pool.events] == [e.kind for e in twin.events]
    assert not any("p_conflict" in e.fields for e in pool.events)


def test_labels_the_parent_has_never_seen_are_a_conflict_with_finite_event_fields(tmp_path):
    pool, W = trained_pool(n_batches=20, n_classes=6)                    # label space of 6, the teacher uses 0-3
    x, y = regime_batch(W, 48, 9005)
    y = 4 + (y % 2)                                                      # labels 4 and 5: masked in the parent
    assert pool._handle_cluster(x, y, evidence=50.0) == 1
    f = pool.events.of_kind("spawn")[-1].fields
    assert f["p_conflict"] <= 0.01 and math.isfinite(f["loss_cluster"]) and f["margin"] == pytest.approx(0.25 * math.log(4))
    pool.events.to_jsonl(tmp_path / "events.jsonl")                      # strict JSON (allow_nan=False) must accept every field


def test_the_alarm_path_leaves_the_cluster_itself_out_of_the_reference(base):
    pool, W = _fresh(base)
    rec = pool.experts[0]
    assert pool._transfer_reference(rec, 0) == list(rec.meta["ref"])
    assert pool._transfer_reference(rec, 32) == list(rec.meta["ref"])[:-32]
    x, y = regime_batch(W, 32, 9006)
    assert pool._transfer_verdict(x, y, force_parent=0).n_ref == 512 - 32
    assert pool._transfer_verdict(x, y, force_parent=None).n_ref == 512


def test_on_the_alarm_path_the_cluster_is_judged_by_its_stored_honest_losses_not_by_a_rescoring(base):
    """The competence alarm hands over the expert's own last samples, which it may already have trained on: re-scoring them with the current weights would understate how badly it did.
    Here the stored losses say 'bad' (4.0) while the samples handed over are from the function the expert knows (a re-score would say 'good'): the verdict must follow the stored losses."""
    pool, W = _fresh(base)
    rec = pool.experts[0]
    rec.meta["ref"] = deque([0.45] * 400 + [4.0] * 32, maxlen=512)
    x, y = regime_batch(W, 32, 9012)                                                   # the known function: a re-score would give about 0.45
    v = pool._transfer_verdict(x, y, force_parent=0)
    assert v.conflict and v.loss_cluster == pytest.approx(4.0) and v.loss_ref == pytest.approx(0.45) and v.n_ref == 400 and v.n_cluster == 32
    pool._handle_cluster(x, y, evidence=0.0, force_parent=0)
    assert pool.events.of_kind("spawn")[-1].fields["p_conflict"] <= 0.01 and sorted(pool.experts) == [0, 1]
    rec.meta["ref"] = deque([0.45] * 432, maxlen=512)                                  # the other way round: nothing bad was stored, whatever the samples handed over look like
    x2, y2 = regime_batch(W, 32, 9013, perm=1)
    assert not pool._transfer_verdict(x2, y2, force_parent=0).conflict


def test_the_parent_is_the_expert_with_the_lowest_loss_on_the_cluster(base):
    pool, W = _fresh(base)
    pool._handle_cluster(*regime_batch(W, 48, 9007, perm=1), evidence=50.0)         # expert 1 knows the permuted function
    pool.experts[1].meta["ref"] = deque([0.5] * 100, maxlen=512)                     # give it a reference so that it can be a parent
    x, y = regime_batch(W, 48, 9008, perm=1)
    assert pool._transfer_verdict(x, y, None).parent == 1
    x, y = regime_batch(W, 48, 9009)
    assert pool._transfer_verdict(x, y, None).parent == 0


def test_the_accounting_counts_the_reference_scalars(base):
    pool, _ = _fresh(base)
    twin = copy.deepcopy(pool)
    twin.cfg.spawn_rule = "probe"
    a, b = pool_report(pool, D), pool_report(twin, D)
    n = len(pool.experts[0].meta["ref"])
    assert a.extra_scalars == n and b.extra_scalars == 0 and a.memory_bytes - b.memory_bytes == 4 * n and a.params == b.params


def test_the_same_cluster_gives_the_same_verdict_in_two_identical_pools():
    a, W = trained_pool(n_batches=12)
    b, _ = trained_pool(n_batches=12)
    x, y = regime_batch(W, 48, 9010, perm=1)
    va, vb = a._transfer_verdict(x, y, None), b._transfer_verdict(x, y, None)
    assert (va.parent, va.conflict, va.n_ref) == (vb.parent, vb.conflict, vb.n_ref)
    assert va.loss_cluster == pytest.approx(vb.loss_cluster, rel=1e-4) and va.p == pytest.approx(vb.p)           # two runs may differ in the last bit of a threaded matmul


# ------------------------------------------------------------------------------------------------ merge
def test_a_merge_gives_the_survivor_the_references_of_both(base):
    pool, W = _fresh(base)
    pool._handle_cluster(*regime_batch(W, 48, 9011, perm=1), evidence=50.0)
    a, b = pool.experts[0], pool.experts[1]
    a.meta["ref"], b.meta["ref"] = deque([1.0] * 300, maxlen=512), deque([2.0] * 300, maxlen=512)
    pool._merge_refs(a, b)
    ref = list(a.meta["ref"])
    assert len(ref) == 512 and ref[:212] == [1.0] * 212 and ref[212:] == [2.0] * 300                     # the newest 512 of the union, the second expert's entries last


def _forced_duplicate_then_merge(seed):
    """AUDIT (spec section 5.5): merge has never fired in a benchmark. Here a duplicate of one function is forced (the probe can never accept an extend) and, after the
    infant has settled, the pool's own merge routine is asked to clean it up."""
    pool, W = trained_pool(n_batches=30, seed=seed, spawn_rule="probe", delta_forget=-1.0)
    assert pool._handle_cluster(*regime_batch(W, 48, 9100 + seed), evidence=50.0) == 1 and len(pool.experts) == 2
    for k in range(10):
        pool.observe(*regime_batch(W, 64, 9200 + 10 * seed + k))
    n_before = len(pool.experts)
    pool._attempt_merges()
    return pool, n_before


@pytest.mark.xfail(strict=True, reason="AUDIT RESULT 2026-10-15: merge cannot clean a forced duplicate (0 of 4 seeds merged, seeds 0-3). The overlap test passes (mutual coverage 0.44-0.52 against 0.2), but the "
                   "acceptance test compares experts that MEMORISED their replay with a clone that never trained on the held-out half: the clone loses 6-9 points on the validation halves "
                   "against a tolerance of 2 (delta_merge). Same flaw as the spawn probe of E-030, mirrored. A false spawn is therefore never repaired; see FAILURES and the open questions.")
@pytest.mark.parametrize("seed", [0, 1])
def test_merge_cleans_up_a_forced_duplicate(seed):
    """The audit of spec section 5.5. It FAILS today (see the xfail reason), which is the finding: the transfer rule is built to favour spawn and relies on merge for the clean-up, and
    merge cannot do it. strict=True: the day merge is repaired this test passes and the marker has to go."""
    pool, n_before = _forced_duplicate_then_merge(seed)
    assert n_before == 2 and len(pool.experts) == 1 and pool.events.of_kind("merge")
