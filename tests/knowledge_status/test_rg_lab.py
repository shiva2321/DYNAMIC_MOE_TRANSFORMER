"""The general routing lab (routing_general/lab_general.py) with stand-ins for the language model and the encoder."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

_KS = Path(__file__).resolve().parents[2] / "research" / "knowledge_status"
sys.path.insert(0, str(_KS / "routing_general"))

import domains as DM  # noqa: E402
import lab_general as LG  # noqa: E402

NAMES = ["card_arrival", "age_limit", "atm_support", "top_up", "exchange_rate", "lost_card"]


def table(n_each, tag):
    texts, labels = [], []
    for c in range(len(NAMES)):
        for j in range(n_each):
            texts.append(f"{tag} message {j} about {NAMES[c]} please")
            labels.append(c)
    return texts, labels


def make_world(seed=0):
    return DM.build_labelled_world("toy", table(40, "train"), table(20, "test"), NAMES, n_taught=2, n_teach=10, n_cal=10, n_known=20, n_unknown=20, seed=seed)


def fake_ask_for(world):
    k = len(world.taught)
    keys = [n.replace(" ", "_") for n in world.taught]

    def ask(texts):
        out = np.full((len(texts), k + 1), 0.01)
        for i, t in enumerate(texts):
            hit = [j for j, key in enumerate(keys) if key in t]
            out[i, hit[0] if hit else k] = 0.98
        return out / out.sum(axis=1, keepdims=True)
    return ask


def fake_encode(texts):
    out = np.zeros((len(texts), 24))
    for i, t in enumerate(texts):
        for w in t.split():
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 24] += 1.0
    return out


FOREIGN = [f"the weather will be mild this {d} and the harvest looks fine" for d in ("week", "month", "year", "autumn", "spring", "summer", "winter", "season")] * 3


def test_run_cell_with_a_perfect_reader_claims_known_and_leaves_the_rest(tmp_path):
    w = make_world()
    rec = LG.run_cell(w, FOREIGN, ask=fake_ask_for(w), encode=fake_encode, cache_dir=tmp_path)
    for arm in LG.ARMS:
        t = rec["reader"][arm]["tau"]["0.5"]
        assert t["known_claimed"] == 1.0 and t["known_right"] == 1.0 and t["right_given_claimed"] == 1.0
        assert t["unknown_unclaimed"] == 1.0 and t["foreign_unclaimed"] == 1.0
        assert rec["reader"][arm]["top1_known"] == 1.0
        assert rec["reader"][arm]["op"]["auroc_unknown"] == 1.0
    assert set(rec["detectors"]) == set(LG.DETECTORS)
    assert set(rec["detectors"]["mahalanobis"]) >= {"known_flagged", "unknown_found", "foreign_found", "auroc_unknown", "auroc_foreign", "cut"}
    assert 0.0 <= rec["classifier_top1_known"] <= 1.0
    assert rec["n"] == {"cal": 20, "known": 20, "unknown": 20, "foreign": len(FOREIGN)}


def test_report_has_a_block_per_domain_with_every_arm_and_detector(tmp_path):
    w = make_world()
    rec = LG.run_cell(w, FOREIGN, ask=fake_ask_for(w), encode=fake_encode, cache_dir=tmp_path)
    rec.update({"seed": 0})
    text = LG.report({"toy": [rec, rec]}, [0, 1])
    assert "## toy (2 seeds; 2 taught classes" in text
    for arm in LG.ARMS:
        assert f"| ask, {arm} |" in text
    for name in LG.DETECTORS:
        assert f"| {name} (MiniLM) |" in text
    assert "logistic regression" in text


def test_foreign_messages_are_deterministic_and_the_right_size():
    a = LG.foreign_messages("banking77", 3, 20)
    assert a == LG.foreign_messages("banking77", 3, 20) and len(a) == 20 and len(set(a)) == 20
    assert a != LG.foreign_messages("banking77", 4, 20)


def test_extras_and_fusions_are_reported_for_every_source(tmp_path):
    w = make_world()
    traps = [f"the bank of the river looks calm near the old bridge {d}" for d in ("today", "tonight", "again", "soon", "later")]
    rec = LG.run_cell(w, FOREIGN, ask=fake_ask_for(w), encode=fake_encode, cache_dir=tmp_path, extras={"trap": traps})
    assert rec["n"]["trap"] == 5
    sources = {"ask_names", "ask_exemplars", *LG.DETECTORS}
    assert set(rec["extras"]["trap"]) >= sources | {f"{'+'.join(p)}|{r}" for p in LG.FUSIONS for r in LG.RULES}
    assert all(0.0 <= v <= 1.0 for v in rec["extras"]["trap"].values())
    assert set(rec["fused"]) == {f"{'+'.join(p)}|{r}" for p in LG.FUSIONS for r in LG.RULES}
    assert set(rec["reader"]["names"]["extras_unclaimed_tau0.5"]) == {"trap"}
    one = rec["fused"]["ask_exemplars+vmf_conformal|either"]
    assert 0.0 <= one["auroc_unknown"] <= 1.0                                                          # NOT bounded below by the better reader: an OR inherits a noisy partner's false alarms


def test_arithmetic_is_the_only_domain_that_needs_a_name_in_the_lab():
    assert set(LG.ALL_DOMAINS) == set(LG.DOMAINS) | {"arithmetic"}
