"""Task 11, tuning the standard detectors on seeds that are not report seeds: one result file per (reader, detector, setting, seed), resumable, one setting chosen per detector."""

import json

import numpy as np
import pytest

import assignments
import reps
import run_e2 as R
import tune_baselines as T

TINY = R.Sizes(counts=(("solid", 60), ("thin", 8)), other_teach=45, cal=30, adapt_per_family=2, test_heldout=12, test_taught=8, twins=12, traps=12, others=15,
               unseen_content=10, tangled=6, batches=4, batch_sizes=(1, 4), background=40, map_cap=40, qs=(0.05,))
GRID = {"knn_distance": [{"k": 5}, {"k": 1}], "energy": [{"C": 1.0}, {"C": 0.01}], "mahalanobis": [{}]}


def fake_competence(pool, skill, count, seed):
    return {"accuracy": 0.97 if count >= 60 else 0.5, "chance": 0.1}


@pytest.fixture(autouse=True)
def small_thresholds(monkeypatch):
    for k, v in dict(solid_min_count=50, thin_max_count=20).items():
        monkeypatch.setitem(assignments.THRESHOLDS, k, v)


def run(tmp_path, seeds=(150, 151), **kw):
    return T.run_tuning("R0_hash", list(seeds), "dev", GRID, TINY, tmp_path, rep=reps.get_rep("R0_hash"), competence_fn=fake_competence, cache_dir=None, **kw)


def test_setting_names_are_file_safe_and_distinct():
    assert T.setting_name({}) == "default" and T.setting_name({"k": 5}) == "k=5"
    assert T.setting_name({"hidden": 8, "epochs": 100}) == "epochs=100,hidden=8"
    assert len({T.setting_name(p) for p in T.GRID["autoencoder_descriptor"]}) == len(T.GRID["autoencoder_descriptor"])


def test_every_detector_has_a_grid_and_the_default_is_in_it():
    assert set(T.GRID) == set(R.BL.BASELINE_NAMES)
    defaults = {"knn_distance": {"k": 10}, "vmf_conformal": {"max_protos": 256}, "classifier_confidence": {"C": 1.0}, "energy": {"C": 1.0}, "autoencoder_descriptor": {"hidden": 32, "epochs": 300}}
    for name, d in defaults.items():
        assert d in T.GRID[name], name
    assert T.GRID["mahalanobis"] == [{}] and T.GRID["relative_mahalanobis"] == [{}]


def test_tuning_writes_one_file_per_setting_and_seed_and_summarises(tmp_path):
    summary = run(tmp_path)
    files = sorted(p.name for p in tmp_path.glob("*.json"))
    assert len(files) == 2 * (2 + 2 + 1)                                       # seeds x settings
    assert set(summary) == set(GRID) and all(isinstance(r["mean_auroc"], float) and r["n_seeds"] == 2 for rows in summary.values() for r in rows)
    rec = json.loads((tmp_path / "R0_hash__knn_distance__k=5__s150.json").read_text(encoding="utf-8"))
    assert {"auroc_taught_vs_untaught", "unseen_recall", "false_unseen_heldout", "code_stamp", "params"} <= set(rec) and rec["params"] == {"k": 5}


def test_a_finished_cell_is_not_redone_and_a_stale_one_is(tmp_path):
    run(tmp_path)
    path = tmp_path / "R0_hash__knn_distance__k=5__s150.json"
    before = path.stat().st_mtime_ns
    run(tmp_path)
    assert path.stat().st_mtime_ns == before
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["code_stamp"] = "old"
    path.write_text(json.dumps(rec), encoding="utf-8")
    run(tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["code_stamp"] != "old"
    path.write_text('{"auroc', encoding="utf-8")
    run(tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["params"] == {"k": 5}


def test_the_best_mean_auroc_wins_and_a_near_tie_goes_to_the_default():
    rows = {"knn_distance": [{"setting": "k=10", "params": {"k": 10}, "mean_auroc": 0.800, "n_seeds": 5, "is_default": True},
                             {"setting": "k=5", "params": {"k": 5}, "mean_auroc": 0.801, "n_seeds": 5, "is_default": False},
                             {"setting": "k=50", "params": {"k": 50}, "mean_auroc": 0.700, "n_seeds": 5, "is_default": False}],
            "energy": [{"setting": "C=1.0", "params": {"C": 1.0}, "mean_auroc": 0.60, "n_seeds": 5, "is_default": True},
                       {"setting": "C=0.01", "params": {"C": 0.01}, "mean_auroc": 0.70, "n_seeds": 5, "is_default": False}]}
    chosen = T.choose(rows)
    assert chosen == {"knn_distance": {}, "energy": {"C": 0.01}}                # within 0.002 of the best: the default stands (an empty dict means no override)


def test_the_report_names_the_chosen_setting_and_the_seeds(tmp_path):
    summary = run(tmp_path)
    text = T.report_tuning("R0_hash", [150, 151], summary, T.choose(summary))
    assert "R0_hash" in text and "150" in text and "chosen" in text.lower() and "knn_distance" in text and "mahalanobis" in text
    with pytest.raises(ValueError, match="report seeds"):
        T.run_tuning("R0_hash", [100], "dev", GRID, TINY, tmp_path)               # tuning on a report seed is refused
