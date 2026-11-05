"""
Diagnostic `dmoe_v2_fm_bwt_diagnostic`: why did BWT rise when fast memory (FM) was added?
Protocol: results/dmoe_v2_fm_bwt_diagnostic/PREREG.md (written before running).
Runs on the hard-control task (generate_hard_domain_batch), 60 steps/domain, same wrapper settings.
"""
import argparse
import datetime
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_dmoe_v2_easy_eval as easy  # noqa: E402
import run_dmoe_v2_hard_control_eval as hc  # noqa: E402
from dmoe.chunker import BYTE_PAD_ID  # noqa: E402
from dmoe.continual import ContinualLearner  # noqa: E402
from dmoe.fast_memory import MultiSlotExactLdaMemory  # noqa: E402

EXP_ID = "dmoe_v2_fm_bwt_diagnostic"
OUT_DIR = ROOT / "results" / EXP_ID
STEPS = 60
BLEND = 0.55
SWEEP = [0.0, 0.25, 0.55, 0.75, 1.0]
DOM = ["A", "B", "C"]


def _batch(d: int, seed: int):
    return hc.generate_hard_domain_batch(d, batch_size=64, seq_len=16, seed=seed * 100000 + 95000 + d)


@torch.no_grad()
def readouts(model, learner: ContinualLearner, d: int, seed: int, fm=None, weights=(BLEND,)):
    x, y = _batch(d, seed)
    model.eval()
    out = learner._forward_replay(x, "byte_chunk")
    ls = F.log_softmax(out["output"], dim=-1)
    feats = learner._extract_fast_memory_features(out)
    lf = F.log_softmax((fm or learner.fast_memory).predict_slot_logits(feats), dim=-1)
    res = {}
    yf = y.reshape(-1)

    def score(lg):
        flat = lg.reshape(-1, 256)
        return F.cross_entropy(flat, yf).item(), (flat.argmax(-1) == yf).float().mean().item()

    res["slow"] = score(ls)
    res["fast"] = score(lf)
    for w in weights:
        res[f"blend@{w}"] = score((1.0 - w) * ls + w * lf)
    return res


def run_condition(name: str, seed: int) -> Dict[str, float]:
    model = hc._make_dmoe_model(seed) if name == "dMoE_v2_Full" else easy._make_dense_for_learner(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3)
    learner = ContinualLearner(
        model, opt, replay_buffer_capacity=128, replay_batch_size=8, distillation_weight=0.5,
        maturity_age=40, mutation_check_interval=25, enable_fast_memory=True,
        fast_memory_blend=BLEND, fast_memory_reg=0.25,
    )
    m: Dict[str, float] = {}
    probe_x, _ = _batch(0, seed)
    feats_after_a = None
    for d in range(3):
        for st in range(STEPS):
            x, y = hc.generate_hard_domain_batch(d, 16, 16, seed * 10000 + d * 1000 + st)
            learner.step(x, y, output_head="byte_chunk")
        if d < 2:
            for dd in range(d + 1):
                r = readouts(model, learner, dd, seed)
                for k, (l, a) in r.items():
                    kk = "blend" if k.startswith("blend") else k
                    m[f"after{DOM[d]}_{DOM[dd]}_{kk}_loss"] = l
                    m[f"after{DOM[d]}_{DOM[dd]}_{kk}_acc"] = a
        if d == 0:
            model.eval()
            with torch.no_grad():
                feats_after_a = learner._extract_fast_memory_features(learner._forward_replay(probe_x, "byte_chunk"))

    # consistency: manual blend == learner.predict_logits
    x0, _ = _batch(0, seed)
    manual = readouts(model, learner, 0, seed)
    pl = learner.predict_logits(x0, output_head="byte_chunk")
    ls_chk = F.cross_entropy(pl.reshape(-1, 256), _batch(0, seed)[1].reshape(-1)).item()
    m["blend_consistency_abs_loss_diff"] = abs(ls_chk - manual["blend@0.55"][0])

    # final readouts (online FM)
    final: Dict[str, List[float]] = {}
    for dd in range(3):
        r = readouts(model, learner, dd, seed, weights=tuple(SWEEP))
        for k, (l, a) in r.items():
            if k == "blend@0.55":
                kk = "blend"
            elif k.startswith("blend@"):
                kk = None
            else:
                kk = k
            if kk is not None:
                m[f"end_{DOM[dd]}_{kk}_loss"] = l
                m[f"end_{DOM[dd]}_{kk}_acc"] = a
            final.setdefault(k, []).append((l, a))
    for k, v in final.items():
        if k.startswith("blend@"):
            w = k.split("@")[1]
            m[f"sweep_w{w}_avg_loss"] = sum(p[0] for p in v) / 3
            m[f"sweep_w{w}_avg_acc"] = sum(p[1] for p in v) / 3
    for r in ("slow", "fast", "blend"):
        m[f"end_avg_{r}_loss"] = sum(m[f"end_{D}_{r}_loss"] for D in DOM) / 3
        m[f"end_avg_{r}_acc"] = sum(m[f"end_{D}_{r}_acc"] for D in DOM) / 3
        m[f"bwt_{r}"] = 0.5 * ((m[f"end_A_{r}_loss"] - m[f"afterA_A_{r}_loss"])
                                + (m[f"end_B_{r}_loss"] - m[f"afterB_B_{r}_loss"]))
        m[f"afterstage_avg_{r}_loss_AB"] = 0.5 * (m[f"afterA_A_{r}_loss"] + m[f"afterB_B_{r}_loss"])
        m[f"end_AB_avg_{r}_loss"] = 0.5 * (m[f"end_A_{r}_loss"] + m[f"end_B_{r}_loss"])

    # drift of fixed domain-A probe features
    with torch.no_grad():
        model.eval()
        feats_end = learner._extract_fast_memory_features(learner._forward_replay(probe_x, "byte_chunk"))
        m["feature_drift_rel_l2"] = ((feats_end - feats_after_a).norm() / feats_after_a.norm()).item()

    # refit FM from all training batches with final network features (training data only)
    fm2 = MultiSlotExactLdaMemory(
        d_in=learner.fast_memory.d_in, num_classes=256, num_slots=4, reg=learner.fast_memory_reg)
    model.eval()
    with torch.no_grad():
        for d in range(3):
            for st in range(STEPS):
                x, y = hc.generate_hard_domain_batch(d, 16, 16, seed * 10000 + d * 1000 + st)
                feats = learner._extract_fast_memory_features(learner._forward_replay(x, "byte_chunk"))
                fm2.absorb_slots(feats, y, ignore_index=BYTE_PAD_ID)
    rl, ra = [], []
    for dd in range(3):
        r = readouts(model, learner, dd, seed, fm=fm2)
        rl.append(r["blend@0.55"][0])
        ra.append(r["blend@0.55"][1])
        m[f"refit_end_{DOM[dd]}_fast_loss"] = r["fast"][0]
    m["refit_end_avg_blend_loss"] = sum(rl) / 3
    m["refit_end_avg_blend_acc"] = sum(ra) / 3
    m["paired_refit_minus_online_blend_loss"] = m["refit_end_avg_blend_loss"] - m["end_avg_blend_loss"]
    m["paired_refit_minus_online_blend_acc"] = m["refit_end_avg_blend_acc"] - m["end_avg_blend_acc"]
    return m


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        return f"unavailable: {err}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(20))
    summary = {}
    raw = {}
    for name in ("dMoE_v2_Full", "Regular+Replay+FM"):
        per_seed = []
        for s in seeds:
            per_seed.append(run_condition(name, s))
            print(f"{name} seed {s} done", flush=True)
        raw[name] = per_seed
        summary[name] = {k: hc._summarize([r[k] for r in per_seed]) for k in per_seed[0]}
    payload = {
        "meta": {
            "exp_id": EXP_ID, "git_commit": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "torch_version": torch.__version__, "python": platform.python_version(),
            "command": " ".join(sys.argv), "seeds": seeds, "steps_per_domain": STEPS,
            "blend": BLEND, "sweep": SWEEP,
        },
        "summary": summary,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name, s in summary.items():
        print(f"=== {name} ===")
        for r in ("slow", "fast", "blend"):
            print(f"  {r:6s} end_avg_loss={s[f'end_avg_{r}_loss']['mean']:.4f} end_avg_acc={s[f'end_avg_{r}_acc']['mean']:.4f} "
                  f"after(A,B)_loss={s[f'afterstage_avg_{r}_loss_AB']['mean']:.4f} "
                  f"end(A,B)_loss={s[f'end_AB_avg_{r}_loss']['mean']:.4f} BWT={s[f'bwt_{r}']['mean']:+.4f}")
        print(f"  refit-online blend loss diff: {s['paired_refit_minus_online_blend_loss']['mean']:+.4f} "
              f"± {s['paired_refit_minus_online_blend_loss']['ci95']:.4f}; drift={s['feature_drift_rel_l2']['mean']:.4f}")
        for w in SWEEP:
            print(f"  sweep w={w}: loss={s[f'sweep_w{w}_avg_loss']['mean']:.4f} acc={s[f'sweep_w{w}_avg_acc']['mean']:.4f}")
        print(f"  blend consistency max abs loss diff: {s['blend_consistency_abs_loss_diff']['max']:.2e}")


if __name__ == "__main__":
    main()
