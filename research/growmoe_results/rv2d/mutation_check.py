"""Mutation spot-check of the stage-4 decision functions (spec section 9: at least 90 % of the mutants must be killed).

    python mutation_check.py [--only router|transfer|gapgrow|phylo]

For each mutant: one exact-text replacement in a temporary copy of growmoe/, tests/growmoe/ and the golden files, then the owning tests with `pytest -x`. Killed = the tests fail.
A survivor needs either a new test (red, green) or a written reason that the mutant is equivalent.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RT = ["tests/growmoe/test_gm_rv4_router.py"]
RP = ["tests/growmoe/test_gm_rv4_router_pool.py"]
TT = ["tests/growmoe/test_gm_rv4_transfer.py"]
TP = ["tests/growmoe/test_gm_rv4_transfer_pool.py"]
GT = ["tests/growmoe/test_gm_rv4_growth.py"]
GP = ["tests/growmoe/test_gm_rv4_growth_pool.py"]
VT = ["tests/growmoe/test_gm_rv4_verdicts.py"]

MUTANTS = [   # (group, name, file, exact old text, new text, owning tests)
    ("router", "margin: > becomes >=", "growmoe/router.py", 'cv[best] > cv["density"] + margin', 'cv[best] >= cv["density"] + margin', RT),
    ("router", "margin dropped", "growmoe/router.py", 'cv[best] > cv["density"] + margin', 'cv[best] > cv["density"]', RT),
    ("router", "hysteresis: > becomes >=", "growmoe/router.py", "not cv[candidate] > cv[current] + margin", "not cv[candidate] >= cv[current] + margin", RT),
    ("router", "folds: one extra fold id", "growmoe/router.py", "torch.arange(idx.numel()) % folds", "torch.arange(idx.numel()) % (folds + 1)", RT),
    ("router", "folds: train on the held-out fold", "growmoe/router.py", "tr = ~te", "tr = te", RT),
    ("router", "softmax: class weights dropped", "growmoe/router.py", "cls_w = (n / (k * counts.clamp_min(1))).float()", "cls_w = torch.ones(k)", RT),
    ("router", "lda: empty class not masked", "growmoe/router.py", "(Z.double() @ self.w + self.b).masked_fill(~self.present, NEG_INF).float()", "(Z.double() @ self.w + self.b).float()", RT),
    ("transfer", "conflict: > margin becomes >=", "growmoe/transfer.py", "bool(p <= alpha and diff > margin)", "bool(p <= alpha and diff >= margin)", TT),
    ("transfer", "conflict: and becomes or", "growmoe/transfer.py", "bool(p <= alpha and diff > margin)", "bool(p <= alpha or diff > margin)", TT),
    ("transfer", "ties not counted", "growmoe/transfer.py", ">= obs).sum())", "> obs).sum())", TT),
    ("gapgrow", "gap formula", "growmoe/gapgrow.py", "else 1.0 - s_post / s_pre", "else s_post / s_pre", GT),
    ("gapgrow", "improvement inverted", "growmoe/gapgrow.py", "else last / prev", "else prev / last", GT),
    ("gapgrow", "plateau: >= becomes >", "growmoe/gapgrow.py", "imp is not None and imp >= plateau", "imp is not None and imp > plateau", GT),
    ("gapgrow", "age: >= becomes >", "growmoe/gapgrow.py", "self.age >= min_age and", "self.age > min_age and", GT),
    ("gapgrow", "guard: > becomes >=", "growmoe/gapgrow.py", "return after > (1.0 + eps) * before", "return after >= (1.0 + eps) * before", GT),
    ("gapgrow", "increment: >= becomes >", "growmoe/gapgrow.py", "return k if k >= min_widen else 0", "return k if k > min_widen else 0", GT),
    ("phylo", "router: restrict accepts p == q no more", "growmoe/pools/phylo.py", "(P.double() >= float(self.cfg.q))\n        best_accepted", "(P.double() > float(self.cfg.q))\n        best_accepted", RP),
    ("phylo", "router: young expert threshold", "growmoe/pools/phylo.py", "min(counts) < cfg.router_min_protos", "min(counts) <= cfg.router_min_protos", RP),
    ("phylo", "growth: gap >= becomes >", "growmoe/pools/phylo.py", "gap >= cfg.grow_gap", "gap > cfg.grow_gap", GP),
    ("phylo", "growth: budget > becomes >=", "growmoe/pools/phylo.py", "self._live_params() + added > cfg.grow_budget_params", "self._live_params() + added >= cfg.grow_budget_params", GP),
    ("phylo", "transfer: min reference < becomes <=", "growmoe/pools/phylo.py", "if len(ref) < cfg.transfer_min_ref:", "if len(ref) <= cfg.transfer_min_ref:", TP),
    ("phylo", "transfer: alarm path keeps the cluster in the reference", "growmoe/pools/phylo.py", "return ref[:-exclude] if exclude > 0 else ref", "return ref", TP),
    ("phylo", "transfer: alarm path min reference < becomes <=", "growmoe/pools/phylo.py", "len(stored) - n < cfg.transfer_min_ref", "len(stored) - n <= cfg.transfer_min_ref", TP),
    ("verdict", "S4-5: ratio against S3 even when the true count is known", "growmoe/verdicts_rv4.py", "ratio = mean_t / float(np.mean(truth)) if known else mean_t / mean_s3", "ratio = mean_t / mean_s3", VT),
    ("verdict", "S4-5: one record with a true count is enough", "growmoe/verdicts_rv4.py", "known = all(isinstance(x, (int, float)) and x > 0 for x in truth)", "known = any(isinstance(x, (int, float)) and x > 0 for x in truth)", VT),
    ("verdict", "S4-5: the ratio bar is ignored", "growmoe/verdicts_rv4.py", 'd["holds"] = bool(d["mean_diff_points"] >= bars["S4_5_gain"] and ratio <= bars["S4_5_ratio"])', 'd["holds"] = bool(d["mean_diff_points"] >= bars["S4_5_gain"])', VT),
    ("phylo", "transfer: margin scale", "growmoe/pools/phylo.py", "math.log(max(int(self.experts[eid].module.seen.sum()), 2))", "math.log(max(int(self.experts[eid].module.seen.sum()), 1))", TP),
]


def prepare(tmp: Path) -> None:
    for sub in ("growmoe", "tests/growmoe"):
        shutil.copytree(ROOT / sub, tmp / sub, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    d = tmp / "research/growmoe_results/rv2c"
    d.mkdir(parents=True)
    for f in ("golden.py", "golden_flags_off.json"):
        shutil.copy(ROOT / "research/growmoe_results/rv2c" / f, d / f)


def run_tests(tmp: Path, tests) -> bool:
    """True when the tests PASS. No timeout on purpose: when the PC sleeps the monotonic clock jumps and subprocess raised a negative TimeoutExpired in the first run."""
    p = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *tests], cwd=tmp, env={**os.environ, "PYTHONPATH": str(tmp), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
                       capture_output=True, text=True)
    return p.returncode == 0


def stamp() -> str:
    """Hash of everything a mutation result depends on: the code under test, the tests and the list of mutants."""
    h = hashlib.sha256(repr(MUTANTS).encode("utf-8"))
    for f in sorted((ROOT / "growmoe").rglob("*.py")) + sorted((ROOT / "tests" / "growmoe").glob("*.py")):
        h.update(str(f.relative_to(ROOT)).encode("utf-8"))
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def load_done(out: Path, st: str) -> list:
    """Results of an earlier, interrupted run of the SAME code, tests and mutants (so a power cut or a sleep costs one mutant, not the whole run)."""
    try:
        d = json.loads(out.read_text(encoding="utf-8"))
        return d["results"] if d.get("stamp") == st else []
    except Exception:
        return []


def save(out: Path, st: str, results: list) -> None:
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"stamp": st, "results": results}, indent=1))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    out, st = HERE / "mutation_results.json", stamp()
    results = load_done(out, st)
    done = {r["name"] for r in results}
    todo = [m for m in MUTANTS if a.only in (None, m[0]) and m[1] not in done]
    if results:
        print(f"resuming: {len(results)} mutants already judged for this code and these tests", flush=True)
    if todo:
        with tempfile.TemporaryDirectory(prefix="rv2d_mut_") as t:
            tmp = Path(t)
            prepare(tmp)
            for tests in sorted({tuple(m[5]) for m in todo}):
                assert run_tests(tmp, list(tests)), f"the unmutated tests {tests} do not pass in the copy: fix that first"
            for group, name, rel, old, new, tests in todo:
                path = tmp / rel
                src = path.read_text(encoding="utf-8")
                assert src.count(old) == 1, f"{name}: the old text occurs {src.count(old)} times in {rel}"
                path.write_text(src.replace(old, new), encoding="utf-8")
                try:
                    survived = run_tests(tmp, tests)
                finally:
                    path.write_text(src, encoding="utf-8")
                results.append({"group": group, "name": name, "killed": not survived})
                save(out, st, results)
                print(f"{'KILLED  ' if not survived else 'SURVIVED'} {group}: {name}", flush=True)
    killed = sum(r["killed"] for r in results)
    print(f"\n{killed}/{len(results)} mutants killed ({100 * killed / max(len(results), 1):.0f} %)")
    for r in results:
        if not r["killed"]:
            print("  survivor:", r["group"], "-", r["name"])


if __name__ == "__main__":
    main()
