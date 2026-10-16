"""Mechanism check for E-037: the founding cluster of an expert enters its signature twice. For the expert founded at the first batch of regime 1, before each later batch of that regime:
how many prototypes the signature holds, how many of them have an exact twin, what share of the calibration scores equals the largest one, the mean conformal p-value of the batch (what the gate
would see before the infant relaxation) and the share of the batch at the floor 1/(n+1). With and without the twin fix. No product file is touched.

    python twin_signature_diag.py            # seeds 0, 1, 2; writes twin_signature_diag_output.txt
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import torch  # noqa: E402

import healing2_study as H  # noqa: E402
from growmoe.signature import VmfSignature  # noqa: E402
from growmoe.streams_m1 import function_regimes  # noqa: E402

TOL = 1.0 - 1e-5
_ORIGINAL = VmfSignature.update


def _without_twins(self, x) -> None:
    u = self._unit(x)
    if self._n > 0:
        u = u[(u @ self.protos.T).max(dim=1).values < TOL]
    if u.shape[0]:
        _ORIGINAL(self, u)


def twins(sig) -> int:
    p = sig.protos
    d = (p @ p.T)
    d.fill_diagonal_(-2.0)
    return int((d.max(dim=1).values > TOL).sum())


def trace(seed: int, dedupe: bool, batches: int = 8) -> list:
    VmfSignature.update = _without_twins if dedupe else _ORIGINAL
    try:
        st = function_regimes(seed, n_regimes=H.REGIMES, d=H.D, n_classes=H.C, n_train=64 * 10, n_test=200)
        pool = H.make_pool("clean", seed, None)
        rows, founded_at = [], None
        for b, (k, seg, idx) in enumerate(st._walk(64, seed)):
            x, y = seg.x[idx], seg.y[idx]
            if founded_at is not None and len(pool.experts) >= 2 and b - founded_at <= batches:
                sig = pool.experts[max(pool.experts)].meta["sig"]
                xn = torch.nn.functional.normalize(x.float(), dim=1)
                cal, p = sig.cal_scores, sig.pvalue(xn)
                rows.append((b - founded_at, sig.n, twins(sig), float((cal >= cal.max() - 1e-3).float().mean()), float(p.mean()), float((p <= 1.0 / (sig.n + 1) + 1e-9).float().mean())))
            pool.observe(x, y)
            if founded_at is None and len(pool.experts) == 2:
                founded_at = b
            if founded_at is not None and b - founded_at > batches:
                break
        return rows
    finally:
        VmfSignature.update = _ORIGINAL


def main() -> None:
    torch.set_num_threads(2)
    lines = ["Newest expert (founded at the first batch of regime 1), looked at before each later batch of the regime. Pool: stage-2 gate, transfer test, nothing injected; infant_relax is left at the shipped 4,",
             "so the pool itself does not react to these numbers; they are what the signature says about a fresh batch of its own region.\n",
             "| seed | twin fix | batches since founding | prototypes | prototypes with an exact twin | share of calibration scores at the maximum | mean p-value of the batch | share of the batch at the floor 1/(n+1) |", "|---|---|---|---|---|---|---|---|"]
    for seed in (0, 1, 2):
        for dedupe in (False, True):
            for since, n, tw, top, mp, fl in trace(seed, dedupe):
                lines.append(f"| {seed} | {'yes' if dedupe else 'no'} | {since} | {n} | {tw} | {top:.2f} | {mp:.3f} | {fl:.2f} |")
    text = "\n".join(lines) + "\n"
    (HERE / "twin_signature_diag_output.txt").write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
