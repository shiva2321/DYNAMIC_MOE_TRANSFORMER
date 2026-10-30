"""P4 / improvement 3: the refusal layer wired to the live pool and tested on the stream's own messages.

A pool is trained on the sequential stream. Then every text message of its test sets (the arithmetic regimes are digit codes, not messages) and the three foreign sets (stories, wiki, school science
questions) are read by two readers: the POOL (novelty = 1 - its conformal p-value) and the ASKING READER (six teacher-named domain cards plus "none of these"; unclaimed = 1 - probability of its
best card). A sample of ``CAL_N`` verified known messages sets each reader's cut and the readers are fused by "either" at level ``Q`` (unimoe/refusal.py). The calibration sample is drawn
``DRAWS`` times from the known messages; the rest of the known messages are the evaluation. Nothing is tuned here: the level and the sample size are fixed before the first number.

    python -m unimoe.p4_refusal run      # train, read, save the scores (cached by code stamp)
    python -m unimoe.p4_refusal report   # decisions and metrics, written to results/unimoe/p4/REFUSAL.md
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from . import density as DN
from . import domain_routing as DR
from . import experiment as E
from . import p4 as P4
from . import refusal as RF
from . import runlib
from . import systems2 as S2
from .runner import run_system

TEXT = ("banking", "agnews", "newsgroups", "emotion", "sst2", "svamp")
SEEDS = (500, 501, 502)
CAL_N, Q, DRAWS, EVAL_CAP, N_OOD = 60, 0.05, 5, 150, 300
OUT = E.OUT / "p4_refusal"
SYSTEMS = ("pool flag (built-in, uncalibrated)", "pool calibrated", "reader claim (uncalibrated)", "reader calibrated", "either calibrated (pool + reader)", "density calibrated (Mahalanobis)",
           "either calibrated (pool + reader + density)")


def recipe_for(name: str) -> dict:
    base = dict(E.load_tuning("seq")["U-real"])
    if name == "UG0":
        return base
    t = P4.load_tuning()
    return {**base, "grow_gap": t["theta"], "lda_blend": t["seq"]["lda_blend"], **t["seq"]["domain"]}


def train_live(name: str, stream, seed: int):
    holder: List[object] = []
    blocks = [tuple(v) for v in stream.label_blocks.values()]

    def make():
        holder.append(S2.make_system2(name, stream.n_classes, seed, recipe_for(name), blocks))
        return holder[0]

    run_system(make, stream, seed, system=name)
    return holder[0]


def asking_reader(stream, doms):
    cards = DR.domain_cards(doms, DR.TEACHER_NAMES, TEXT)
    import reader as RD                                                                     # routing_general/reader.py; domain_cards() put it on the path
    return RD.CardReader(cards, with_exemplars=True, cache_dir=E.OUT / "ask")


def read(pool, reader, x: torch.Tensor, texts: List[str]) -> Dict[str, np.ndarray]:
    with torch.no_grad():
        p = pool.predict(x)
    pr = np.asarray(reader.probabilities(texts))
    return {"pool_nov": (1.0 - p.p_nov).detach().numpy(), "pool_flag": p.unknown.numpy().astype(bool), "reader_unclaimed": 1.0 - pr[:, :len(TEXT)].max(1),
            "reader_none": pr.argmax(1) == len(TEXT), "pred": p.probs.argmax(1).numpy()}


def known_messages(stream, doms):
    xs, ys, texts, regs = [], [], [], []
    for rid, reg in sorted(stream.regimes.items()):
        if reg.domain not in TEXT:
            continue
        idx = np.asarray(reg.meta["test_index"])
        xs.append(reg.x_te)
        ys.append(reg.y_te)
        texts += [str(doms[reg.domain].meta["texts_test"][int(i)]) for i in idx]
        regs += [rid] * len(idx)
    return torch.cat(xs), torch.cat(ys).numpy(), texts, np.asarray(regs)


def run_one(name: str, seed: int) -> None:
    path = OUT / f"{name}_s{seed}.npz"
    stamp = P4.stamp2()
    if path.exists():
        with np.load(path) as z:
            if "stamp" in z.files and str(z["stamp"]) == stamp:
                return
    st, doms = E.streams(seed)["seq"], E.domains_for(seed)
    pool = train_live(name, st, seed)
    reader = asking_reader(st, doms)
    xk, yk, tk, rk = known_messages(st, doms)
    density = DN.DensityHead(st.d, st.n_classes)                                            # the third reader: distance to the nearest taught class, from the same labelled batches the pool saw
    for ph in st.phases:
        density.update(ph.x, ph.y)
    out = {f"known_{k}": v for k, v in read(pool, reader, xk, tk).items()}
    out["known_y"], out["known_regime"], out["known_maha"] = yk, rk, density.distance(xk).numpy()
    ood_t = DR.ood_texts(seed, N_OOD)
    for nm, texts in ood_t.items():
        xo = st.ood[nm][:len(texts)]
        assert len(xo) == len(texts), (nm, len(xo), len(texts))
        for k, v in read(pool, reader, xo, texts).items():
            out[f"ood_{nm}_{k}"] = v
        out[f"ood_{nm}_maha"] = density.distance(xo).numpy()
    runlib.atomic_save_npz(path, stamp=np.asarray(stamp), **out)
    print(f"{name} seed {seed}: {len(yk)} known messages, accuracy {float((out['known_pred'] == yk).mean()):.3f}, pool flag on known {out['known_pool_flag'].mean():.3f}, on OOD "
          + ", ".join(f"{nm} {out[f'ood_{nm}_pool_flag'].mean():.2f}" for nm in ood_t), flush=True)


def decisions(z: Dict[str, np.ndarray], draw: int, cal_n: int, q: float):
    """Per system: (refuse on the evaluation known messages, {ood name: refuse}), plus the evaluation mask."""
    rng = np.random.default_rng([draw, 0xCA1])
    n = len(z["known_y"])
    cal = np.sort(rng.choice(n, size=cal_n, replace=False))
    ev = np.ones(n, dtype=bool)
    ev[cal] = False
    keep = np.zeros(n, dtype=bool)
    for r in np.unique(z["known_regime"]):
        keep[np.flatnonzero(ev & (z["known_regime"] == r))[:EVAL_CAP]] = True
    ev = keep
    names = sorted({k[4:-len("_pool_nov")] for k in z if k.startswith("ood_") and k.endswith("_pool_nov")})

    def run(sources):
        cal_s = {"pool": z["known_pool_nov"][cal], "reader": z["known_reader_unclaimed"][cal], "density": z["known_maha"][cal]}
        kn = RF.refusal_decision(cal_s, {"pool": z["known_pool_nov"][ev], "reader": z["known_reader_unclaimed"][ev], "density": z["known_maha"][ev]}, sources, q)
        oo = {nm: RF.refusal_decision(cal_s, {"pool": z[f"ood_{nm}_pool_nov"], "reader": z[f"ood_{nm}_reader_unclaimed"], "density": z[f"ood_{nm}_maha"]}, sources, q) for nm in names}
        return kn, oo

    out = {SYSTEMS[0]: (z["known_pool_flag"][ev], {nm: z[f"ood_{nm}_pool_flag"] for nm in names}),
           SYSTEMS[1]: run(["pool"]),
           SYSTEMS[2]: (z["known_reader_none"][ev], {nm: z[f"ood_{nm}_reader_none"] for nm in names}),
           SYSTEMS[3]: run(["reader"]),
           SYSTEMS[4]: run(["pool", "reader"]),
           SYSTEMS[5]: run(["density"]),
           SYSTEMS[6]: run(["pool", "reader", "density"])}
    return out, ev, names


def evaluate(z, cal_n=CAL_N, q=Q) -> Dict[str, dict]:
    rows: Dict[str, List[dict]] = {s: [] for s in SYSTEMS}
    for d in range(DRAWS):
        dec, ev, names = decisions(z, d, cal_n, q)
        correct = z["known_pred"][ev] == z["known_y"][ev]
        for s in SYSTEMS:
            kn, oo = dec[s]
            rows[s].append(RF.selective_metrics(correct, kn, oo))
    out = {}
    for s, ms in rows.items():
        out[s] = {"false_refusal": float(np.mean([m["false_refusal"] for m in ms])), "coverage": float(np.mean([m["coverage"] for m in ms])),
                  "accuracy_all": float(np.mean([m["accuracy_all"] for m in ms])), "accuracy_answered": float(np.nanmean([m["accuracy_answered"] for m in ms])),
                  "ood_refused": {nm: float(np.mean([m["ood_refused"][nm] for m in ms])) for nm in ms[0]["ood_refused"]}}
    return out


SCORES = (("pool_nov", "pool"), ("reader_unclaimed", "asking reader"), ("maha", "density"))


def aurocs(z: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Threshold-free quality of each reader's novelty score: the chance a foreign message scores higher than a known one (known = all known text messages)."""
    from growmoe.metrics import auroc
    names = sorted({k[4:-len("_pool_nov")] for k in z if k.startswith("ood_") and k.endswith("_pool_nov")})
    out: Dict[str, Dict[str, float]] = {}
    for key, label in SCORES:
        kn = torch.from_numpy(np.asarray(z[f"known_{key if key != 'maha' else 'maha'}"], dtype=np.float64))
        out[label] = {}
        for nm in names:
            ood = torch.from_numpy(np.asarray(z[f"ood_{nm}_{key}"], dtype=np.float64))
            out[label][nm] = float(auroc(torch.cat([kn, ood]), torch.cat([torch.zeros(len(kn)), torch.ones(len(ood))])))
    return out


def report(names=("UG0", "UPLUS")) -> str:
    L = ["# P4 / I3: the refusal layer on the live pool (generated)", "",
         f"Sequential schedule, seeds {', '.join(map(str, SEEDS))}. Calibration: {CAL_N} verified known messages (mean over {DRAWS} draws), level q = {Q}, evaluation: the rest of the known text messages (<= {EVAL_CAP} per regime) and "
         f"{N_OOD} messages of each foreign set. Arithmetic regimes are digit codes and are not part of this test.", ""]
    for name in names:
        zs = [dict(np.load(OUT / f"{name}_s{s}.npz")) for s in SEEDS if (OUT / f"{name}_s{s}.npz").exists()]
        if not zs:
            continue
        for cal_n, q, tag in ((CAL_N, Q, "primary"), (200, Q, "more verified messages (200)"), (CAL_N, 0.10, "looser level (q = 0.10)")):
            res = [evaluate(z, cal_n, q) for z in zs]
            oods = list(res[0][SYSTEMS[0]]["ood_refused"])
            L += [f"## {name}: {tag}", "", "| system | false refusal | coverage | accuracy (all) | accuracy (answered) | " + " | ".join(f"{o} refused" for o in oods) + " |", "|" + "---|" * (5 + len(oods))]
            for s in SYSTEMS:
                m = lambda key: float(np.mean([r[s][key] for r in res]))
                L.append(f"| {s} | {m('false_refusal'):.3f} | {m('coverage'):.3f} | {m('accuracy_all'):.3f} | {m('accuracy_answered'):.3f} | "
                         + " | ".join(f"{float(np.mean([r[s]['ood_refused'][o] for r in res])):.3f}" for o in oods) + " |")
            L.append("")
        au = [aurocs(z) for z in zs]
        oods = list(au[0]["pool"])
        L += [f"## {name}: AUROC of each reader's novelty score (known text against each foreign set; 0.5 = chance)", "", "| reader | " + " | ".join(oods) + " |", "|" + "---|" * (1 + len(oods))]
        for label in au[0]:
            L.append(f"| {label} | " + " | ".join(f"{float(np.mean([a[label][o] for a in au])):.3f}" for o in oods) + " |")
        L.append("")
    text = "\n".join(L)
    runlib.atomic_write_text(E.OUT / "p4" / "REFUSAL.md", text)
    return text


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "report"))
    ap.add_argument("--systems", nargs="+", default=["UG0", "UPLUS"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = ap.parse_args(argv)
    if a.cmd == "run":
        for name in a.systems:
            for seed in a.seeds:
                run_one(name, seed)
    else:
        print(report(tuple(a.systems)))


if __name__ == "__main__":
    main()
