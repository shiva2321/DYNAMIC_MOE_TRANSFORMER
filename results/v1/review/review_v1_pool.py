"""Reviewer's independent re-measurement of Gemini's V-3 pool removal (scratchpad, not in the repo).
Differences to Gemini's runner: experts are matched by the ORIGINAL stream batch index of their founding batch (not by pool step), module parameters are compared by name through
dict(named_parameters()), the per-expert LdaHead and reservoir are compared too, routing agreement maps predicted ids to founding batches.
Usage: python review_v1_pool.py OUT.json seed:R1,R3 seed:R4 ...
"""
import copy, json, os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "D:/spherical_hyperspace_memory")
import torch
from costlab.env import fixed_threads
import m1.beds, v0.system
from v1.pool_removal import ExposureTracker, remove_labels, scan_traces
from v1.run_pool_experiments import REMOVAL_SETS

CUR = [0]


def instrument(pool, log):
    orig = pool.add_expert

    def add_expert(module, parent=None, state="active", meta=None):
        rec = orig(module, parent=parent, state=state, meta=meta)
        log[rec.id] = {"orig_idx": CUR[0], "birth_step": rec.birth_step, "parent": parent}
        return rec
    pool.add_expert = add_expert


def train(bed, drop=None, track=False, bump=None):
    arm = v0.system.build_v0(bed)
    log = {}
    instrument(arm.model, log)
    tracker = ExposureTracker(arm.model) if track else None
    skipped = 0
    stream = []
    for idx, (x, y, *_r) in enumerate(bed.walk_full(64)):
        CUR[0] = idx
        stream.append((x, y))
        n_before = arm.model._next_id
        if drop is None:
            arm.observe(x, y)
        else:
            keep = ~torch.isin(y, torch.tensor(drop))
            if bool(keep.any()):
                arm.observe(x[keep], y[keep])
                if bump and idx in bump and arm.model._next_id == n_before:
                    arm.model._next_id += 1       # id-aligned control: the full pool created an (exposed) expert here, the never-trained one did not; keep later ids equal
            else:
                skipped += 1
                if os.environ.get("ALIGN") == "1":
                    arm.model.tick()          # advance the pool clock for a batch that is entirely removed, so step-keyed seeds line up
    return arm, log, tracker, stream, skipped


def maxdiff(a, b):
    return float((a.double() - b.double()).abs().max().item()) if a.numel() else 0.0


def compare_experts(arm_r, log_r, arm_n, log_n):
    pr, pn = arm_r.model, arm_n.model
    by_idx_n = {}
    for eid, v in log_n.items():
        if eid in pn.experts:
            by_idx_n.setdefault(v["orig_idx"], []).append(eid)
    rows = []
    for eid, rec in sorted(pr.experts.items()):
        i = log_r[eid]["orig_idx"]
        cands = by_idx_n.get(i, [])
        row = {"eid": eid, "orig_idx": i, "birth_step": rec.birth_step, "parent": log_r[eid]["parent"], "n_candidates": len(cands)}
        if len(cands) != 1:
            row["matched"] = False
            rows.append(row)
            continue
        rn = pn.experts[cands[0]]
        row.update({"matched": True, "never_eid": rn.id, "never_birth_step": rn.birth_step})
        pa, pb = dict(rec.module.named_parameters()), dict(rn.module.named_parameters())
        row["n_params_compared"] = len(pa)
        row["module_max_diff"] = max(maxdiff(pa[k], pb[k]) for k in pa) if pa and pa.keys() == pb.keys() and all(pa[k].shape == pb[k].shape for k in pa) else None
        la, lb = rec.meta.get("lda"), rn.meta.get("lda")
        if la is not None and lb is not None:
            row["lda_n_diff"] = maxdiff(la.n, lb.n); row["lda_s_diff"] = maxdiff(la.s, lb.s); row["lda_S_diff"] = maxdiff(la.S, lb.S); row["lda_total"] = [la.total, lb.total]
        row["replay_n"] = [rec.replay.n, rn.replay.n]
        ya = torch.sort(rec.replay.y[:rec.replay.n])[0]; yb = torch.sort(rn.replay.y[:rn.replay.n])[0]
        row["replay_labels_equal"] = bool(ya.shape == yb.shape and bool((ya == yb).all()))
        sa, sb = rec.meta["sig"], rn.meta["sig"]
        row["sig_n"] = [int(sa.n), int(sb.n)]
        row["sig_protos_equal_shape"] = bool(sa.protos is not None and sb.protos is not None and sa.protos.shape == sb.protos.shape)
        if row["sig_protos_equal_shape"]:
            row["sig_protos_max_diff"] = maxdiff(sa.protos, sb.protos)
        rows.append(row)
    return rows


def routing_by_founding(arm_r, log_r, arm_n, log_n, bed, labels):
    keepx = []
    for r in sorted(bed.test_sets.keys()):
        tx, ty = bed.test_sets[r]
        m = ~torch.isin(ty, torch.tensor(labels))
        if bool(m.any()):
            keepx.append(tx[m])
    X = torch.cat(keepx)
    er = arm_r.model.predict(X).expert.cpu().tolist()
    en = arm_n.model.predict(X).expert.cpu().tolist()
    same_id = sum(1 for a, b in zip(er, en) if a == b) / len(er)
    same_found = sum(1 for a, b in zip(er, en) if log_r[a]["orig_idx"] == log_n[b]["orig_idx"]) / len(er)
    return {"agreement_by_expert_id_(Gemini)": same_id, "agreement_by_founding_batch": same_found}


def main():
    out_path = sys.argv[1]
    jobs = []
    for a in sys.argv[2:]:
        seed, rids = a.split(":")
        jobs.append((int(seed), rids.split(",")))
    res = {}
    with fixed_threads(n=2, cpus=[8, 10]):
        for seed, rids in jobs:
            t0 = time.perf_counter()
            bed = m1.beds.build_bed("B3seq", 22, seed)
            arm_f, log_f, tracker, stream, _ = train(bed, None, True)
            merges = [e.fields for e in arm_f.model.events.of_kind("merge")]
            print(f"seed {seed} full trained {time.perf_counter()-t0:.0f}s experts={len(arm_f.model.experts)} merges={len(merges)}", flush=True)
            for rid in rids:
                labels = REMOVAL_SETS[rid]
                x_rem = torch.cat([x[torch.isin(y, torch.tensor(labels))] for x, y in stream])
                y_rem = torch.cat([y[torch.isin(y, torch.tensor(labels))] for x, y in stream])
                arm_r = copy.deepcopy(arm_f)
                cert = remove_labels(arm_r.model, labels, data=(x_rem, y_rem), tracker=tracker)
                bump = {log_f[e]["orig_idx"] for e in cert["exposure_summary"]["dropped_experts"]} if os.environ.get("BUMP") == "1" else None
                arm_n, log_n, _, _, skipped = train(bed, labels, False, bump)
                rows = compare_experts(arm_r, log_f, arm_n, log_n)
                matched = [r for r in rows if r["matched"]]
                rec = {
                    "seed": seed, "rid": rid, "labels": labels, "batches_total": len(stream), "batches_skipped_in_never": skipped,
                    "full_experts": len(arm_f.model.experts), "never_experts": len(arm_n.model.experts), "dropped": cert["exposure_summary"]["dropped_experts"],
                    "n_collateral": cert["exposure_summary"]["n_collateral_classes_lost"], "certified_clean": cert["certified_clean"],
                    "trace_audit": {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ("count", "details")}) for k, v in cert["trace_audit"].items()},
                    "retained": len(rows), "matched_by_founding_batch": len(matched),
                    "matched_module_diff_zero": sum(1 for r in matched if r.get("module_max_diff") == 0.0),
                    "matched_module_diff_max": max([r["module_max_diff"] for r in matched if r.get("module_max_diff") is not None] or [None]),
                    "matched_lda_S_diff_zero": sum(1 for r in matched if r.get("lda_S_diff") == 0.0),
                    "n_params_compared_per_expert": sorted({r.get("n_params_compared") for r in matched}),
                    "routing": routing_by_founding(arm_r, log_f, arm_n, log_n, bed, labels) if arm_r.model.experts else None,
                    "merges_in_full": merges,
                    "experts": rows,
                    "lineage_full": {str(k): v["parent"] for k, v in log_f.items()},
                }
                res[f"{seed}_{rid}"] = rec
                print(f"  {rid}: dropped={rec['dropped']} retained={rec['retained']} matched={rec['matched_by_founding_batch']} module_zero={rec['matched_module_diff_zero']} "
                      f"module_max={rec['matched_module_diff_max']} lda_S_zero={rec['matched_lda_S_diff_zero']} skipped={skipped} clean={rec['certified_clean']} routing={rec['routing']}", flush=True)
                with open(out_path, "w", encoding="utf-8") as fh:
                    json.dump(res, fh, indent=1)
            print(f"seed {seed} done {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
