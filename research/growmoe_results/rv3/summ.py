import ast, collections, sys
tags = sys.argv[1:] or ["g2A", "g2B"]
for tag in tags:
    rows = [ast.literal_eval(l) for l in open(f"spike_set_gate_{tag}_log.txt", encoding="utf-8") if l.startswith("{")]
    d = collections.OrderedDict()
    for r in rows:
        d.setdefault((r["stream"], r["variant"]), []).append(r)
    print(tag, len(rows), "rows")
    for (s, v), rr in d.items():
        n = len(rr)
        kn = sum(r["known_novel"] + r["known_known"] + r["known_skip"] + r["known_none"] for r in rr)
        print(f"  {s:13s} {v:14s} n={n} acc={100*sum(r['acc'] for r in rr)/n:5.1f} E={sum(r['experts'] for r in rr)/n:5.1f}/{rr[0]['regimes']} novel={sum(r['set_novel'] for r in rr)/n:5.1f} known={sum(r['set_known'] for r in rr)/n:5.1f} abst={sum(r['abstain'] for r in rr)/n:4.1f} known-batches-called-novel={sum(r['known_novel'] for r in rr)}/{kn}")
