"""The cells of the Study 1 ladder (network x recipe), kept in a module with no heavy imports so that run_ladder.py and queue.py agree on what has to exist.

Cell result files are ``ladder/<arch>__<recipe>__s<seed>.json``; the shards of ``run_ladder.py`` take every ``n``-th job of the job list (index % n == shard).
"""

CELLS = [
    ("E128", "D40"), ("E128", "D10"), ("E128", "O5-r256"), ("E128", "O5-r20000"), ("E128", "O1-r20000"),
    ("E512", "D40"), ("E512", "D10"), ("E512", "O5-r256"), ("E512", "O5-r20000"), ("E512", "O1-r20000"), ("E512", "O5-r20000-anneal"),
    ("S256x2", "D40"), ("S256x2n", "D40"), ("S256x2", "D10"), ("S256x2", "O5-r256"), ("S256x2", "O5-r20000"),
    ("L1024x3", "D40"), ("L1024x3", "D10"), ("L1024x3", "O5-r20000"),
]
CELLS_EXTRA = [
    ("E512", "O5-r2000"), ("E512", "O5-r100000"), ("E512", "O5-r300000"), ("E512", "M2000-D40"), ("E512", "M20000-D40"),
    ("E128", "O5-r2000"), ("E128", "O5-r300000"), ("E128", "M20000-D40"),
    ("S256x2", "M20000-D40"), ("S256x2", "O5-r300000"),
]
CELLS_SUBSET = [(a, r) for a in ("E512", "E128", "S256x2") for r in ("M2000-S3k", "M2000-S23k", "M20000-S3k", "M20000-S23k")]


def job_list(seeds, which):
    cells = {"main": CELLS, "extra": CELLS_EXTRA, "subset": CELLS_SUBSET, "all": CELLS + CELLS_EXTRA}[which]
    return [(arch, recipe, int(s)) for s in seeds for arch, recipe in cells]


def shard_files(shard, n, seeds, which):
    """Names (relative to ``ladder/``) of the result files that shard ``shard`` of ``n`` is responsible for."""
    return [f"{a}__{r}__s{s}.json" for j, (a, r, s) in enumerate(job_list(seeds, which)) if j % n == shard]
