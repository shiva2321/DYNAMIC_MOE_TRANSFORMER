"""Amendment A-15 (persistence): the false-novelty statement S3-5 counts every known batch the set test DOUBTED, whether it was acted on at once ("novel") or held for
confirmation ("held"); records without `fn_doubtful` (the first build) fall back to `fn_called_novel`. Edits to growmoe/fairfight.py. Each entry: (name, old, new). `old` must occur exactly once."""

FILE = "growmoe/fairfight.py"

EDITS = [
    ("fairfight-s35-count",
     '''            called += int(r.get("fn_called_novel", 0))
''',
     '''            called += int(r.get("fn_doubtful", r.get("fn_called_novel", 0)))              # A-15: doubted = acted on at once or held; the first build's records only have the former
'''),

    ("fairfight-s35-rule",
     '''"S3-5": f"known single-source batches called new as a set, pooled over the nine single-source streams: <= {100 * bars['S3_5']:.0f} %",''',
     '''"S3-5": f"known single-source batches the set test called new (acted on at once or held for confirmation), pooled over the nine single-source streams: <= {100 * bars['S3_5']:.0f} %",'''),

    ("fairfight-record-comment",
     '''# batches and how many of them the set gate called new), and `groups` = {kind: {"single": accuracy, "group": accuracy}} for the kinds''',
     '''# batches and how many of them the set gate acted on at once; optional `fn_doubtful`: how many it doubted at all, acted on or held under persistence (S3-5 reads it
# when present), and `groups` = {kind: {"single": accuracy, "group": accuracy}} for the kinds'''),
]
