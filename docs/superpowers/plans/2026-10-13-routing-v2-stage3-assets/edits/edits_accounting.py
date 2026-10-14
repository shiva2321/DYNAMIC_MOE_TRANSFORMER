"""Edit to growmoe/accounting.py (stage 3): source windows are stored vectors too. `old` must occur exactly once."""

FILE = "growmoe/accounting.py"

EDITS = [
    ("accounting-windows",
     '''            if r.replay is not None:
                stored += r.replay.n
''',
     '''            if r.replay is not None:
                stored += r.replay.n
            windows = r.meta.get("windows")
            if windows is not None:                                                     # routing v2 stage 3a source windows are kept vectors too
                stored += windows.n_samples()
'''),
]
