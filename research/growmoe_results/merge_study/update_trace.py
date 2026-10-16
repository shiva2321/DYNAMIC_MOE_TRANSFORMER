"""Independent check of the double count (E-037): log every VmfSignature.update call during the observe call that founds expert 1. No product file is touched.

    python update_trace.py            # writes nothing; the output is kept in update_trace_output.txt
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

LOG = []
_orig = VmfSignature.update


def traced(self, x):
    u = self._unit(x)
    held = self.protos
    dup = int(((u @ held.T).max(dim=1).values > 1 - 1e-5).sum()) if self._n else 0
    LOG.append((id(self), int(self.seen), int(u.shape[0]), dup))
    return _orig(self, x)


VmfSignature.update = traced
torch.set_num_threads(2)
st = function_regimes(0, n_regimes=H.REGIMES, d=H.D, n_classes=H.C, n_train=64 * 10, n_test=200)
pool = H.make_pool("clean", 0, None)
for b, (k, seg, idx) in enumerate(st._walk(64, 0)):
    LOG.clear()
    pool.observe(seg.x[idx], seg.y[idx])
    if len(pool.experts) == 2:
        print(f"observe call {b} (the first batch of regime 1) founded expert 1; signature updates during this ONE call (signature id suffix, points already seen, rows added, rows that were already prototypes):")
        for sid, seen, rows, dup in LOG:
            print(f"   signature ...{sid % 100000:05d}: seen {seen:4d}, adds {rows:3d} rows, {dup:3d} of them were already held")
        print("signatures:", {e: f"...{id(r.meta['sig']) % 100000:05d}" for e, r in pool.experts.items()})
        break
