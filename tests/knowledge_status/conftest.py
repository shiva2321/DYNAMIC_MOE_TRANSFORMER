import os
import sys
import warnings
from pathlib import Path

# Offline always: the whole project runs from the local Hugging Face cache (spec, constraints).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
KS = ROOT / "research" / "knowledge_status"
for p in (str(ROOT), str(KS)):
    if p not in sys.path:
        sys.path.insert(0, p)

# torch emits an environment-specific FutureWarning (pynvml) on first import of torch.cuda; keep real warnings visible.
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
import torch  # noqa: E402,F401
