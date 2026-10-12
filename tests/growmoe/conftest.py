import warnings

# torch emits an environment-specific FutureWarning (pynvml) on first import of torch.cuda.
# Pre-import torch under a filter so the noise does not drown real warnings from our code.
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
import torch  # noqa: E402,F401
