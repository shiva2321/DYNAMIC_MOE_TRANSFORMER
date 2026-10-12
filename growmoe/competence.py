"""Loss-change monitor: a one-sided CUSUM on a loss stream with a calibrated false-alarm probability.

The first ``warmup`` losses define the baseline (mean, std). Afterwards each loss is standardised and fed to

    S_t = max(0, S_{t-1} + (x_t - mean)/std - delta)        alarm while S_t > lam

so only increases of the loss raise alarms and nothing depends on the loss scale. ``alpha`` is the
probability of at least one false alarm within ``horizon`` post-warm-up steps of a stationary stream.
``lam`` is the larger of two (1 - alpha) quantiles of max_t S_t, both obtained by re-running the WHOLE
procedure, including estimating mean and std from ``warmup`` samples, on every replicate:

* ``lam_gaussian``  i.i.d. Gaussian losses (data independent, cached). Ignoring the baseline's estimation
  error would give 8.4 instead of 13.4 for (warmup=50, horizon=300, delta=0.5, alpha=0.01).
* ``lam_bootstrap`` i.i.d. (or block) resampling of the warm-up window itself, which raises the threshold
  when the warm-up shows skewed or heavy-tailed losses. On its own it is too noisy (it varies by +-2 around
  the exact value and the false-alarm probability is convex in lam), hence the Gaussian floor.

Measured marginal false-alarm rate (nominal 0.010, warmup 50, horizon 300, 1500 streams): Gaussian 0.011,
exponential 0.021, lognormal(0.5) 0.029, AR(1) r = 0.3 / 0.5: 0.096 / 0.247. Autocorrelation cannot be
estimated from 50 samples, so feed per-sample prequential losses (independent given the model), arm the
monitor only after the expert's loss has plateaued, and use ``block > 1`` if the stream is known to be dependent.

``PhyloMoE`` does this through ``PhyloCfg.arm_monitor_after_infancy`` (ON by default since 2026-10-10): a newborn's monitor is
neither fed nor tested during its infancy. Before that, the monitor was armed from the loss of the UNTRAINED newborn, read the
newborn's own early over-fitting as concept drift and fired once per regime (research/growmoe_results/m2a/RESULTS_M2A.md).
"""

import math
from functools import lru_cache

import numpy as np

_CHUNK = 2000


def _null_peaks(series: np.ndarray, warmup: int, delta: float) -> np.ndarray:
    """Max CUSUM statistic of every row after estimating its baseline from its first ``warmup`` entries."""
    base, rest = series[:, :warmup], series[:, warmup:]
    mu = base.mean(axis=1, keepdims=True)
    sd = np.maximum(base.std(axis=1, ddof=1, keepdims=True), 1e-12)
    c = np.cumsum((rest - mu) / sd - delta, axis=1)
    floor = np.minimum.accumulate(np.minimum(c, 0.0), axis=1)            # min over s <= t of c_s, with c_0 = 0
    return (c - floor).max(axis=1)


@lru_cache(maxsize=64)
def gaussian_null_lambda(alpha: float, warmup: int, horizon: int, delta: float) -> float:
    rng = np.random.default_rng([0x47415553, warmup, horizon])
    n_rep = max(40_000, int(math.ceil(400.0 / alpha)))
    peaks = [_null_peaks(rng.standard_normal((min(_CHUNK, n_rep - lo), warmup + horizon)), warmup, delta)
             for lo in range(0, n_rep, _CHUNK)]
    return float(np.quantile(np.concatenate(peaks), 1.0 - alpha, method="higher"))


class LossChangeMonitor:
    def __init__(self, alpha: float = 0.01, warmup: int = 50, horizon: int = 1000, delta: float = 0.5,
                 n_null: int = None, block: int = 1, seed: int = 0) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if warmup < 10:
            raise ValueError("warmup must be at least 10 to estimate a baseline")
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        if delta < 0.0:
            raise ValueError("delta must be non-negative (in units of the baseline std)")
        if block < 1:
            raise ValueError("block must be at least 1")
        self.alpha, self.warmup, self.horizon, self.delta = alpha, warmup, horizon, delta
        self.block, self.seed = block, seed
        self.n_null = n_null if n_null is not None else max(2000, int(math.ceil(20.0 / alpha)))
        self._calibrations = 0
        self.reset()

    # ------------------------------------------------------------------- state
    def reset(self) -> None:
        self._warm = []
        self._mean = self._std = self._lam = self._lam_boot = None
        self._s = 0.0

    @property
    def armed(self) -> bool:
        return self._lam is not None

    @property
    def lam(self) -> float:
        return self._lam

    @property
    def lam_gaussian(self) -> float:
        return gaussian_null_lambda(self.alpha, self.warmup, self.horizon, self.delta)

    @property
    def lam_bootstrap(self) -> float:
        return self._lam_boot

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        return self._std

    @property
    def statistic(self) -> float:
        return self._s

    # ------------------------------------------------------------------ update
    def update(self, loss: float) -> bool:
        x = float(loss)
        if not math.isfinite(x):
            raise ValueError("loss must be finite")
        if not self.armed:
            self._warm.append(x)
            if len(self._warm) == self.warmup:
                self._arm()
            return False
        self._s = max(0.0, self._s + (x - self._mean) / self._std - self.delta)
        return self._s > self._lam

    def _arm(self) -> None:
        w = np.asarray(self._warm, dtype=np.float64)
        self._mean = float(w.mean())
        self._std = max(float(w.std(ddof=1)), 1e-12 * max(1.0, abs(self._mean)))
        self._lam_boot = self._bootstrap_lambda((w - self._mean) / self._std)
        self._lam = max(self.lam_gaussian, self._lam_boot)
        self._s = 0.0

    def _bootstrap_lambda(self, z: np.ndarray) -> float:
        rng = np.random.default_rng([self.seed, 0x4C4F53, self._calibrations])
        self._calibrations += 1
        n, b, total = len(z), self.block, self.warmup + self.horizon
        n_blocks = -(-total // b)
        peaks = []
        for lo in range(0, self.n_null, _CHUNK):
            k = min(_CHUNK, self.n_null - lo)
            starts = rng.integers(0, n, size=(k, n_blocks))
            series = z[(starts[:, :, None] + np.arange(b)) % n].reshape(k, -1)[:, :total]
            peaks.append(_null_peaks(series, self.warmup, self.delta))
        return float(np.quantile(np.concatenate(peaks), 1.0 - self.alpha, method="higher"))
