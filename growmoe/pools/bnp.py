"""Task 15: Architecture B (BNP-MoE: CN-DPM-style likelihood gate, scratch experts).

Every expert is a classifier (MLPExpert) plus an unsupervised density estimator (VAEDensity).
Responsibilities are computed via the Chinese Restaurant Process (CRP) prior and VAE ELBO:
    gamma_k(u) = log pi_k + log p_k(u), where pi_k ~ n_k.

A sample is novel if:
    max_k (log pi_k + log p_k(u)) < log alpha_dp + mu_ll - tau_b * sigma_ll
where mu_ll and sigma_ll are running (Welford) mean and standard deviation of the best-component
log-likelihood of accepted samples. Rejected samples accumulate in a short-term memory (STM);
when |STM| >= n_stm, exactly one scratch expert is spawned and trained on the STM.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..config import BNPCfg
from ..experts import MLPExpert
from ..types import Prediction, StepRecord
from .base import BasePool


class VAEDensity(nn.Module):
    """Density estimator using a variational autoencoder with diagonal Gaussian decoder.

    - Encoder: Linear(d, h) -> tanh -> (mu_z, lv_z) via Linear(h, z)
    - Decoder: Linear(z, h) -> tanh -> (x_mu, x_lv) via Linear(h, d), where x_lv in [-8, 4]
    - Single-sample reparameterisation during training; posterior-mean z during evaluation / scoring.
    """

    def __init__(self, d: int = 128, h: int = 64, z: int = 16, seed: int = 0) -> None:
        super().__init__()
        self.d = d
        self.h = h
        self.z = z
        self.seed = seed

        self.gen = torch.Generator().manual_seed(seed)
        self.enc = nn.Linear(d, h)
        self.mu = nn.Linear(h, z)
        self.lv = nn.Linear(h, z)
        self.dec = nn.Linear(z, h)
        self.xmu = nn.Linear(h, d)
        self.xlv = nn.Linear(h, d)

        # Initialize weights with seeded generator
        with torch.no_grad():
            self.enc.weight.copy_(torch.randn(h, d, generator=self.gen) / math.sqrt(d))
            self.enc.bias.zero_()
            self.mu.weight.copy_(torch.randn(z, h, generator=self.gen) / math.sqrt(h))
            self.mu.bias.zero_()
            self.lv.weight.copy_(torch.randn(z, h, generator=self.gen) / math.sqrt(h))
            self.lv.bias.zero_()

            self.dec.weight.copy_(torch.randn(h, z, generator=self.gen) / math.sqrt(z))
            self.dec.bias.zero_()
            self.xmu.weight.copy_(torch.randn(d, h, generator=self.gen) / math.sqrt(h))
            self.xmu.bias.zero_()
            self.xlv.weight.copy_(torch.randn(d, h, generator=self.gen) / math.sqrt(h))
            self.xlv.bias.zero_()

    def forward(self, u: Tensor, sample: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        e = torch.tanh(self.enc(u))
        mu_z = self.mu(e)
        lv_z = self.lv(e)
        if sample:
            eps = torch.randn(mu_z.shape, generator=self.gen, device=mu_z.device)
            z = mu_z + eps * torch.exp(0.5 * lv_z)
        else:
            z = mu_z
        h_dec = torch.tanh(self.dec(z))
        x_mu = self.xmu(h_dec)
        x_lv = self.xlv(h_dec).clamp(-8.0, 4.0)
        return mu_z, lv_z, x_mu, x_lv

    def elbo(self, u: Tensor, sample: Optional[bool] = None) -> Tensor:
        """Compute the evidence lower bound (ELBO) per sample in nats. Returns shape (B,)."""
        u = u.float()
        if sample is None:
            sample = self.training
        mu_z, lv_z, x_mu, x_lv = self.forward(u, sample=sample)

        # Log p(u|z) = -0.5 * sum_j [ (u_j - x_mu_j)^2 / exp(x_lv_j) + x_lv_j + log(2*pi) ]
        log_var = x_lv
        inv_var = torch.exp(-log_var)
        sq_err = (u - x_mu) ** 2
        recon_ll = -0.5 * torch.sum(sq_err * inv_var + log_var + math.log(2.0 * math.pi), dim=-1)

        # KL(q(z|u) || N(0, I)) = 0.5 * sum_i [ exp(lv_z_i) + mu_z_i^2 - 1 - lv_z_i ]
        kl = 0.5 * torch.sum(torch.exp(lv_z) + mu_z ** 2 - 1.0 - lv_z, dim=-1)

        return recon_ll - kl

    def fit_step(self, u: Tensor, opt: torch.optim.Optimizer) -> float:
        """One optimizer step maximizing ELBO on batch u. Returns training loss."""
        self.train()
        loss = -self.elbo(u, sample=True).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return float(loss.item())


class BNPMoE(BasePool):
    """Architecture B: Bayesian Non-Parametric Mixture of Experts.

    Features:
    - Every expert comprises an MLPExpert classifier and a VAEDensity module.
    - Likelihood gate based on CRP mixture weights and VAE log-density (ELBO).
    - Rejection threshold driven by running Welford statistics of accepted log-likelihoods.
    - Rejected samples accumulate in short-term memory (STM).
    - Thresholded STM spawning of scratch experts.
    - No vertical growth, merge, or archival (as published in CN-DPM).
    """

    name = "B_bnpmoe"

    def __init__(self, cfg: Optional[BNPCfg] = None) -> None:
        cfg = cfg if cfg is not None else BNPCfg()
        super().__init__(cfg)
        self.cfg: BNPCfg = cfg

        # Short-term memory (STM)
        self.stm_x: List[Tensor] = []
        self.stm_y: List[Tensor] = []

        # Running Welford statistics of best-component log-likelihood for accepted samples
        self._mu_ll: float = 0.0
        self._m2_ll: float = 0.0
        self._n_accepted: int = 0
        self._sigma_ll: float = 1.0

    def _normalize(self, x: Tensor) -> Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return x / norm

    def _update_welford(self, ll_values: np.ndarray) -> None:
        for val in ll_values:
            self._n_accepted += 1
            delta = val - self._mu_ll
            self._mu_ll += delta / self._n_accepted
            delta2 = val - self._mu_ll
            self._m2_ll += delta * delta2
        if self._n_accepted > 1:
            self._sigma_ll = max(math.sqrt(self._m2_ll / (self._n_accepted - 1)), 1e-4)
        else:
            self._sigma_ll = 1.0

    def _spawn_from_stm(self, x_stm: Tensor, y_stm: Tensor) -> int:
        next_id = self._next_id
        n_pts = len(x_stm)

        # Fresh classifier
        child_mod = MLPExpert(
            d_in=self.cfg.d,
            hidden=self.cfg.hidden,
            depth=self.cfg.depth,
            n_classes=self.cfg.n_classes,
            seed=self.cfg.seed * 1000 + next_id,
        )

        # Fresh VAE density
        child_vae = VAEDensity(
            d=self.cfg.d,
            h=self.cfg.vae_h,
            z=self.cfg.vae_z,
            seed=self.cfg.seed * 1000 + next_id,
        )
        opt_vae = torch.optim.Adam(child_vae.parameters(), lr=self.cfg.vae_lr)

        # Fit VAE on STM samples
        rng_stm = np.random.default_rng([self.cfg.seed, 0x424E50, next_id])
        for _ in range(self.cfg.vae_steps):
            idx = torch.from_numpy(rng_stm.integers(0, n_pts, min(self.cfg.batch_size, n_pts)))
            child_vae.fit_step(x_stm[idx], opt_vae)

        # Fit classifier on STM if labeled
        valid_mask = y_stm >= 0
        if valid_mask.any():
            valid_x = x_stm[valid_mask]
            valid_y = y_stm[valid_mask]
            child_mod.mark_seen(valid_y)
            opt_mod = self.make_optimizer(child_mod, 1.0)
            n_val = len(valid_x)
            for _ in range(self.cfg.vae_steps):
                idx = torch.from_numpy(rng_stm.integers(0, n_val, min(self.cfg.batch_size, n_val)))
                child_mod.train_step(valid_x[idx], valid_y[idx], opt_mod)

        # Register expert with BasePool
        rec = self.add_expert(
            child_mod,
            parent=None,
            state="active",
            meta={
                "vae": child_vae,
                "opt_vae": opt_vae,
                "n_assigned": n_pts,
            },
        )

        if valid_mask.any():
            rec.replay.add(x_stm[valid_mask], y_stm[valid_mask])
        else:
            rec.replay.add(x_stm, y_stm)

        # Log spawn event
        self.events.add(
            "spawn",
            self.step,
            expert=rec.id,
            parent=None,
            affinity=0.0,
            evidence=0.0,
            n_cluster=n_pts,
            init="scratch",
            lam=0.0,
        )

        # Seed initial Welford stats if first expert
        if self._n_accepted == 0:
            with torch.no_grad():
                elbos = child_vae.elbo(x_stm, sample=False).numpy()
            self._mu_ll = float(elbos.mean())
            self._sigma_ll = max(float(elbos.std()), 1e-4)
            self._m2_ll = float(elbos.var() * max(1, len(elbos) - 1))
            self._n_accepted = len(elbos)

        return rec.id

    def observe(self, x: Tensor, y: Optional[Tensor] = None, ctx: Optional[Tensor] = None) -> StepRecord:
        """Observe a batch (x, y); evaluate CRP-VAE gate, update STM/Welford, and train assigned experts."""
        x = self._normalize(x.float())
        b = x.shape[0]
        y_t = torch.as_tensor(y, dtype=torch.long) if y is not None else None

        spawned: List[int] = []

        # Empty pool case: buffer into STM until n_stm is reached
        if len(self.experts) == 0:
            for i in range(b):
                self.stm_x.append(x[i])
                if y_t is not None:
                    self.stm_y.append(y_t[i])
                else:
                    self.stm_y.append(torch.tensor(-1, dtype=torch.long))

            if len(self.stm_x) >= self.cfg.n_stm:
                all_stm_x = torch.stack(self.stm_x, dim=0)
                all_stm_y = torch.stack(self.stm_y, dim=0)
                self.stm_x.clear()
                self.stm_y.clear()
                new_eid = self._spawn_from_stm(all_stm_x, all_stm_y)
                spawned.append(new_eid)
                assigned = torch.full((b,), new_eid, dtype=torch.long)
                novel = torch.ones(b, dtype=torch.bool)
                p_nov = torch.zeros(b, dtype=torch.float32)
            else:
                assigned = torch.full((b,), -1, dtype=torch.long)
                novel = torch.ones(b, dtype=torch.bool)
                p_nov = torch.zeros(b, dtype=torch.float32)

            self.tick()
            return StepRecord(
                novel=novel,
                p_nov=p_nov,
                assigned=assigned,
                loss=float("nan"),
                spawned=spawned,
                n_experts=len(self.experts),
            )

        # Scoring with existing experts: log pi_k + log p_k(u)
        eids = sorted(self.experts.keys())
        resps = []
        with torch.no_grad():
            for eid in eids:
                rec = self.experts[eid]
                vae_k: VAEDensity = rec.meta["vae"]
                elbo_k = vae_k.elbo(x, sample=False)  # (B,)
                n_k = max(1, rec.meta.get("n_assigned", 1))
                log_pi_k = math.log(n_k)
                resps.append(log_pi_k + elbo_k)

        R = torch.stack(resps, dim=1)  # (B, K)
        best_resps, best_idx = R.max(dim=1)
        best_eids = [eids[idx.item()] for idx in best_idx]

        threshold = math.log(self.cfg.alpha_dp) + self._mu_ll - self.cfg.tau_b * self._sigma_ll
        is_novel = (best_resps < threshold).cpu()
        margin = (best_resps - threshold).cpu()
        p_nov = torch.sigmoid(margin / max(self._sigma_ll, 1e-4))

        assigned = torch.full((b,), -1, dtype=torch.long)
        for i in range(b):
            if not is_novel[i]:
                assigned[i] = best_eids[i]

        # Handle novel samples -> STM
        if is_novel.any():
            nov_idx = torch.nonzero(is_novel).flatten()
            for idx in nov_idx:
                self.stm_x.append(x[idx])
                if y_t is not None:
                    self.stm_y.append(y_t[idx])
                else:
                    self.stm_y.append(torch.tensor(-1, dtype=torch.long))

            if len(self.stm_x) >= self.cfg.n_stm and len(self.experts) < self.cfg.max_experts:
                all_stm_x = torch.stack(self.stm_x, dim=0)
                all_stm_y = torch.stack(self.stm_y, dim=0)
                self.stm_x.clear()
                self.stm_y.clear()
                new_eid = self._spawn_from_stm(all_stm_x, all_stm_y)
                spawned.append(new_eid)
                assigned[nov_idx] = new_eid

        # Handle accepted samples -> hard responsibility training
        losses = []
        if (~is_novel).any():
            acc_idx = torch.nonzero(~is_novel).flatten()
            acc_elbos = best_resps[acc_idx].cpu().numpy().astype(np.float64)
            self._update_welford(acc_elbos)

            for eid in eids:
                mask_k = (assigned == eid)
                if not mask_k.any():
                    continue
                xk = x[mask_k]
                rec = self.experts[eid]
                rec.last_active = self.step
                rec.meta["n_assigned"] = rec.meta.get("n_assigned", 0) + len(xk)

                # Fit VAE on assigned points
                rec.meta["vae"].fit_step(xk, rec.meta["opt_vae"])

                # Fit classifier on assigned points if labeled
                if y_t is not None:
                    yk = y_t[mask_k]
                    valid = yk >= 0
                    if valid.any():
                        l_k = self.train_expert(eid, xk[valid], yk[valid], replay=True)
                        if not math.isnan(l_k):
                            losses.append(l_k)

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        self.tick()
        return StepRecord(
            novel=is_novel,
            p_nov=p_nov,
            assigned=assigned,
            loss=mean_loss,
            spawned=spawned,
            n_experts=len(self.experts),
        )

    def predict(self, x: Tensor, ctx: Optional[Tensor] = None) -> Prediction:
        """Predict class probabilities and novelty without learning or state mutation."""
        x = self._normalize(x.float())
        b = x.shape[0]

        if len(self.experts) == 0:
            probs = torch.full(
                (b, self.cfg.n_classes),
                1.0 / self.cfg.n_classes,
                dtype=torch.float32,
            )
            expert = torch.full((b,), -1, dtype=torch.long)
            p_nov = torch.zeros(b, dtype=torch.float32)
            unknown = torch.ones(b, dtype=torch.bool)
            return Prediction(probs=probs, expert=expert, p_nov=p_nov, unknown=unknown)

        eids = sorted(self.experts.keys())
        resps = []
        with torch.no_grad():
            for eid in eids:
                rec = self.experts[eid]
                vae_k: VAEDensity = rec.meta["vae"]
                elbo_k = vae_k.elbo(x, sample=False)
                n_k = max(1, rec.meta.get("n_assigned", 1))
                log_pi_k = math.log(n_k)
                resps.append(log_pi_k + elbo_k)

        R = torch.stack(resps, dim=1)  # (B, K)
        best_resps, best_idx = R.max(dim=1)
        routed_expert = torch.tensor([eids[idx.item()] for idx in best_idx], dtype=torch.long)

        threshold = math.log(self.cfg.alpha_dp) + self._mu_ll - self.cfg.tau_b * self._sigma_ll
        margin = (best_resps - threshold).cpu()
        p_nov = torch.sigmoid(margin / max(self._sigma_ll, 1e-4))
        unknown = margin < 0

        probs = torch.zeros((b, self.cfg.n_classes), dtype=torch.float32)
        with torch.no_grad():
            for eid in eids:
                mask_k = (routed_expert == eid)
                if mask_k.any():
                    probs[mask_k] = self.experts[eid].module.proba(x[mask_k]).detach()

        return Prediction(probs=probs, expert=routed_expert, p_nov=p_nov, unknown=unknown)
