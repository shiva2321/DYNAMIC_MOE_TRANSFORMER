"""RV2-B: default settings flip to the two-stage cascade (fingerprint shortlist), and historical entry points keep the exact scan."""

import importlib.util
from pathlib import Path
import types

import numpy as np
import pytest
import torch

from growmoe import bench
from growmoe.config import PhyloCfg
from growmoe.experts import MLPExpert
from growmoe.pools.phylo import PhyloMoE
from growmoe.signature import VmfSignature

ROOT = Path(__file__).resolve().parents[2]
D = 32


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _unit(n, seed, d=D):
    z = np.random.default_rng(seed).standard_normal((n, d))
    return torch.from_numpy((z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32))


def _signature(seed, n=32, centre=None):
    x = _unit(n, seed)
    if centre is not None:
        x = torch.nn.functional.normalize(x * 0.6 + centre, dim=1)
    sig = VmfSignature(D, 256, seed=seed)
    sig.update(x)
    sig.fit()
    return sig


def _fake_pool(n_experts, infants=True, **cfg):
    pool = PhyloMoE(PhyloCfg(d=D, hidden=16, n_classes=4, seed=0, q=0.02, max_experts=4096, route_rule="accept_then_logf", **cfg))
    centres = _unit(n_experts, seed=99)
    for e in range(n_experts):
        state = "infant" if infants and e % 7 == 0 else "active"
        pool.add_expert(MLPExpert(D, 16, 1, 4, seed=e), parent=None, state=state, meta={"sig": _signature(e, centre=centres[e])})
    return pool, centres


def _near(centres, n, seed):
    idx = torch.from_numpy(np.random.default_rng(seed).integers(0, len(centres), n))
    return torch.nn.functional.normalize(centres[idx] * 0.6 + _unit(n, seed + 1), dim=1)


def test_default_phylo_cfg_has_cascade_on():
    cfg = PhyloCfg()
    assert cfg.gate_shortlist == 2
    assert cfg.gate_fingerprint_k == 1
    assert cfg.gate_shortlist_min_experts == 64
    assert cfg.gate_fingerprint_refresh == 128


def test_phase_one_entry_point_pins_the_cascade_off_but_callers_can_override():
    assert bench._phase1_cfg(seed=0).gate_shortlist == 0
    assert bench._phase1_cfg(seed=0, gate_shortlist=2).gate_shortlist == 2


def test_m0_settings_pin_the_cascade_off_and_so_do_the_m2a_and_rv2a_arms():
    m0 = _load(ROOT / "research" / "growmoe_results" / "m0" / "run_m0.py")
    assert m0.A_KW["gate_shortlist"] == 0
    assert PhyloCfg(**m0.A_KW).gate_shortlist == 0


def test_small_pool_with_default_config_is_bit_identical_to_exact_scan():
    # 20 experts < 64: the cascade is inactive, exact scan is used
    pool_def, centres = _fake_pool(20)
    pool_exact, _ = _fake_pool(20, gate_shortlist=0)
    x = _near(centres, 64, seed=5)
    eids = list(pool_def.experts)

    P_def, F_def = pool_def._gate_matrices(x, eids)
    P_exact, F_exact = pool_exact._gate_matrices(x, eids)
    assert torch.equal(P_def, P_exact)
    assert torch.equal(F_def, F_exact)

    out_def = pool_def.predict(x)
    out_exact = pool_exact.predict(x)
    assert torch.equal(out_def.unknown, out_exact.unknown)


def test_pool_above_threshold_uses_the_cascade():
    # 70 experts >= 64: the cascade is active under default config
    pool, centres = _fake_pool(70)
    x = _near(centres, 64, seed=7)
    pool.predict(x)
    assert pool.gate_stats["predict"]["stage1"] > 0


def test_recorded_rv2b_arms_still_pin_themselves():
    rv2b = _load(ROOT / "research" / "growmoe_results" / "rv2b" / "run_rv2b.py")
    dummy_st = types.SimpleNamespace(d=D, n_classes=4)
    pool_exact = rv2b.factory("A_exact", dummy_st, seed=0, steps=10, k=1)()
    assert pool_exact.cfg.gate_shortlist == 0

    pool_cascade = rv2b.factory("A_cascade", dummy_st, seed=0, steps=10, k=1)()
    assert pool_cascade.cfg.gate_shortlist == 2
    assert pool_cascade.cfg.gate_fingerprint_k == 1
    assert pool_cascade.cfg.gate_shortlist_min_experts == 8
