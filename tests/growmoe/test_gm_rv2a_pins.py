"""RV2-A: historical entry points keep the per-sample assignment they were run with, so recorded results reproduce."""

import importlib.util
from pathlib import Path

from growmoe import bench
from growmoe.config import PhyloCfg

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_phase_one_entry_point_pins_the_vote_off_but_callers_can_override():
    assert bench._phase1_cfg(seed=0).batch_vote is False
    assert bench._phase1_cfg(seed=0, batch_vote=True).batch_vote is True


def test_m0_settings_pin_the_vote_off_and_so_do_the_m2a_arms_built_from_them():
    m0 = _load(ROOT / "research" / "growmoe_results" / "m0" / "run_m0.py")
    assert m0.A_KW["batch_vote"] is False
    assert PhyloCfg(**m0.A_KW).batch_vote is False
