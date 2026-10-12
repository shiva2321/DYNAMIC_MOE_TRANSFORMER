import dataclasses

import growmoe
from growmoe.config import CommonCfg, PhyloCfg


def test_package_version():
    assert growmoe.__version__ == "0.1.0"


def test_phylo_defaults_are_error_rates_not_magic_constants():
    cfg = PhyloCfg()
    assert cfg.q == 0.05               # tolerated false-novelty rate / FDR
    assert cfg.delta_forget == 0.05    # nats of old-data loss we may trade
    assert cfg.d == 128 and cfg.n_classes == 64


def test_configs_are_replaceable_and_independent():
    a = PhyloCfg()
    b = dataclasses.replace(a, q=0.1, seed=7)
    assert (a.q, a.seed) == (0.05, 0)
    assert (b.q, b.seed) == (0.1, 7)
    assert isinstance(a, CommonCfg)
