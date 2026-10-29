import pytest

import run_fs_dev as RD


def test_every_arm_gets_e1_on_five_seeds_and_e2_on_thirty_dev_seeds(tmp_path):
    calls = []

    def e1(arm, seed, pool, out, stamp):
        calls.append(("e1", arm, seed, pool, str(out), stamp))
        return {"b1a": 0.9, "b1b": 0.8, "b1c": 0.7}

    def e2(arm, seed, pool, out, stamp):
        calls.append(("e2", arm, seed, pool, str(out), stamp))
        return {"machine": {"q0.05": {"standard": {"macro_f1": {"macro_f1": 0.8}, "false_unseen_heldout": 0.3}}}}

    done = RD.run(out=tmp_path, stamp="S", e1=e1, e2=e2, log=lambda s: None)
    assert len(done) == 2 * (5 + 30) and len(calls) == len(done)
    assert {c[3] for c in calls} == {"dev"} and {c[5] for c in calls} == {"S"}
    assert sorted({c[2] for c in calls if c[0] == "e1"}) == list(range(100, 105)) and sorted({c[2] for c in calls if c[0] == "e2"}) == list(range(100, 130))
    assert {c[1] for c in calls} == {"R5_FS_strict", "R5_FS_pre"} and all(c[4].endswith(c[0]) for c in calls)


def test_it_refuses_seeds_outside_the_dev_range(tmp_path):
    for bad in (dict(e1_seeds=[99]), dict(e2_seeds=[130]), dict(e1_seeds=[400])):
        with pytest.raises(ValueError, match="100 to 129"):
            RD.run(out=tmp_path, stamp="S", e1=lambda *a: {}, e2=lambda *a: {}, **bad)
