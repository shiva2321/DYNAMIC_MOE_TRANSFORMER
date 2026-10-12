import json

import numpy as np
import torch

from growmoe.telemetry import Event, EventLog
from growmoe.types import Prediction, StepRecord


def test_add_and_filter_by_kind():
    log = EventLog()
    log.add("spawn", step=3, expert=1, parent=0)
    log.add("grow", step=5, expert=1, op="widen")
    log.add("spawn", step=9, expert=2, parent=1)
    assert len(log) == 3
    assert [e.fields["expert"] for e in log.of_kind("spawn")] == [1, 2]
    assert log.of_kind("nope") == []


def test_step_is_stored_as_given_and_log_is_iterable():
    log = EventLog()
    log.add("alarm", step=42, expert=0)
    events = list(log)
    assert isinstance(events[0], Event)
    assert events[0].step == 42 and events[0].kind == "alarm"


def test_fields_are_converted_to_plain_json_python():
    log = EventLog()
    log.add("spawn", step=1, p=torch.tensor(0.25), ids=np.array([1, 2, 3]),
            n=np.int64(4), scores=torch.tensor([0.5, 0.75]))
    f = log.of_kind("spawn")[0].fields
    assert f["p"] == 0.25 and type(f["p"]) is float
    assert f["ids"] == [1, 2, 3]
    assert f["n"] == 4 and type(f["n"]) is int
    assert f["scores"] == [0.5, 0.75]
    json.dumps(f, allow_nan=False)  # must be strictly serialisable


def test_non_finite_floats_become_null_so_json_stays_valid():
    log = EventLog()
    log.add("alarm", step=1, loss=float("nan"), gain=float("inf"))
    f = log.of_kind("alarm")[0].fields
    assert f["loss"] is None and f["gain"] is None
    json.dumps(f, allow_nan=False)


def test_jsonl_roundtrip(tmp_path):
    log = EventLog()
    log.add("spawn", step=1, expert=1, affinity=0.4)
    log.add("merge", step=7, kept=1, removed=2)
    path = tmp_path / "events.jsonl"
    log.to_jsonl(path)
    back = EventLog.from_jsonl(path)
    assert [(e.kind, e.step, e.fields) for e in back] == [(e.kind, e.step, e.fields) for e in log]


def test_step_record_and_prediction_hold_tensors():
    rec = StepRecord(novel=torch.zeros(4, dtype=torch.bool), p_nov=torch.ones(4),
                     assigned=torch.zeros(4, dtype=torch.long), loss=0.5, spawned=[], n_experts=1)
    assert rec.novel.shape == (4,) and rec.n_experts == 1
    pred = Prediction(probs=torch.full((4, 3), 1 / 3), expert=torch.zeros(4, dtype=torch.long),
                      p_nov=torch.ones(4), unknown=torch.zeros(4, dtype=torch.bool))
    assert pred.probs.shape == (4, 3)
