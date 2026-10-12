"""Structured event log for every structural change (spawn / extend / grow / merge / prune / ...)."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


def _plain(obj: Any) -> Any:
    """Convert tensors / numpy / non-finite floats into strictly JSON-serialisable Python."""
    if isinstance(obj, torch.Tensor):
        return _plain(obj.item() if obj.numel() == 1 else obj.tolist())
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, np.generic):
        return _plain(obj.item())
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return str(obj)


@dataclass
class Event:
    kind: str
    step: int
    fields: dict = field(default_factory=dict)


class EventLog:
    def __init__(self) -> None:
        self._events: list[Event] = []

    def add(self, *args: Any, **fields: Any) -> Event:
        if len(args) >= 2:
            event_kind = str(args[0])
            event_step = int(args[1])
        elif len(args) == 1:
            event_kind = str(args[0])
            event_step = int(fields.pop("step"))
        else:
            event_kind = str(fields.pop("kind"))
            event_step = int(fields.pop("step"))
        ev = Event(kind=event_kind, step=event_step, fields=_plain(fields))
        self._events.append(ev)
        return ev

    def of_kind(self, kind: str) -> list[Event]:
        return [e for e in self._events if e.kind == kind]

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def to_jsonl(self, path) -> None:
        with open(Path(path), "w", encoding="utf-8") as f:
            for e in self._events:
                f.write(json.dumps({"kind": e.kind, "step": e.step, "fields": e.fields}, allow_nan=False) + "\n")

    @classmethod
    def from_jsonl(cls, path) -> "EventLog":
        log = cls()
        with open(Path(path), "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    log._events.append(Event(kind=d["kind"], step=d["step"], fields=d["fields"]))
        return log
