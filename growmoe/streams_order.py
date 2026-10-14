"""A guard stream for routing v2 stage 3: the same data, but every batch is ordered by label.

The batch-level gate assumes that the samples of a batch are exchangeable draws from one source. A stream that hands over its samples sorted by label breaks
that assumption (a batch then holds one or two classes), which is what a badly shuffled data loader or a time-ordered log looks like. ``label_sorted(stream)``
keeps the segments, the samples and the batch boundaries of ``stream`` and only changes the order inside each segment: ascending label, ties in the original
order. It is a reported guard of the stage-3 run (the circuit breaker must hold the damage), not a pre-registered statement.
"""

from dataclasses import dataclass, fields
from typing import Iterator, Tuple

import torch
from torch import Tensor

from .streams import Segment, Stream


@dataclass
class LabelSortedStream(Stream):
    def _walk(self, batch_size: int, seed: int) -> Iterator[Tuple[int, Segment, Tensor]]:
        for k, seg in enumerate(self.segments):
            order = torch.argsort(seg.y, stable=True)
            for lo in range(0, len(order), batch_size):
                yield k, seg, order[lo:lo + batch_size]


def label_sorted(stream: Stream) -> LabelSortedStream:
    """The same stream with every segment walked in ascending label order (``seed`` no longer matters for the order)."""
    kw = {f.name: getattr(stream, f.name) for f in fields(Stream)}
    kw["name"] = f"{stream.name}-SORTED"
    kw["meta"] = {**stream.meta, "ordered": "label"}
    return LabelSortedStream(**kw)
