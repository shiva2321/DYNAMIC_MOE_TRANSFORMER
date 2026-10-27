"""Cutting a message into parts (spec 5.4). A part is meaning-sized (a request with its values, a sentence), never a token. Every cutter returns character spans ``(start, end)``
into the message text, in order, with no leading or trailing space inside a span.

* ``gold_parts``: the spans the generator wrote down (the exact answer key; evaluating with these isolates the reader and the status machine from segmentation).
* ``sentence_clause_parts``: break after a sentence end (. ? !) or a semicolon, and after a comma that is followed by a clause word (and, but, then, so, or, because); a piece of fewer
  than three words is joined to its neighbour, so "Hi." never stands alone. A bare "and" is never a break ("what is 17 and 5 together" is one request).
* ``window_parts``: overlapping windows of ``width`` words every ``stride`` words, the last one anchored at the end so that no word is left out.
"""

import re
from typing import List, Tuple

from generator import Message

Span = Tuple[int, int]
MIN_WORDS = 3
_BOUNDARY = re.compile(r"(?<=[.!?;])\s+|(?<=,)\s+(?=(?:and|but|then|so|or|because)\b)", re.IGNORECASE)
_WORD = re.compile(r"\S+")


def gold_parts(msg: Message) -> List[Span]:
    return [(p.start, p.end) for p in msg.parts]


def sentence_clause_parts(text: str) -> List[Span]:
    pieces: List[Span] = []
    start = 0
    for m in list(_BOUNDARY.finditer(text)) + [None]:
        end = len(text) if m is None else m.start()
        raw = text[start:end]
        if raw.strip():
            lead = len(raw) - len(raw.lstrip())
            pieces.append((start + lead, start + len(raw.rstrip())))
        if m is not None:
            start = m.end()
    out: List[Span] = []
    held = None
    for s, e in pieces:
        if held is not None:
            s, held = held, None
        if len(text[s:e].split()) < MIN_WORDS:
            held = s
            held_end = e
            continue
        out.append((s, e))
    if held is not None:
        if out:
            out[-1] = (out[-1][0], held_end)
        else:
            out.append((held, held_end))
    return out


def window_parts(text: str, width: int = 8, stride: int = 4) -> List[Span]:
    if width < 1 or stride < 1 or stride > width:
        raise ValueError(f"need 1 <= stride <= width, got width={width} stride={stride}")
    words = [m.span() for m in _WORD.finditer(text)]
    n = len(words)
    if n == 0:
        return []
    if n <= width:
        return [(words[0][0], words[-1][1])]
    starts = list(range(0, n - width + 1, stride))
    if starts[-1] != n - width:
        starts.append(n - width)
    return [(words[i][0], words[i + width - 1][1]) for i in starts]
