"""Cards and the domain-free question.

A card is what the router knows about one expert: the name its teacher gave it (any string) and a few of its own teaching messages. The question shown to the language model is built from the
cards alone: the template below contains no word about any domain, so the same code asks "which operation?", "which customer request?" or "which news topic?" depending only on the roster.
"""

import hashlib
from dataclasses import dataclass
from typing import List, Sequence, Tuple

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_CARDS = 12                                       # one letter per card and one for "none of these"; more cards need candidates picked first (see reader.py)
EXEMPLAR_CHARS = 100
NONE_TEXT = "none of these (the message fits none of the categories)"
TEMPLATE = ("Read the message below and choose the category it belongs to.\n\n"
            "Message: {text}\n\n"
            "Categories:\n{options}\n\n"
            "Answer with one letter.")


@dataclass(frozen=True)
class Card:
    name: str
    exemplars: Tuple[str, ...] = ()


def _check(cards: Sequence[Card]) -> None:
    if not 1 <= len(cards) <= MAX_CARDS:
        raise ValueError(f"a question needs between 1 and {MAX_CARDS} cards, got {len(cards)}")
    names = [c.name.strip() for c in cards]
    if any(not n for n in names):
        raise ValueError("a card needs a name")
    if len(set(names)) != len(names):
        raise ValueError(f"card names must be distinct, got {names}")


def _one_line(text: str) -> str:
    s = " ".join(text.split()).replace('"', "'")
    return s if len(s) <= EXEMPLAR_CHARS else s[:EXEMPLAR_CHARS].rstrip() + "..."


def option_lines(cards: Sequence[Card], with_exemplars: bool) -> List[str]:
    """One line per card, in order, then the line for "none of these"; each line starts with its letter."""
    _check(cards)
    lines = []
    for i, c in enumerate(cards):
        line = f"{LETTERS[i]}. {c.name.strip()}"
        if with_exemplars and c.exemplars:
            line += " (examples: " + "; ".join(f'"{_one_line(e)}"' for e in c.exemplars) + ")"
        lines.append(line)
    lines.append(f"{LETTERS[len(cards)]}. {NONE_TEXT}")
    return lines


def letters_for(cards: Sequence[Card]) -> List[str]:
    _check(cards)
    return list(LETTERS[:len(cards) + 1])


def prompt_text(text: str, cards: Sequence[Card], with_exemplars: bool) -> str:
    return TEMPLATE.format(text=text, options="\n".join(option_lines(cards, with_exemplars)))


def signature(cards: Sequence[Card], with_exemplars: bool) -> str:
    """A short hash of the question without its message (template, options in order, exemplars only when they are shown): the name under which answers to this question are cached."""
    body = TEMPLATE + "\n" + "\n".join(option_lines(cards, with_exemplars))
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]
