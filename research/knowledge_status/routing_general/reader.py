"""The card reader: shows the language model a message and the domain-free question of cards.py and returns the probabilities of the answer letters (one forward pass, no generation).

The model call itself is the one of the asking reader of the arithmetic lab (``ask_reader._choice_model``); everything else here is about the question and the cache. Answers are cached in
one json file per question (the question's signature names the file), written atomically, so a power cut loses at most the batch in flight.
"""

import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

_HERE = Path(__file__).resolve().parent
_KS = _HERE.parent
for _p in (str(_KS.parents[1]), str(_KS), str(_KS / "routing_lab"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import ask_reader as AR  # noqa: E402
import cards as CD  # noqa: E402

MAX_TEXT_TOKENS = AR.MAX_TEXT_TOKENS


def build_chat(tok, text: str, cards: Sequence[CD.Card], with_exemplars: bool) -> str:
    """The chat-formatted prompt for one message: the message is cut by tokens (never by characters), the model's own template adds the generation prompt."""
    ids = tok(text, add_special_tokens=False, truncation=True, max_length=MAX_TEXT_TOKENS)["input_ids"]
    body = CD.prompt_text(tok.decode(ids), cards, with_exemplars)
    return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)


class CardReader:
    def __init__(self, cards: Sequence[CD.Card], with_exemplars: bool, cache_dir, ask: Optional[Callable[[Sequence[str]], np.ndarray]] = None, gpu: bool = True) -> None:
        self.cards, self.with_exemplars, self.gpu = list(cards), bool(with_exemplars), gpu
        self.path = Path(cache_dir) / f"cards_{CD.signature(self.cards, self.with_exemplars)}.json"      # validates the cards too
        self._ask = ask if ask is not None else self._model_ask

    def _model_ask(self, texts: Sequence[str]) -> np.ndarray:
        builder = lambda tok, t: build_chat(tok, t, self.cards, self.with_exemplars)
        return AR._choice_model(texts, builder, CD.letters_for(self.cards), self.gpu)

    def probabilities(self, texts: Sequence[str]) -> np.ndarray:
        """(n, cards + 1): the probability of each card and, last, of "none of these"."""
        return AR.probabilities(list(texts), self.path, self._ask, width=len(self.cards) + 1)
