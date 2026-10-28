"""Generic rewrites of teaching sentences (option (a), wording diversity in teaching; user decision 2026-10-28). The reader's own model (Qwen) writes a few rewrites of each teaching request;
they are used as extra positives for the learned map, as extra prototypes and as extra calibration parts. The protocol that keeps the claim honest:

* the instructions are GENERIC: they ask for "different words", "a different sentence structure" and so on, and name no wording family of the test bed (stories, noise, register, chatter); a guard
  test enforces this, and no instruction names the skill;
* a rewrite is kept only if it contains exactly the same numbers as its source (digits or number words) and says something different; whether the OPERATION survived is checked by reading a sample
  (the numbers cannot tell);
* generation is cached on disk in blocks, written atomically, so a power cut costs one block; the filter is applied when the cache is read, so it can change without generating again.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence

import function_check as FC
import runlib

INSTRUCTIONS = (
    "Rewrite the following request in different words. Keep every number and the exact meaning. Reply with the rewritten request only.",
    "Say the same thing with a different sentence structure. Keep every number and the exact meaning. Reply with the rewritten request only.",
    "Rewrite the following the way a person would naturally put it. Keep every number and the exact meaning. Reply with the rewritten request only.",
    "Rewrite the following using more words, so that it is very clear what is being asked. Keep every number and the exact meaning. Reply with the rewritten request only.",
)
MAX_WORDS = 60
_CHATTER_START = ("sure", "here is", "here's", "certainly", "of course", "okay", "ok,", "rewritten")
_SPACES = re.compile(r"\s+")

Generator = Callable[[Sequence[str], int], List[str]]


def build_prompt(instruction: str, source: str) -> str:
    return f"{instruction}\n\nRequest: {source}"


def _norm(text: str) -> str:
    return _SPACES.sub(" ", text.lower()).strip()


def numbers_in(text: str) -> List[int]:
    """Every number of the text, digits and number words together, sorted (the order in a rewrite may differ)."""
    return sorted([int(d) for d in re.findall(r"\d+", text)] + FC.parse_numbers(re.sub(r"\d+", " ", text)))


def keep(source: str, rewrite: str) -> bool:
    r = rewrite.strip()
    if not r or "\n" in r or len(r.split()) > MAX_WORDS:
        return False
    if _norm(r) == _norm(source) or _norm(r).startswith(_CHATTER_START):
        return False
    return numbers_in(source) == numbers_in(r)


def _block_path(cache_dir: Path, instruction: str, seed: int, tag: str, block: Sequence[str]) -> Path:
    h = hashlib.sha256()
    for part in (instruction, str(seed), tag, *block):
        raw = part.encode("utf-8", "surrogatepass")
        h.update(len(raw).to_bytes(8, "little") + raw)
    return Path(cache_dir) / f"rw_{h.hexdigest()}.json"


def _read_block(path: Path, n: int):
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))["raw"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return raw if isinstance(raw, list) and len(raw) == n and all(isinstance(x, str) for x in raw) else None


def rewrites_for(sources: Sequence[str], instructions: Sequence[str], generate: Generator, cache_dir, seed: int = 0, batch: int = 64, tag: str = "") -> Dict[str, List[str]]:
    """For each distinct source the kept rewrites, in instruction order, without repeats. ``generate(prompts, seed)`` returns one text per prompt. One cache file per (instruction, seed, tag, block
    of ``batch`` sources); a missing or damaged file is generated again."""
    uniq = list(dict.fromkeys(sources))
    out: Dict[str, List[str]] = {s: [] for s in uniq}
    if not uniq:
        return out
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    seen: Dict[str, set] = {s: {_norm(s)} for s in uniq}
    for instruction in instructions:
        for start in range(0, len(uniq), batch):
            block = uniq[start:start + batch]
            path = _block_path(cache_dir, instruction, seed, tag, block)
            raw = _read_block(path, len(block))
            if raw is None:
                raw = [str(x) for x in generate([build_prompt(instruction, s) for s in block], seed)]
                if len(raw) != len(block):
                    raise RuntimeError(f"the generator returned {len(raw)} texts for {len(block)} prompts")
                runlib.atomic_write_text(path, json.dumps({"raw": raw}))
            for s, r in zip(block, raw):
                if keep(s, r) and _norm(r) not in seen[s]:
                    seen[s].add(_norm(r))
                    out[s].append(r.strip())
    return out


SOLVE_INSTRUCTION = "Answer with a single whole number only."


def solve_prompt(text: str) -> str:
    return SOLVE_INSTRUCTION + "\n\n" + text


def first_int(reply: str):
    m = re.search(r"-?\d+", reply)
    return int(m.group()) if m else None


def verify(rewrites: Mapping[str, List[str]], answers: Mapping[str, int], solve: Generator, cache_dir, seed: int = 0, batch: int = 64, tag: str = "") -> Dict[str, List[str]]:
    """Keep a rewrite only if the writer's own model, asked to solve it, gives the TRUE answer of the teaching example it came from. This is a check by function: it catches a rewrite that changed
    the operation, swapped the operands or stopped being a request (the numbers alone cannot). A source whose answer is not known keeps nothing. Cached like the rewrites."""
    texts = list(dict.fromkeys(r for rs in rewrites.values() for r in rs))
    solved: Dict[str, object] = {}
    if texts:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    for start in range(0, len(texts), batch):
        block = texts[start:start + batch]
        path = _block_path(Path(cache_dir), SOLVE_INSTRUCTION, seed, tag, block)
        raw = _read_block(path, len(block))
        if raw is None:
            raw = [str(x) for x in solve([solve_prompt(t) for t in block], seed)]
            if len(raw) != len(block):
                raise RuntimeError(f"the solver returned {len(raw)} texts for {len(block)} prompts")
            runlib.atomic_write_text(path, json.dumps({"raw": raw}))
        solved.update(zip(block, (first_int(x) for x in raw)))
    return {src: [r for r in rs if src in answers and solved.get(r) == answers[src]] for src, rs in rewrites.items()}


def make_qwen_solver(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> Generator:
    """The same model answering with a few greedy tokens."""
    return make_qwen_generator(max_new_tokens=8, temperature=0.0, model_name=model_name)


def make_qwen_generator(max_new_tokens: int = 48, temperature: float = 0.8, top_p: float = 0.9, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> Generator:
    """The reader's own model as the writer of rewrites (sampled, so repeated instructions differ). Left padding is set only for the duration of a call."""
    import torch
    import reps

    def generate(prompts: Sequence[str], seed: int) -> List[str]:
        tok, model = reps._hf_model(model_name, "AutoModelForCausalLM", True, True)
        chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
        side, tok.padding_side = tok.padding_side, "left"
        try:
            enc = tok(chats, return_tensors="pt", padding=True).to(model.device)
            with torch.random.fork_rng(devices=[model.device] if model.device.type == "cuda" else []):
                torch.manual_seed(seed + len(prompts))
                with torch.no_grad():
                    kw = dict(do_sample=True, temperature=temperature, top_p=top_p) if temperature > 0 else dict(do_sample=False)
                    out = model.generate(**enc, **kw, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]
        finally:
            tok.padding_side = side

    return generate
