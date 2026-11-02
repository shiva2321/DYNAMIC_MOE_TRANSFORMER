"""A restricted tool for running small Python functions and reading their BEHAVIOUR (design: docs/superpowers/specs/2026-10-31-fsmoe-e2e-design.md).

A text is run only if it parses as Python with a top-level function, passes a static allowlist (imports of ``math`` and ``sys`` only, ``sys`` only as ``sys.stderr``, no dunder access, no class, and
every name it reads is bound by the text itself or is a safe builtin), and it is executed with a restricted set of builtins (no file, process, network or introspection names, ``__import__`` limited to math
and sys) under a step limit. Anything else is NEVER executed. The behaviour of a function is (the kind of input it accepts, its outputs on fixed probe inputs): the first of strings, integers, lists
whose five probes all run without error. Two styles of the same function give the same behaviour; a different function gives a different one.
"""

import ast
import contextlib
import io
import sys
from typing import Optional, Tuple

from .domains import code as CO

ALLOWED_IMPORTS = {"math", "sys"}
BANNED_CALLS = {"exec", "eval", "open", "compile", "getattr", "setattr", "delattr", "globals", "locals", "vars", "input", "breakpoint", "__import__", "memoryview", "type"}    # names that are REFUSED, never called here
SAFE_NAMES = ("range", "len", "sum", "max", "min", "abs", "str", "int", "float", "bool", "all", "any", "sorted", "reversed", "list", "set", "tuple", "dict", "enumerate", "zip", "map", "filter",
              "round", "pow", "print", "divmod", "ord", "chr", "object", "Exception", "ValueError", "TypeError", "IndexError", "KeyError", "ZeroDivisionError", "StopIteration")   # `object` only for type hints (evaluated at def time); its dunder attributes are rejected statically
SAFE_BUILTINS = {n: getattr(__builtins__, n) if not isinstance(__builtins__, dict) else __builtins__[n] for n in SAFE_NAMES}
MAX_CHARS = 4000
STEP_LIMIT = 200_000
KINDS = ("str", "int", "list")


class StepLimit(Exception):
    pass


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in ALLOWED_IMPORTS or level:
        raise ImportError(f"import of {name!r} is not allowed")
    return __import__(name, globals, locals, fromlist, level)


def _bound_names(tree: ast.AST) -> set:
    """Every name the text itself binds: functions, arguments, assignment and loop targets, comprehension variables, imports, exception names."""
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.Import):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return bound


def is_safe(tree: ast.AST) -> bool:
    """A true allowlist: imports of math and sys only (sys only as sys.stderr), no dunder access, no class, async or from-import, and every name that is read must be bound by the text itself, one of the
    safe builtins, True, False, None or __name__. A reference to any other global (os, self.__dict__, a module-level helper, ...) refuses the text."""
    bound = _bound_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] not in ALLOWED_IMPORTS for a in node.names):
                return False
        elif isinstance(node, (ast.ImportFrom, ast.ClassDef, ast.AsyncFunctionDef, ast.Await)):
            return False
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return False
            if isinstance(node.value, ast.Name) and node.value.id == "sys" and node.attr != "stderr":
                return False
        elif isinstance(node, ast.Name):
            if node.id in BANNED_CALLS:
                return False
            if isinstance(node.ctx, ast.Load) and node.id not in bound and node.id not in SAFE_NAMES and node.id not in ("True", "False", "None", "__name__"):
                return False
    return True


def find_function(text: str) -> Optional[Tuple[str, ast.Module]]:
    """(name of the first public top-level function, the parsed module) if ``text`` is Python with one and is short enough, else None."""
    if len(text) > MAX_CHARS:
        return None
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            return node.name, tree
    return None


def _limited(fn, *args):
    count = [0]

    def tracer(frame, event, arg):
        count[0] += 1
        if count[0] > STEP_LIMIT:
            raise StepLimit()
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        return fn(*args)
    finally:
        sys.settrace(old)


def behaviour(text: str) -> Tuple[Optional[tuple], bool]:
    """(behaviour, ran): ``ran`` is True when the code passed the checks and was executed; the behaviour is None when it was not run or accepts none of the probe kinds."""
    found = find_function(text)
    if found is None:
        return None, False
    name, tree = found
    if not is_safe(tree):
        return None, False
    ns = {"__builtins__": {**SAFE_BUILTINS, "__import__": _safe_import}, "__name__": "fsmoe_tool"}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _limited(exec, compile(tree, "<snippet>", "exec"), ns)
            f = ns[name]
            for kind in KINDS:
                outs = []
                for p in CO.PROBES[kind]:
                    try:
                        outs.append(repr(_limited(f, list(p) if kind == "list" else p))[:120])
                    except Exception:
                        break
                else:
                    return (kind, tuple(outs)), True
    except Exception:
        return None, True
    return None, True
