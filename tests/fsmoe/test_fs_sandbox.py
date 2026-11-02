import os
import time

import pytest

from fsmoe import sandbox as SB
from fsmoe.domains import code as CO
from fsmoe.domains.base import rng_for
from fsmoe.domains.far import stdlib_code


def test_every_style_of_a_function_has_one_behaviour_and_every_function_a_different_one():
    rng = rng_for(7, "sandbox")
    by_skill = {}
    for skill in CO.SKILLS:
        sigs = set()
        for style in CO.TEACH + CO.HELDOUT:
            for _ in range(3):
                text, _fn = CO.render(skill, style, rng)
                sig, ran = SB.behaviour(text)
                assert ran and sig is not None, (skill, style)
                sigs.add(sig)
        assert len(sigs) == 1, (skill, [s[0] for s in sigs])
        by_skill[skill] = next(iter(sigs))
    assert len(set(by_skill.values())) == len(CO.SKILLS)


def test_unsafe_code_is_never_executed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # These are ATTACK TEXTS fed to the sandbox to prove it refuses them; nothing in this file evaluates or executes them itself.
    attacks = [
        "import os\ndef f(x):\n    os.makedirs('fsmoe_made_dir')\n    return 1\n",
        "def f(x):\n    open('fsmoe_made_file.txt', 'w').write('x')\n    return 1\n",
        "def f(x):\n    return __import__('os').getcwd()\n",
        "import sys\ndef f(x):\n    return sys.modules\n",
        "def f(x):\n    return ().__class__.__bases__\n",
        "def f(x):\n    return eval('1+1')\n",
        "import subprocess\ndef f(x):\n    return 1\n",
        "from os import path\ndef f(x):\n    return 1\n",
        "class A:\n    pass\ndef f(x):\n    return 1\n",
    ]
    for code in attacks:
        sig, ran = SB.behaviour(code)
        assert sig is None and not ran, code
    assert not list(tmp_path.iterdir())


def test_a_runaway_loop_is_stopped_by_the_step_limit():
    t0 = time.time()
    sig, ran = SB.behaviour("def f(x):\n    while True:\n        pass\n")
    assert sig is None and ran and time.time() - t0 < 30


def test_non_code_and_helper_only_texts_are_not_run():
    for text in ["I lost my card, can you help me?", "17 + 5", "def _hidden(x):\n    return x\n", "def f(x:\n", "x" * 5000]:
        assert SB.behaviour(text) == (None, False)


def test_texts_from_outside_code_are_never_run_and_real_library_functions_rarely_produce_a_behaviour():
    from fsmoe.domains.far import far_texts
    for name, texts in far_texts(60, 0).items():
        if name != "stdlib_code":
            assert not any(SB.behaviour(t)[1] for t in texts), name
    texts = stdlib_code(120, 0)
    ran = sum(1 for t in texts if SB.behaviour(t)[1])
    produced = sum(1 for t in texts if SB.behaviour(t)[0] is not None)
    assert ran < len(texts) and produced <= 0.05 * len(texts), (ran, produced, len(texts))        # running in the sandbox is harmless; a behaviour would open a spurious class


def test_a_reference_to_an_unbound_global_name_is_refused_before_running():
    assert SB.behaviour("def f(x):\n    return undefined_global_helper(x)\n") == (None, False)
    assert SB.behaviour("def f(x):\n    return x.append(1) or len(x)\n")[1]                    # arguments, locals and safe builtins are fine


def test_a_function_that_fails_on_every_probe_has_no_behaviour():
    sig, ran = SB.behaviour("def f(x):\n    return x.nonexistent_method()\n")
    assert sig is None and ran
