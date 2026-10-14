"""Install the validated stage-3 assets into the repository, one stage at a time (docs/superpowers/plans/2026-10-13-routing-v2-stage3.md).

    python docs/superpowers/plans/2026-10-13-routing-v2-stage3-assets/apply_stage3.py <stage> [--root DIR]

Stages, in the order the plan runs them (tests first, then code):
    golden   A_tests   A_code   B_tests   B_code   C_tests   C_code   D_tests   D_code   E_code      (the first build, installed on 2026-10-13)
    F_tests  F_code                                                                                 (amendments A-15 persistence and A-16 group veto level, 2026-10-14)
A stage copies its asset files to their final places (a file that already exists is never overwritten unless it is byte-identical), REPLACES files of an earlier
stage by their v2 versions (only if the file on disk is still byte-identical to the original asset: otherwise the stage stops), and applies its edits (every old
text must occur exactly once). Applied stages are recorded in <root>/.stage3_applied; a stage cannot run twice, and a stage whose predecessor has not run is
refused. `--root` defaults to the repository root (three levels above this file's folder).
"""

import hashlib
import importlib.util
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
FILES, FILES2, EDITS = HERE / "files", HERE / "files_v2", HERE / "edits"
T = "tests/growmoe/"
G = "growmoe/"
R = "research/growmoe_results/rv2c/"

STAGES = {
    "golden": dict(copy=[R + "golden.py"], edits=[]),
    "A_tests": dict(copy=[T + "test_gm_rv3_settest.py", T + "test_gm_rv3_sources.py"], edits=[]),
    "A_code": dict(copy=[G + "settest.py", G + "sources.py"], edits=["edits_config.py"]),
    "B_tests": dict(copy=[T + "test_gm_rv3_setgate.py", T + "test_gm_rv3_groupvote.py"], edits=[]),
    "B_code": dict(copy=[G + "setgate.py", G + "groupvote.py"], edits=[]),
    "C_tests": dict(copy=[T + "test_gm_rv3_pool.py"], edits=[]),
    "C_code": dict(copy=[], edits=["edits_phylo.py", "edits_accounting.py"]),
    "D_tests": dict(copy=[T + "test_gm_rv3_groupeval.py", T + "test_gm_rv3_verdicts.py"], edits=[]),
    "D_code": dict(copy=[G + "groupeval.py", G + "streams_order.py"], edits=["edits_fairfight.py"]),
    "E_code": dict(copy=[R + "run_rv2c.py"], edits=[]),
    # amendments A-15 (persistence) and A-16 (group veto level): new files come from files_v2/, replaced files must still be the original asset
    "F_tests": dict(copy=[], copy_v2=[T + "test_gm_rv3_confirm.py"], replace=[T + "test_gm_rv3_pool.py"], edits=[]),
    "F_code": dict(copy=[], replace=[G + "setgate.py", R + "run_rv2c.py"],
                   edits=["edits_v2/edits_confirm_config.py", "edits_v2/edits_confirm_phylo.py", "edits_v2/edits_confirm_fairfight.py"]),
}
ORDER = list(STAGES)


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_edit_module(root: pathlib.Path, path: pathlib.Path) -> None:
    spec = importlib.util.spec_from_file_location("edits", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    target = root / mod.FILE
    text = target.read_bytes().decode("utf-8")
    crlf = "\r\n" in text
    text = text.replace("\r\n", "\n")
    for name, old, new in mod.EDITS:
        if old is None:
            text = text.rstrip("\n") + "\n" + new
            print(f"  appended {mod.FILE}:{name}")
            continue
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"EDIT FAILED {mod.FILE}:{name}: the old text occurs {n} times (need exactly 1). Nothing of this stage was written for that file.")
        text = text.replace(old, new)
        print(f"  applied  {mod.FILE}:{name}")
    target.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))


def main() -> None:
    args = sys.argv[1:]
    root = pathlib.Path(args[args.index("--root") + 1]).resolve() if "--root" in args else HERE.parents[3]
    stage = next((a for a in args if a in STAGES), None)
    if stage is None:
        sys.exit(f"usage: apply_stage3.py <{'|'.join(ORDER)}> [--root DIR]")
    mark = root / ".stage3_applied"
    done = mark.read_text().split() if mark.exists() else []
    if stage in done:
        sys.exit(f"stage {stage} was already applied to {root}")
    missing = [s for s in ORDER[:ORDER.index(stage)] if s not in done]
    if missing:
        sys.exit(f"stage {stage} needs {missing} first (applied so far: {done})")
    plan = STAGES[stage]
    for rel in plan.get("copy", []):                                                    # check everything before writing anything
        src, dst = FILES / rel, root / rel
        if not src.exists():
            sys.exit(f"asset missing: {src}")
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            sys.exit(f"{dst} already exists and differs from the asset; stop and report (do not overwrite)")
    for rel in plan.get("copy_v2", []):
        src, dst = FILES2 / rel, root / rel
        if not src.exists():
            sys.exit(f"asset missing: {src}")
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            sys.exit(f"{dst} already exists and differs from the asset; stop and report (do not overwrite)")
    for rel in plan.get("replace", []):
        old_asset, new_asset, dst = FILES / rel, FILES2 / rel, root / rel
        if not new_asset.exists() or not old_asset.exists():
            sys.exit(f"asset missing for {rel}")
        if not dst.exists():
            sys.exit(f"{dst} does not exist; stage {stage} replaces it")
        if sha(dst) != sha(old_asset) and sha(dst) != sha(new_asset):
            sys.exit(f"{dst} is neither the original asset nor the v2 asset (it was edited); stop and report (do not overwrite)")
    for rel in plan.get("copy", []):
        src, dst = FILES / rel, root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"  copied   {rel}")
    for rel in plan.get("copy_v2", []):
        src, dst = FILES2 / rel, root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"  copied   {rel} (v2)")
    for rel in plan.get("replace", []):
        shutil.copyfile(FILES2 / rel, root / rel)
        print(f"  replaced {rel} (v2)")
    for name in plan["edits"]:
        apply_edit_module(root, (HERE / name) if "/" in name else (EDITS / name))
    mark.write_text(" ".join(done + [stage]) + "\n")
    print(f"stage {stage} applied to {root}")


if __name__ == "__main__":
    main()
