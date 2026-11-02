"""Pin the Python environment this repository runs in (road plan M0 step 2).

Collects the third-party modules imported by every .py file that git knows about or has not ignored, maps them to the
installed distributions, adds their dependency closure and writes requirements-lock.txt at the repository root with the
versions installed now. Distributions installed from a local build (for example the Rust extension in crates/) are listed
as comments because pip cannot fetch them. Nothing is typed by hand. Run from anywhere:  python docs/roadmap/make_env_lock.py
"""
import ast
import importlib.metadata as md
import json
import pathlib
import platform
import subprocess
import sys
import warnings
from datetime import date

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

warnings.simplefilter("ignore", SyntaxWarning)    # old lab files contain invalid escape sequences; they only matter when parsed
ROOT = pathlib.Path(__file__).resolve().parents[2]


def source_files() -> list:
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "*.py"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [ROOT / p for p in out.splitlines() if p.strip()]


def imported_roots(path: pathlib.Path):
    """Top-level module names imported by one file, or None when the file cannot be parsed."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError, ValueError):
        return None
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def is_local_build(dist) -> bool:
    """True when pip recorded that the distribution came from a local path (PEP 610), so no index can supply it."""
    raw = dist.read_text("direct_url.json")
    return bool(raw) and json.loads(raw).get("url", "").startswith("file:")


def closure(direct: set) -> dict:
    """Installed distributions reachable from the direct ones: name -> (version, local build?)."""
    seen, queue = {}, list(direct)
    while queue:
        name = canonicalize_name(queue.pop())
        if name in seen:
            continue
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        seen[name] = (dist.version, is_local_build(dist))
        for line in dist.requires or []:
            req = Requirement(line)
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            queue.append(req.name)
    return seen


def main() -> None:
    files = source_files()
    rel = [f.relative_to(ROOT) for f in files]
    # a bare `import foo` in a lab script usually means a sibling file or a folder of this repository, not a distribution
    local = {p.stem for p in rel} | {part for p in rel for part in p.parts[:-1]}
    roots, unparsed = set(), []
    for f in files:
        found = imported_roots(f)
        if found is None:
            unparsed.append(str(f.relative_to(ROOT)))
        else:
            roots |= found
    third = sorted(n for n in roots if n not in sys.stdlib_module_names and n not in local and n != "__future__")
    owners = md.packages_distributions()
    direct, unresolved = set(), []
    for n in third:
        if n in owners:
            direct.update(owners[n])
        else:
            unresolved.append(n)
    pinned = closure(direct)
    torch_version = pinned.get("torch", ("", False))[0]
    lines = [
        "# Environment lock of the road plan (M0 step 2), written by docs/roadmap/make_env_lock.py",
        f"# Python {platform.python_version()} on {platform.platform()}, {date.today().isoformat()}",
        f"# {len(files)} source files scanned, {len(direct)} direct distributions, {len(pinned)} pinned in all",
        f"# files that could not be parsed: {len(unparsed)}; imported names with no installed distribution: {', '.join(unresolved) or 'none'}",
    ]
    if "+cu" in torch_version:
        lines.append(f"--extra-index-url https://download.pytorch.org/whl/{torch_version.split('+')[1]}")
    for name, (version, local_build) in sorted(pinned.items()):
        lines.append(f"# {name}=={version}   (installed from a local build, not on an index)" if local_build else f"{name}=={version}")
    (ROOT / "requirements-lock.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"scanned {len(files)} files; {len(direct)} direct, {len(pinned)} pinned, {sum(1 for v in pinned.values() if v[1])} local builds")
    print("unparsed files:", ", ".join(unparsed) or "none")
    print("imported names with no installed distribution:", ", ".join(unresolved) or "none")


if __name__ == "__main__":
    main()
