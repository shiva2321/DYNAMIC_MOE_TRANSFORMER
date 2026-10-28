"""Copy a piece of a GENERATED results file into a notebook entry (lab-notebook rule 1: numbers are copied by script from generated files, never retyped).

    python copy_table.py results/E1_TABLE.md                      # the whole file
    python copy_table.py results/E2_TABLE.md "## B3a"             # from the heading that starts with "## B3a" up to the next heading of the same or a higher level
    python copy_table.py results/E1_TABLE.md --rows R0_hash,R4_qwen_L26,R5_   # only the header lines and the table rows that start with one of these prefixes
    python copy_table.py --inject entry.md                        # replace every line "{{copy results/FILE.md | heading | rows}}" of an entry by the copied text

``--inject`` is how the entries E-044 and later were filled: the entry holds the marker, the script holds the numbers.
"""

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARK = re.compile(r"^\{\{copy\s+([^|}]+?)\s*(?:\|\s*([^|}]*?)\s*)?(?:\|\s*([^}]*?)\s*)?\}\}\s*$")


def section(text: str, heading: str) -> str:
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith(heading)), None)
    if start is None:
        raise ValueError(f"no heading starting with {heading!r}")
    level = len(re.match(r"#*", lines[start]).group())
    end = len(lines)
    for j in range(start + 1, len(lines)):
        m = re.match(r"(#+)\s", lines[j])
        if m and len(m.group(1)) <= level:
            end = j
            break
    return "\n".join(lines[start:end]).rstrip()


def only_rows(text: str, prefixes) -> str:
    out = []
    for i, l in enumerate(text.splitlines()):
        if not l.startswith("|"):
            out.append(l)
            continue
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if i < 2 or set(l.replace("|", "").strip()) <= set("-: ") or any(c.startswith(p) or c.strip("*").startswith(p) for c in cells[:2] for p in prefixes) or l.startswith("| reader") or l.startswith("| representation"):
            out.append(l)
    return "\n".join(out)


def copy(path: str, heading: str = "", rows: str = "") -> str:
    p = Path(path)
    p = p if p.is_absolute() else HERE / p
    text = p.read_text(encoding="utf-8")
    if heading:
        text = section(text, heading)
    if rows:
        text = only_rows(text, [r.strip() for r in rows.split(",") if r.strip()])
    return qualify(text.strip("\n"))


def qualify(text: str) -> str:
    """A path written as `results/FILE` in a file of the project folder means research/knowledge_status/results/FILE from the repository root; the notebook verifier needs the full path."""
    return re.sub(r"`(results/[^`\s]+)`", r"`research/knowledge_status/\1`", text)


def inject(entry_path: str) -> int:
    p = Path(entry_path)
    out, n = [], 0
    for line in p.read_text(encoding="utf-8").splitlines():
        m = MARK.match(line.strip())
        if m:
            out.append(copy(m.group(1), m.group(2) or "", m.group(3) or ""))
            n += 1
        else:
            out.append(line)
    p.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("heading", nargs="?", default="")
    ap.add_argument("--rows", default="")
    ap.add_argument("--inject", default="")
    a = ap.parse_args()
    if a.inject:
        print(f"{inject(a.inject)} markers replaced")
    elif a.path:
        sys.stdout.write(copy(a.path, a.heading, a.rows) + "\n")
    else:
        ap.error("give a path or --inject")
