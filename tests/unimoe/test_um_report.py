"""The report generator (unimoe/report.py): every section, the prediction scoreboard and the figures, from real (tiny) run records."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_um_runner as TR  # noqa: E402

from unimoe import report as RP  # noqa: E402
from unimoe import runner as RN  # noqa: E402
from unimoe import systems as SY  # noqa: E402


def tiny_records():
    streams, records = {}, {}
    for sch in ("seq", "mix"):
        st = TR.stream(sch)
        streams[(sch, 0)] = st
        for name in ("shipped", "finetune"):
            records[(sch, name, 0)] = RN.run_system(lambda name=name: SY.make_system(name, st.n_classes, 0, {"inner_steps": 2}), st, 0, system=name)
    return records, streams


def test_the_report_has_every_section_a_scoreboard_and_figures(tmp_path):
    records, streams = tiny_records()
    text = RP.build(records, streams, {"shipped": {"inner_steps": 2}}, tmp_path / "report.md", tmp_path / "figs")
    for head in ("## Prediction scoreboard", "## A. Performance", "## B. Continual", "## C. Structure", "## D. Routing", "## E. Novelty", "## F. Cost", "## G. Arithmetic", "## H. Failure analysis",
                 "## I. Representation geometry", "## J. Tuned recipes", "## Figures"):
        assert head in text, head
    assert "| finetune |" in text and "| shipped |" in text
    assert (tmp_path / "report.md").exists() and (tmp_path / "figs" / "accuracy_matrices.png").exists() and (tmp_path / "figs" / "experts_over_time.png").exists()
    assert "best-of-seeds warning" in text                                                       # the disclosure the repository's hook asks for


def test_collect_reads_back_the_cells_a_run_wrote(tmp_path):
    records, streams = tiny_records()
    for (sch, name, seed), rec in records.items():
        RN.save_record(rec, tmp_path, "stamp")
    got = RP.collect(tmp_path)
    assert set(got) == set(records) and all((got[k].R == records[k].R).all() for k in got)
