"""Tests for the self-contained progress bar (tools/shared/progress.py).

The behaviour that matters most: non-TTY (cloud log file) output must be clean — no
carriage returns, header once, and exactly one final line.
"""

import io

from tools.shared.progress import Progress, progress, _fmt_time


def _run(total, n_updates, **kw):
    buf = io.StringIO()
    p = Progress(total=total, stream=buf, file_interval=0.0, **kw)
    for i in range(n_updates):
        p.update(1, status=f"x={i}")
    p.close()
    return buf.getvalue()


def test_non_tty_has_no_carriage_returns():
    out = _run(5, 5, desc="Job", unit="img")
    assert "\r" not in out
    assert "img/s" in out and "5/5 (100%)" in out


def test_header_printed_once():
    out = _run(3, 3, desc="Job", header="  col1  col2")
    assert out.count("  col1  col2") == 1


def test_no_duplicate_final_line():
    # the last update reaches total (renders final); close() must not repeat it
    out = _run(4, 4, desc="Job")
    final_lines = [l for l in out.splitlines() if "4/4 (100%)" in l]
    assert len(final_lines) == 1


def test_unknown_total_has_no_percent():
    out = _run(None, 3, desc="Scan", unit="file")
    assert "%" not in out and "3file" in out


def test_close_is_idempotent():
    buf = io.StringIO()
    p = Progress(total=2, stream=buf, file_interval=0.0)
    p.update(2)
    p.close()
    before = buf.getvalue()
    p.close()                      # second close — no extra output
    assert buf.getvalue() == before


def test_disabled_is_silent():
    buf = io.StringIO()
    p = Progress(total=5, stream=buf, enable=False)
    for _ in range(5):
        p.update(1)
    p.close()
    assert buf.getvalue() == ""


def test_progress_iterable_wrapper():
    buf = io.StringIO()
    items = list(progress(range(4), desc="Loop", stream=buf, file_interval=0.0))
    assert items == [0, 1, 2, 3]


def test_fmt_time():
    assert _fmt_time(5) == "00:05"
    assert _fmt_time(65) == "01:05"
    assert _fmt_time(3661) == "1:01:01"
    assert _fmt_time(float("nan")) == "??:??"
