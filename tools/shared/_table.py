"""Terminal-aware table printer shared by compare_checkpoints and trace_shapes."""

from __future__ import annotations

import shutil
from typing import Dict, List, Optional

_COLOURS: Dict[str, str] = {
    "green":  "\033[92m",
    "cyan":   "\033[96m",
    "red":    "\033[91m",
    "yellow": "\033[93m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}

STATUS_COLOUR: Dict[str, str] = {
    "MATCH":          "green",
    "MATCH~":         "cyan",
    "SHAPE MISMATCH": "red",
    "MISMATCH~":      "yellow",
    "UNMATCHED":      "red",
    "ONLY IN 2":      "yellow",
    "MISSING":        "red",
    "EXTRA":          "yellow",
}


def coloured(text: str, colour: Optional[str], enabled: bool) -> str:
    if not enabled or not colour or colour not in _COLOURS:
        return text
    return f"{_COLOURS[colour]}{text}{_COLOURS['reset']}"


def trunc(s: str, width: int) -> str:
    """Fit string into *width* chars, appending '...' if cut."""
    if width < 4:
        return s[:width]
    return s if len(s) <= width else s[: width - 3] + "..."


# ---------------------------------------------------------------------------

class Table:
    """Renders a fixed-layout table that fits the current terminal width.

    Column schema (by key):
        "idx"    – optional leading index column   (fixed width)
        "name1"  – first name column               (auto-sized)
        "shape1" – first shape column              (fixed width)
        "name2"  – second name column              (auto-sized)
        "shape2" – second shape column             (fixed width)
        "status" – status / result column          (fixed width)

    Both name columns share the available width equally after fixed columns
    are subtracted from the terminal width.
    """

    _W_IDX    = 5
    _W_SHAPE  = 20
    _W_STATUS = 16
    _MIN_NAME = 24

    def __init__(
        self,
        header1: str,           # label for name1 column
        header2: str,           # label for name2 column
        show_index: bool = False,
        use_colour: bool = True,
        min_name_w: int = _MIN_NAME,
    ):
        self._h1          = header1
        self._h2          = header2
        self._show_index  = show_index
        self._colour      = use_colour
        self._name_w      = max(self._auto_name_w(), min_name_w)

    def _auto_name_w(self) -> int:
        term = shutil.get_terminal_size(fallback=(180, 40)).columns
        n_fixed_cols = 4 + (1 if self._show_index else 0)  # shape1 shape2 status [idx]
        fixed_px = (
            self._W_SHAPE * 2
            + self._W_STATUS
            + (self._W_IDX if self._show_index else 0)
            + (n_fixed_cols + 2) * 3   # "| " per separator, leading + trailing
        )
        return max((term - fixed_px) // 2, self._MIN_NAME)

    # ------------------------------------------------------------------

    def _sep(self) -> str:
        parts = []
        if self._show_index:
            parts.append("-" * (self._W_IDX + 2))
        parts += [
            "-" * (self._name_w + 2),
            "-" * (self._W_SHAPE + 2),
            "-" * (self._name_w + 2),
            "-" * (self._W_SHAPE + 2),
            "-" * (self._W_STATUS + 2),
        ]
        return "+" + "+".join(parts) + "+"

    def _render(self, idx: str, n1: str, s1: str, n2: str, s2: str, st: str) -> str:
        nw = self._name_w
        cells = []
        if self._show_index:
            cells.append(f" {idx:<{self._W_IDX}} ")
        cells += [
            f" {trunc(n1, nw):<{nw}} ",
            f" {s1:<{self._W_SHAPE}} ",
            f" {trunc(n2, nw):<{nw}} ",
            f" {s2:<{self._W_SHAPE}} ",
            f" {st:<{self._W_STATUS}} ",
        ]
        return "|" + "|".join(cells) + "|"

    # ------------------------------------------------------------------

    def header(self) -> None:
        print(self._sep())
        line = self._render("", self._h1, "Shape 1", self._h2, "Shape 2", "Status")
        print(coloured(line, "bold", self._colour))
        print(self._sep())

    def row(self, n1: str, s1, n2: str, s2, status: str, idx: int = 0) -> None:
        colour = STATUS_COLOUR.get(status)
        s1_str = str(s1) if s1 else ""
        s2_str = str(s2) if s2 else ""
        line   = self._render(str(idx) if self._show_index else "", n1, s1_str, n2, s2_str, status)
        print(coloured(line, colour, self._colour))

    def footer(self) -> None:
        print(self._sep())

    def summary(self, counts: Dict[str, int], total: int) -> None:
        print()
        print("Summary:")
        for k, v in sorted(counts.items()):
            colour = STATUS_COLOUR.get(k)
            print(coloured(f"  {k:<22}: {v}", colour, self._colour))
        print(f"  {'TOTAL':<22}: {total}")
        exact = counts.get("MATCH", 0) + counts.get("MATCH~", 0)
        pct   = 100 * exact / total if total else 0.0
        print(f"\n  Matched (exact+fuzzy): {exact} / {total}  ({pct:.1f}%)")
