"""Self-contained, dependency-free progress bar for long-running tasks.

An Ultralytics-style live progress bar (header columns + bar + rate + ETA + a live
status field for losses/metrics) that is TTY-aware:

  * Interactive terminal (``stdout.isatty()``): a live in-place bar updated with ``\\r``.
  * Non-TTY (redirected to a file / cron): a one-line summary printed at a coarse
    interval (``file_interval`` seconds) and once at the end, with no ``\\r``.

Used across train/val and the batch tools (eval, re-encode, export, infer). The
``progress`` helper wraps an iterable; the ``Progress`` class is a context manager.
"""

from __future__ import annotations

import shutil
import sys
import time
from typing import Optional, TextIO


def _fmt_time(seconds: float) -> str:
    """``mm:ss`` or ``h:mm:ss``."""
    if seconds < 0 or seconds != seconds:   # negative / NaN
        return "??:??"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class Progress:
    """A TTY-aware progress bar / periodic logger.

    Args:
        total: total number of steps; ``None`` for an unknown length (no bar/ETA, just
            a running count + rate).
        desc: left-hand label; can be replaced each ``update`` (e.g. the formatted metric
            row). Aligns under ``header`` when both are given.
        unit: rate unit shown as ``{unit}/s`` (e.g. ``it``, ``img``, ``batch``).
        header: a static column-header line printed once before the first bar (the
            Ultralytics look). Optional.
        min_interval: TTY refresh throttle in seconds (avoid flooding the terminal).
        file_interval: non-TTY print interval in seconds (keeps log files readable).
        stream: output stream (default ``sys.stdout``).
        enable: set ``False`` to silence entirely (e.g. for nested/quiet runs).
    """

    def __init__(self, total: Optional[int] = None, desc: str = "", unit: str = "it",
                 header: Optional[str] = None, min_interval: float = 0.1,
                 file_interval: float = 15.0, stream: Optional[TextIO] = None,
                 enable: bool = True):
        self.total = int(total) if total is not None else None
        self.desc = desc
        self.unit = unit
        self.header = header
        self.min_interval = min_interval
        self.file_interval = file_interval
        self.stream = stream or sys.stdout
        self.enable = enable

        self.n = 0
        self._status = ""
        self._start = time.monotonic()
        self._last_render = 0.0
        self._rate = 0.0            # EMA-smoothed steps/sec
        self._last_n = 0
        self._last_t = self._start
        self._header_done = False
        self._closed = False
        self._final_rendered = False
        try:
            self._is_tty = bool(self.stream.isatty())
        except Exception:
            self._is_tty = False

    def __enter__(self) -> "Progress":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def update(self, n: int = 1, desc: Optional[str] = None,
               status: Optional[str] = None) -> None:
        """Advance by ``n`` steps; optionally update the left label / right status."""
        if not self.enable:
            return
        self.n += n
        if desc is not None:
            self.desc = desc
        if status is not None:
            self._status = status

        now = time.monotonic()
        dt = now - self._last_t
        if dt > 0:
            inst = (self.n - self._last_n) / dt
            self._rate = inst if self._rate == 0.0 else 0.7 * self._rate + 0.3 * inst
            self._last_n, self._last_t = self.n, now

        interval = self.min_interval if self._is_tty else self.file_interval
        done = self.total is not None and self.n >= self.total
        if done or (now - self._last_render) >= interval:
            self._render(final=done)
            self._last_render = now

    def set_status(self, status: str) -> None:
        self._status = status

    def _bar_str(self, width: int) -> str:
        if self.total is None or self.total <= 0:
            return ""
        frac = min(self.n / self.total, 1.0)
        filled = int(width * frac)
        return "█" * filled + ("" if filled >= width else "░" * (width - filled))

    def _line(self, for_tty: bool) -> str:
        elapsed = time.monotonic() - self._start
        rate = self._rate
        parts = []
        if self.desc:
            parts.append(self.desc)
        if self.total is not None:
            pct = min(100, int(100 * self.n / self.total)) if self.total else 100
            eta = (self.total - self.n) / rate if rate > 0 else float("nan")
            count = f"{self.n}/{self.total}"
            timing = f"[{_fmt_time(elapsed)}<{_fmt_time(eta)}, {rate:.1f}{self.unit}/s]"
            if for_tty:
                # leave room for: " {pct}%|" + "| {count} {timing} {status}"
                cols = shutil.get_terminal_size((100, 20)).columns
                fixed = " ".join(p for p in parts) + f" {pct:3d}%|| {count} {timing} {self._status}"
                bar_w = max(5, min(40, cols - len(fixed) - 1))
                return (" ".join(parts) + f" {pct:3d}%|{self._bar_str(bar_w)}| "
                        f"{count} {timing} {self._status}").rstrip()
            return (" ".join(parts) + f" {count} ({pct}%) {timing} {self._status}").rstrip()
        # unknown total: count + rate only
        timing = f"[{_fmt_time(elapsed)}, {rate:.1f}{self.unit}/s]"
        return (" ".join(parts) + f" {self.n}{self.unit} {timing} {self._status}").rstrip()

    def _render(self, final: bool = False) -> None:
        if not self.enable or self._final_rendered:
            return
        if final:
            self._final_rendered = True
        if self.header and not self._header_done:
            self.stream.write(self.header.rstrip() + "\n")
            self._header_done = True
        if self._is_tty:
            self.stream.write("\r" + self._line(for_tty=True) + "\033[K")
            if final:
                self.stream.write("\n")
            self.stream.flush()
        else:
            self.stream.write(self._line(for_tty=False) + "\n")
            self.stream.flush()

    def close(self) -> None:
        """Render the final state once (and a trailing newline on a TTY)."""
        if self._closed or not self.enable:
            return
        self._closed = True
        self._render(final=True)


def progress(iterable, total: Optional[int] = None, desc: str = "", unit: str = "it",
             header: Optional[str] = None, **kw):
    """Wrap an iterable with a :class:`Progress` bar (auto-advances one per item)."""
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    p = Progress(total=total, desc=desc, unit=unit, header=header, **kw)
    try:
        for item in iterable:
            yield item
            p.update(1)
    finally:
        p.close()
