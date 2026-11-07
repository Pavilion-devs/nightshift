"""Shared plumbing for incremental metric sources."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Protocol

from ..models import MetricPoint

NUM = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf(?:inity)?)"
_NUM_RE = re.compile(rf"^(?:{NUM})$", re.I)

STEP_KEYS = ("step", "global_step", "iter", "iteration", "it", "trainer/global_step", "_step")


class MetricSource(Protocol):
    """Anything that yields new metric points each time it is polled."""

    def poll(self) -> list[MetricPoint]: ...


def parse_num(text: str) -> float | None:
    text = text.strip().rstrip("%")
    if not _NUM_RE.match(text):
        return None
    try:
        return float(text)
    except ValueError:  # pragma: no cover - regex already guards this
        return None


def as_float(value: object) -> float | None:
    """Numeric JSON/CSV values only; bools and strings that are not numbers are skipped."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return parse_num(value)
    return None


def finite_or_flagged(v: float) -> bool:
    """True for any real float, NaN/Inf included: those are exactly what we want to see."""
    return isinstance(v, float) and (math.isfinite(v) or math.isnan(v) or math.isinf(v))


class FileFollower:
    """Reads complete new lines from a growing file, like ``tail -F``.

    Survives the file not existing yet, truncation (offset reset) and rotation (inode change). A trailing
    partial line is held back until its newline arrives, so a half-written number is never parsed.
    """

    def __init__(self, path: str | os.PathLike[str], max_read: int = 4 * 1024 * 1024) -> None:
        self.path = Path(path)
        self.max_read = max_read
        self._offset = 0
        self._inode: int | None = None
        self._partial = b""

    def read_lines(self) -> list[str]:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return []
        if self._inode is not None and st.st_ino != self._inode:
            self._offset, self._partial = 0, b""  # rotated
        if st.st_size < self._offset:
            self._offset, self._partial = 0, b""  # truncated
        self._inode = st.st_ino
        if st.st_size == self._offset:
            return []
        with self.path.open("rb") as fh:
            fh.seek(self._offset)
            chunk = fh.read(self.max_read)
        self._offset += len(chunk)
        data = self._partial + chunk
        *complete, self._partial = data.split(b"\n")
        # Progress bars rewrite one line with \r; each segment is its own update.
        lines: list[str] = []
        for raw in complete:
            for seg in raw.split(b"\r"):
                text = seg.decode("utf-8", errors="replace").strip()
                if text:
                    lines.append(text)
        return lines
