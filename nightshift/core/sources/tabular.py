"""Metrics from JSONL and CSV files that a trainer appends to."""

from __future__ import annotations

import csv
import json
import os

from ..clock import Clock, SystemClock
from ..models import MetricPoint, normalize_name
from .base import STEP_KEYS, FileFollower, as_float

_WALL_KEYS = ("wall", "wall_time", "time", "timestamp", "_timestamp")


def _row_points(row: dict, fallback_step: int, now: float) -> tuple[int, list[MetricPoint]]:
    step: int | None = None
    wall = now
    values: dict[str, float] = {}
    for key, raw in row.items():
        if key is None:
            continue
        name = normalize_name(str(key))
        val = as_float(raw)
        if val is None:
            continue
        if name in STEP_KEYS and step is None and val == val and float(val).is_integer():
            step = int(val)
        elif name in _WALL_KEYS and val > 1e9:  # looks like a unix timestamp
            wall = val
        else:
            values[name] = val
    step = fallback_step if step is None else step
    return step, [MetricPoint(step, n, v, wall) for n, v in values.items()]


class JsonlSource:
    def __init__(self, path: str | os.PathLike[str], clock: Clock | None = None) -> None:
        self.follower = FileFollower(path)
        self.clock = clock or SystemClock()
        self._rows = 0

    def poll(self) -> list[MetricPoint]:
        now = self.clock.now()
        out: list[MetricPoint] = []
        for line in self.follower.read_lines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            self._rows += 1
            _, pts = _row_points(row, self._rows, now)
            out.extend(pts)
        return out


class CsvSource:
    def __init__(self, path: str | os.PathLike[str], clock: Clock | None = None) -> None:
        self.follower = FileFollower(path)
        self.clock = clock or SystemClock()
        self._header: list[str] | None = None
        self._rows = 0

    def poll(self) -> list[MetricPoint]:
        now = self.clock.now()
        out: list[MetricPoint] = []
        lines = self.follower.read_lines()
        if not lines:
            return out
        for record in csv.reader(lines):
            if not record:
                continue
            if self._header is None:
                self._header = [c.strip() for c in record]
                continue
            if record == self._header:  # header repeated after a restart that re-created the file
                continue
            self._rows += 1
            row = dict(zip(self._header, record, strict=False))
            _, pts = _row_points(row, self._rows, now)
            out.extend(pts)
        return out
