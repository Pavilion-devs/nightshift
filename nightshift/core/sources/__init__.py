"""Metric sources. Each one is polled for the points that appeared since the last poll."""

from __future__ import annotations

import os

from ..clock import Clock
from .base import FileFollower, MetricSource, parse_num
from .logtail import LogTailSource, parse_line
from .tabular import CsvSource, JsonlSource
from .tfevents import TfEventsSource

__all__ = [
    "CsvSource",
    "FileFollower",
    "JsonlSource",
    "LogTailSource",
    "MetricSource",
    "TfEventsSource",
    "make_source",
    "parse_line",
    "parse_num",
]


def make_source(spec: dict, clock: Clock | None = None) -> MetricSource:
    """Build a source from a JSON-able spec.

    ``{"type": "log" | "jsonl" | "csv" | "tfevents", "path": ..., "pattern": optional regex for logs}``
    """
    kind = spec.get("type")
    path = os.path.expanduser(str(spec.get("path", "")))
    if not path:
        raise ValueError("source spec needs a path")
    if kind == "log":
        return LogTailSource(path, pattern=spec.get("pattern"), clock=clock)
    if kind == "jsonl":
        return JsonlSource(path, clock=clock)
    if kind == "csv":
        return CsvSource(path, clock=clock)
    if kind == "tfevents":
        return TfEventsSource(path)
    raise ValueError(f"unknown source type {kind!r}")
