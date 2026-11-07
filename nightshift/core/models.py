"""Engine data model. Plain dataclasses, no Hermes imports."""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Severity(StrEnum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"info": 0, "warn": 1, "critical": 2}[self.value]


class IncidentKind(StrEnum):
    NAN_INF = "nan_inf"
    SPIKE = "spike"
    DIVERGENCE = "divergence"
    PLATEAU = "plateau"
    OVERFIT = "overfit"
    THROUGHPUT = "throughput"
    STALL = "stall"
    CRASH = "crash"
    OOM = "oom"
    GPU_IDLE = "gpu_idle"


@dataclass(frozen=True)
class MetricPoint:
    """One scalar observation. ``wall`` is when it was observed (or logged, when the source records that)."""

    step: int
    name: str
    value: float
    wall: float


@dataclass
class Incident:
    kind: IncidentKind
    severity: Severity
    message: str
    metric: str | None = None
    step: int | None = None
    value: float | None = None
    onset_step: int | None = None  # first step the anomaly is visible at; rollback must predate it
    onset_wall: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        """Identity used to de-duplicate an incident that stays true across ticks."""
        return (self.kind.value, self.metric or "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "severity": self.severity.value,
            "message": self.message,
            "metric": self.metric,
            "step": self.step,
            "value": _json_float(self.value),
            "onset_step": self.onset_step,
            "onset_wall": self.onset_wall,
            "details": self.details,
        }


def _json_float(v: float | None) -> float | str | None:
    if v is None:
        return None
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


# ---------------------------------------------------------------------------------------------------------
# Metric naming. Trainers spell one thing many ways: loss, train/loss, train_loss, "train loss", eval/loss.

_SPACES = re.compile(r"\s+")
_LOSS = re.compile(r"(^|[/_.\-])loss($|[/_.\-])")
_VAL = re.compile(r"(^|[/_.\-])(val|valid|validation|eval|evaluation|test|dev)($|[/_.\-])")
_GRAD_NORM = re.compile(r"grad(ient)?[/_.\-]?norm")


def normalize_name(name: str) -> str:
    return _SPACES.sub("_", name.strip().lower())


def is_loss(name: str) -> bool:
    return bool(_LOSS.search(name))


def is_val(name: str) -> bool:
    return bool(_VAL.search(name))


def is_train_loss(name: str) -> bool:
    return is_loss(name) and not is_val(name)


def is_val_loss(name: str) -> bool:
    return is_loss(name) and is_val(name)


def is_grad_norm(name: str) -> bool:
    return bool(_GRAD_NORM.search(name))


# ---------------------------------------------------------------------------------------------------------


class MetricWindow:
    """Recent history per metric, the input every detector reads. Bounded so a week-long run stays cheap."""

    def __init__(self, max_points: int = 10_000) -> None:
        self.max_points = max_points
        self._series: dict[str, deque[MetricPoint]] = {}

    def add(self, point: MetricPoint) -> None:
        series = self._series.get(point.name)
        if series is None:
            series = self._series[point.name] = deque(maxlen=self.max_points)
        series.append(point)

    def extend(self, points: Iterable[MetricPoint]) -> None:
        for p in points:
            self.add(p)

    def names(self) -> list[str]:
        return sorted(self._series)

    def points(self, name: str) -> list[MetricPoint]:
        return list(self._series.get(name, ()))

    def values(self, name: str) -> list[float]:
        return [p.value for p in self._series.get(name, ())]

    def last(self, name: str) -> MetricPoint | None:
        series = self._series.get(name)
        return series[-1] if series else None

    def last_wall(self) -> float | None:
        walls = [s[-1].wall for s in self._series.values() if s]
        return max(walls) if walls else None

    def last_step(self) -> int | None:
        steps = [s[-1].step for s in self._series.values() if s]
        return max(steps) if steps else None

    def train_losses(self) -> list[str]:
        return [n for n in self.names() if is_train_loss(n)]

    def val_losses(self) -> list[str]:
        return [n for n in self.names() if is_val_loss(n)]

    def __len__(self) -> int:
        return sum(len(s) for s in self._series.values())
