"""The engine's two outward-facing interfaces. The Hermes adapter implements both; tests use the fakes below."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Alert:
    run_id: str
    run_name: str
    title: str  # one line, e.g. "🌙 tinygpt: NaN at step 640 → rolled back to step 600, lr 3e-3 → 1.5e-3"
    body: str  # a few lines for the phone / chat
    page: bool  # True: push to the user's phone. False: pane + morning report only
    kind: str  # incident kind, or "finished" / "launched"
    chart_png: bytes | None = None
    details: dict[str, Any] = field(default_factory=dict)


class Notifier(Protocol):
    def send(self, alert: Alert) -> None: ...


class Diagnoser(Protocol):
    def diagnose(self, context: dict[str, Any]) -> str | None: ...


class NullNotifier:
    def send(self, alert: Alert) -> None:
        return None


class RecordingNotifier:
    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def send(self, alert: Alert) -> None:
        self.alerts.append(alert)


class NullDiagnoser:
    def diagnose(self, context: dict[str, Any]) -> str | None:
        return None
