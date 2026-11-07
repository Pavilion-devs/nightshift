"""Metrics from a trainer's stdout/stderr log.

Understands, without configuration:
  * JSON lines                          {"step": 10, "loss": 2.31}
  * Python dict reprs (HF Trainer)      {'loss': 2.31, 'grad_norm': 1.2, 'learning_rate': 5e-05, 'epoch': 0.1}
  * key=value / key: value pairs        step=10 loss=2.31 lr=3e-4   |   Epoch 1 | step: 10 | loss: 2.31
  * nanoGPT style                       iter 100: loss 2.3456, time 12.34ms
                                        step 200: train loss 2.1, val loss 2.2
A custom regex with named groups (``step`` plus one group per metric) overrides all of the above.
"""

from __future__ import annotations

import ast
import json
import os
import re

from ..clock import Clock, SystemClock
from ..models import MetricPoint, normalize_name
from .base import NUM, STEP_KEYS, FileFollower, as_float

_STEP_RE = re.compile(r"\b(?:global[_ ]step|step|iter(?:ation)?|it)\b\s*[:=#]?\s*(\d+)(?:\s*/\s*\d+)?", re.I)
_PAIR_RE = re.compile(rf"(?<![\w./-])([A-Za-z][\w./-]*)\s*[:=]\s*({NUM})%?(?![\w.])", re.I)
# "loss 2.31", "train loss 2.1", "val loss 2.2", "grad_norm 1.3": a word, a space, a number.
_SPACED_RE = re.compile(
    r"\b((?:(?:train|training|val|valid|validation|eval|test)\s+)?"
    r"(?:loss|grad[_ ]?norm|lr|perplexity|ppl|acc(?:uracy)?))"
    rf"\s+({NUM})(?![\w.])",
    re.I,
)
_BAR_STEP_RE = re.compile(r"\|\s*(\d+)/\d+\s*\[")  # tqdm / Lightning: "| 450/1000 [00:12<00:15"
_IGNORED = {"time", "dt", "eta", "elapsed", "epoch_time", "v_num", "pid", "rank", "seed"}


def _dict_metrics(obj: dict) -> tuple[int | None, dict[str, float]]:
    step: int | None = None
    metrics: dict[str, float] = {}
    for key, raw in obj.items():
        name = normalize_name(str(key))
        val = as_float(raw)
        if val is None:
            continue
        if name in STEP_KEYS and step is None and float(val).is_integer():
            step = int(val)
            continue
        metrics[name] = val
    return step, metrics


def parse_line(line: str, custom: re.Pattern[str] | None = None) -> tuple[int | None, dict[str, float]]:
    """Return (step or None, {metric: value}) for one log line. Pure; exposed for tests."""
    line = line.strip()
    if custom is not None:
        m = custom.search(line)
        if not m:
            return None, {}
        groups = m.groupdict()
        step = int(groups.pop("step")) if groups.get("step") else None
        metrics = {
            normalize_name(k): v for k, raw in groups.items() if raw and (v := as_float(raw)) is not None
        }
        return step, metrics

    if line.startswith("{") and line.endswith("}"):
        obj = None
        try:
            obj = json.loads(line)
        except ValueError:
            try:
                obj = ast.literal_eval(line)  # literals only; never executes code
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                obj = None
        if isinstance(obj, dict):
            return _dict_metrics(obj)

    step_m = _STEP_RE.search(line) or _BAR_STEP_RE.search(line)
    step = int(step_m.group(1)) if step_m else None
    metrics: dict[str, float] = {}
    for name, raw in _PAIR_RE.findall(line):
        key = normalize_name(name)
        if key in STEP_KEYS or key in _IGNORED:
            continue
        val = as_float(raw)
        if val is not None:
            metrics[key] = val
    for name, raw in _SPACED_RE.findall(line):
        key = normalize_name(name)
        if key.startswith("training_"):
            key = "train_" + key[len("training_") :]
        val = as_float(raw)
        if val is not None and key not in metrics:
            metrics[key] = val
    return step, metrics


class LogTailSource:
    """Follow a log file and turn each line into metric points."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        pattern: str | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.follower = FileFollower(path)
        self.custom = re.compile(pattern) if pattern else None
        self.clock = clock or SystemClock()
        self._last_step: int | None = None
        self._auto_step = 0
        self.lines_seen = 0
        self.tail: list[str] = []  # last lines, for crash classification and diagnosis

    def poll(self) -> list[MetricPoint]:
        now = self.clock.now()
        points: list[MetricPoint] = []
        for line in self.follower.read_lines():
            self.lines_seen += 1
            self.tail.append(line)
            if len(self.tail) > 200:
                del self.tail[:-200]
            step, metrics = parse_line(line, self.custom)
            if not metrics:
                continue
            if step is None:
                # No step on this line (HF dicts): reuse the last one seen, or count lines that carry metrics.
                if self._last_step is None:
                    self._auto_step += 1
                    step = self._auto_step
                else:
                    step = self._last_step
            else:
                self._last_step = step
            points.extend(MetricPoint(step, name, value, now) for name, value in metrics.items())
        return points
