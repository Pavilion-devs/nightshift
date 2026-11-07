"""Synthetic but realistic training curves for detector tests.

Clean curves cover the shapes real runs produce that a naive detector mistakes for failure: noisy small-batch
losses, warmup bumps, step-schedule drops, cosine decay, warm restarts, long flat-ish tails, logging jitter.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

from nightshift.core.models import MetricPoint, MetricWindow

T0 = 1_700_000_000.0


def window_from(
    values: list[float], name: str = "loss", *, every: int = 10, dt: float = 1.0, t0: float = T0
) -> MetricWindow:
    w = MetricWindow()
    for i, v in enumerate(values):
        w.add(MetricPoint((i + 1) * every, name, v, t0 + i * dt))
    return w


def add_series(
    w: MetricWindow, values: list[float], name: str, *, every: int = 10, dt: float = 1.0, t0: float = T0
) -> MetricWindow:
    for i, v in enumerate(values):
        w.add(MetricPoint((i + 1) * every, name, v, t0 + i * dt))
    return w


def exp_decay(
    n: int, seed: int = 0, noise: float = 0.02, floor: float = 1.2, scale: float = 3.0, tau: float = 40
):
    rng = random.Random(seed)
    return [(floor + scale * math.exp(-i / tau)) * (1 + rng.gauss(0, noise)) for i in range(n)]


def warmup_then_decay(n: int, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        bump = 0.4 * math.sin(math.pi * i / 30) if i < 30 else 0.0  # loss rises a little during lr warmup
        out.append((1.3 + 2.5 * math.exp(-i / 60) + bump) * (1 + rng.gauss(0, 0.03)))
    return out


def step_schedule(n: int, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    out, level = [], 2.6
    for i in range(n):
        if i in (n // 3, 2 * n // 3):
            level *= 0.8  # lr drop -> sudden improvement
        out.append(level * (1 + rng.gauss(0, 0.025)) + 0.3 * math.exp(-i / 20))
    return out


def cosine(n: int, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [(1.1 + 1.8 * (1 + math.cos(math.pi * i / n)) / 2) * (1 + rng.gauss(0, 0.02)) for i in range(n)]


def warm_restarts(n: int, seed: int = 0) -> list[float]:
    """SGDR: loss climbs a little after each restart, never anywhere near 1.5x the best."""
    rng = random.Random(seed)
    period = max(1, n // 4)
    out = []
    for i in range(n):
        k = i % period
        base = 1.4 + 1.0 * math.exp(-i / (n / 2))
        out.append((base + 0.15 * (1 - k / period)) * (1 + rng.gauss(0, 0.02)))
    return out


def heavy_noise(n: int, seed: int = 0) -> list[float]:
    """Tiny batches: 8% multiplicative noise with occasional mild outliers (1.4x)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        v = (1.5 + 2.0 * math.exp(-i / 50)) * (1 + rng.gauss(0, 0.08))
        if rng.random() < 0.01:
            v *= 1.4
        out.append(v)
    return out


CLEAN: dict[str, Callable[[int, int], list[float]]] = {
    "exp_decay": lambda n, s: exp_decay(n, s),
    "warmup": warmup_then_decay,
    "step_schedule": step_schedule,
    "cosine": cosine,
    "warm_restarts": warm_restarts,
    "heavy_noise": heavy_noise,
}


def with_nan(values: list[float], at: int) -> list[float]:
    return values[:at] + [math.nan] * (len(values) - at)


def with_spike(values: list[float], at: int, factor: float = 10.0) -> list[float]:
    out = list(values)
    out[at] = out[at] * factor
    return out


def with_divergence(values: list[float], at: int, growth: float = 1.06) -> list[float]:
    out = list(values[:at])
    v = values[at - 1]
    for _ in range(len(values) - at):
        v *= growth
        out.append(v)
    return out
