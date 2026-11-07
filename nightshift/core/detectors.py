"""Failure detectors. Pure functions over a MetricWindow: no I/O, no clocks, no state.

Each detector looks at the *current* tail of the run and reports what is true right now, at most one incident
per metric. The supervisor de-duplicates incidents that stay true across ticks (``Incident.key``).
Every threshold is in ``DetectorConfig``; every detector ships with a firing fixture and a clean-curve
fixture in the tests.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields
from statistics import median
from typing import Any

from .models import Incident, IncidentKind, MetricPoint, MetricWindow, Severity, is_grad_norm

_K = IncidentKind


@dataclass
class DetectorConfig:
    # which metrics to watch (None = auto-detect by name)
    train_metric: str | None = None
    val_metric: str | None = None
    disabled: list[str] = field(default_factory=list)
    # spike: robust z-score of the newest point against the recent window
    spike_window: int = 50
    spike_min_history: int = 20
    spike_z: float = 6.0
    spike_ema_ratio: float = 2.0
    spike_escalate_within: int = 3  # two spikes within this many points -> critical
    # divergence: smoothed loss stays far above its best for a while
    ema_span: int = 20
    divergence_ratio: float = 1.5
    divergence_patience: int = 30
    divergence_min_points: int = 50
    # plateau
    plateau_min_steps: int = 500
    plateau_tail_frac: float = 0.15
    plateau_min_tail_points: int = 10
    plateau_min_rel_improvement: float = 0.005
    # overfit
    overfit_consecutive: int = 3
    overfit_min_rel_rise: float = 0.02  # noise on a converged val loss wobbles ~1%; real overfitting climbs
    overfit_train_tolerance: float = (
        0.02  # train may be flat within noise (2%); it just must not be climbing too
    )
    # onset refinement: walk back from the first bad point while the loss is clearly above its recent norm
    onset_ratio: float = 1.15
    onset_baseline_points: int = 20
    # throughput
    throughput_window_s: float = 120.0
    throughput_drop_ratio: float = 0.5
    throughput_min_intervals: int = 20
    # stall
    stall_min_s: float = 300.0
    stall_interval_mult: float = 10.0
    startup_grace_s: float = 900.0
    # gpu idle
    gpu_idle_util: float = 5.0
    gpu_idle_s: float = 300.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DetectorConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


# --- helpers ----------------------------------------------------------------------------------------------


def ema(values: Sequence[float], span: int) -> list[float]:
    """Exponential moving average that skips non-finite values (carries the previous average)."""
    alpha = 2.0 / (span + 1)
    out: list[float] = []
    cur: float | None = None
    for v in values:
        if math.isfinite(v):
            cur = v if cur is None else alpha * v + (1 - alpha) * cur
        out.append(cur if cur is not None else math.nan)
    return out


def robust_z(x: float, history: Sequence[float]) -> float:
    med = median(history)
    mad = median(abs(h - med) for h in history)
    scale = 1.4826 * mad
    floor = 1e-3 * abs(med) + 1e-12  # a perfectly flat history must not make every wiggle infinite
    return (x - med) / max(scale, floor)


def _train_metrics(w: MetricWindow, cfg: DetectorConfig) -> list[str]:
    if cfg.train_metric:
        return [cfg.train_metric] if cfg.train_metric in w.names() else []
    return w.train_losses()


def _val_metrics(w: MetricWindow, cfg: DetectorConfig) -> list[str]:
    if cfg.val_metric:
        return [cfg.val_metric] if cfg.val_metric in w.names() else []
    return w.val_losses()


def _finite(points: Sequence[MetricPoint]) -> list[MetricPoint]:
    return [p for p in points if math.isfinite(p.value)]


def refine_onset(points: Sequence[MetricPoint], idx: int, cfg: DetectorConfig) -> int:
    """Index where the run first left its healthy trend, at or before ``idx``.

    A loss that ends in NaN usually climbs for a while first; checkpoints saved during that climb are already
    poisoned. Walk back from ``idx`` while each point is clearly above the median of the points before it.
    """
    onset = idx
    j = idx - 1
    while j > 0:
        v = points[j].value
        base = [p.value for p in points[max(0, j - cfg.onset_baseline_points) : j] if math.isfinite(p.value)]
        if not math.isfinite(v) or len(base) < 3:
            break
        ref = median(base)
        if ref > 0 and v > cfg.onset_ratio * ref:
            onset = j
            j -= 1
        else:
            break
    return onset


# --- detectors --------------------------------------------------------------------------------------------


def detect_nan_inf(w: MetricWindow, cfg: DetectorConfig) -> list[Incident]:
    watched = (
        set(_train_metrics(w, cfg)) | set(_val_metrics(w, cfg)) | {n for n in w.names() if is_grad_norm(n)}
    )
    out: list[Incident] = []
    for name in sorted(watched):
        pts = w.points(name)
        if not pts or math.isfinite(pts[-1].value):
            continue
        first_bad = len(pts) - 1
        while first_bad > 0 and not math.isfinite(pts[first_bad - 1].value):
            first_bad -= 1
        onset = refine_onset(pts, first_bad, cfg)
        p = pts[-1]
        out.append(
            Incident(
                _K.NAN_INF,
                Severity.CRITICAL,
                f"{name} is {p.value} at step {p.step}",
                metric=name,
                step=p.step,
                value=p.value,
                onset_step=pts[onset].step,
                onset_wall=pts[onset].wall,
                details={"bad_points": len(pts) - first_bad, "first_bad_step": pts[first_bad].step},
            )
        )
    return out


def _is_spike(pts: Sequence[MetricPoint], i: int, cfg: DetectorConfig) -> bool:
    hist = [p.value for p in pts[max(0, i - cfg.spike_window) : i]]
    if len(hist) < cfg.spike_min_history:
        return False
    smooth = ema(hist, cfg.ema_span)[-1]
    x = pts[i].value
    return robust_z(x, hist) > cfg.spike_z and x > cfg.spike_ema_ratio * smooth


def detect_spike(w: MetricWindow, cfg: DetectorConfig) -> list[Incident]:
    out: list[Incident] = []
    for name in _train_metrics(w, cfg):
        pts = _finite(w.points(name))
        need = cfg.spike_min_history + 1
        if len(pts) < need:
            continue

        last = len(pts) - 1
        if not _is_spike(pts, last, cfg):
            continue
        lo = max(0, last - cfg.spike_escalate_within + 1)
        recent = [i for i in range(lo, last) if _is_spike(pts, i, cfg)]
        p = pts[last]
        hist = [q.value for q in pts[max(0, last - cfg.spike_window) : last]]
        severity = Severity.CRITICAL if recent else Severity.WARN
        onset = pts[refine_onset(pts, recent[0] if recent else last, cfg)]
        out.append(
            Incident(
                _K.SPIKE,
                severity,
                f"{name} spiked to {p.value:.4g} at step {p.step} (recent median {median(hist):.4g})",
                metric=name,
                step=p.step,
                value=p.value,
                onset_step=onset.step,
                onset_wall=onset.wall,
                details={"z": round(robust_z(p.value, hist), 2), "repeated": bool(recent)},
            )
        )
    return out


def detect_divergence(w: MetricWindow, cfg: DetectorConfig) -> list[Incident]:
    out: list[Incident] = []
    for name in _train_metrics(w, cfg):
        pts = _finite(w.points(name))
        if len(pts) < cfg.divergence_min_points:
            continue
        smooth = ema([p.value for p in pts], cfg.ema_span)
        best = math.inf
        best_before: list[float] = []
        for s in smooth:
            best_before.append(best)
            best = min(best, s)
        run = 0
        for i in range(len(smooth) - 1, cfg.ema_span, -1):
            ref = best_before[i]
            if math.isfinite(ref) and ref > 0 and smooth[i] > cfg.divergence_ratio * ref:
                run += 1
            else:
                break
        if run < cfg.divergence_patience:
            continue
        crossed = len(pts) - run
        onset = pts[refine_onset(pts, crossed, cfg)]
        p = pts[-1]
        ref = best_before[crossed]
        out.append(
            Incident(
                _K.DIVERGENCE,
                Severity.CRITICAL,
                f"{name} is diverging: smoothed {smooth[-1]:.4g} vs best {ref:.4g} for {run} points",
                metric=name,
                step=p.step,
                value=p.value,
                onset_step=onset.step,
                onset_wall=onset.wall,
                details={"smoothed": round(smooth[-1], 6), "best": round(ref, 6), "points_above": run},
            )
        )
    return out


def detect_plateau(w: MetricWindow, cfg: DetectorConfig) -> list[Incident]:
    out: list[Incident] = []
    for name in _train_metrics(w, cfg):
        pts = _finite(w.points(name))
        if len(pts) < 2:
            continue
        first, last = pts[0].step, pts[-1].step
        if last - first < cfg.plateau_min_steps:
            continue
        cut = last - cfg.plateau_tail_frac * (last - first)
        before = [p.value for p in pts if p.step < cut]
        tail = [p.value for p in pts if p.step >= cut]
        if len(tail) < cfg.plateau_min_tail_points or not before:
            continue
        # compare smoothed bests so a single lucky noisy point does not count as progress
        sm = ema([p.value for p in pts], cfg.ema_span)
        best_before = min(sm[: len(before)])
        best_tail = min(sm[len(before) :])
        if not math.isfinite(best_before) or best_before == 0:
            continue
        rel = (best_before - best_tail) / abs(best_before)
        if rel >= cfg.plateau_min_rel_improvement:
            continue
        p = pts[-1]
        out.append(
            Incident(
                _K.PLATEAU,
                Severity.INFO,
                f"{name} has plateaued: {rel * 100:.2f}% improvement over the last {last - int(cut)} steps",
                metric=name,
                step=p.step,
                value=p.value,
                details={"relative_improvement": round(rel, 5), "since_step": int(cut)},
            )
        )
    return out


def detect_overfit(w: MetricWindow, cfg: DetectorConfig) -> list[Incident]:
    out: list[Incident] = []
    trains = _train_metrics(w, cfg)
    if not trains:
        return out
    train_pts = _finite(w.points(trains[0]))
    if len(train_pts) < cfg.ema_span:
        return out
    t_smooth = ema([p.value for p in train_pts], cfg.ema_span)

    def train_at(step: int) -> float:
        best = None
        for p, s in zip(train_pts, t_smooth, strict=True):
            if p.step <= step:
                best = s
            else:
                break
        return best if best is not None else math.nan

    k = cfg.overfit_consecutive
    for name in _val_metrics(w, cfg):
        vals = _finite(w.points(name))
        if len(vals) < k + 1:
            continue
        tail = vals[-(k + 1) :]
        rising = all(b.value > a.value for a, b in itertools.pairwise(tail))
        if rising and tail[0].value > 0:
            rising = (tail[-1].value - tail[0].value) / tail[0].value >= cfg.overfit_min_rel_rise
        t0, t1 = train_at(tail[0].step), train_at(tail[-1].step)
        train_not_rising = (
            math.isfinite(t0) and math.isfinite(t1) and t1 <= t0 * (1 + cfg.overfit_train_tolerance)
        )
        if not (rising and train_not_rising):
            continue
        p = tail[-1]
        out.append(
            Incident(
                _K.OVERFIT,
                Severity.WARN,
                f"{name} rose for {k} evals ({tail[0].value:.4g} → {p.value:.4g}) "
                "while train loss held steady or fell",
                metric=name,
                step=p.step,
                value=p.value,
                onset_step=tail[0].step,
                onset_wall=tail[0].wall,
                details={"val_from": tail[0].value, "val_to": p.value, "train_from": t0, "train_to": t1},
            )
        )
    return out


def detect_throughput(w: MetricWindow, cfg: DetectorConfig) -> list[Incident]:
    trains = _train_metrics(w, cfg) or w.names()
    if not trains:
        return []
    pts = w.points(trains[0])
    # one observation per step, in step order
    by_step: dict[int, float] = {}
    for p in pts:
        by_step.setdefault(p.step, p.wall)
    seq = sorted(by_step.items())
    if len(seq) < cfg.throughput_min_intervals + 2:
        return []
    end_wall = seq[-1][1]
    split = next(
        (i for i, (_, wall) in enumerate(seq) if wall >= end_wall - cfg.throughput_window_s), len(seq)
    )
    if split == 0 or split >= len(seq) - 1:
        return []
    if seq[-1][1] - seq[split - 1][1] < cfg.throughput_window_s:
        return []
    rates = []
    for (s0, w0), (s1, w1) in itertools.pairwise(seq[:split]):
        if w1 > w0:
            rates.append((s1 - s0) / (w1 - w0))
    if len(rates) < cfg.throughput_min_intervals:
        return []
    baseline = median(rates)
    s0, w0 = seq[split - 1]
    s1, w1 = seq[-1]
    recent = (s1 - s0) / (w1 - w0) if w1 > w0 else 0.0
    if baseline <= 0 or recent >= cfg.throughput_drop_ratio * baseline:
        return []
    return [
        Incident(
            _K.THROUGHPUT,
            Severity.WARN,
            f"throughput fell to {recent:.3g} steps/s from a usual {baseline:.3g} steps/s",
            metric=trains[0],
            step=s1,
            details={"recent_steps_per_s": round(recent, 4), "baseline_steps_per_s": round(baseline, 4)},
        )
    ]


def detect_stall(
    w: MetricWindow, cfg: DetectorConfig, *, now: float, alive: bool, started_at: float
) -> list[Incident]:
    """The process is alive but has gone quiet for far longer than its usual logging interval."""
    if not alive:
        return []
    last = w.last_wall()
    if last is None:
        silent = now - started_at
        if silent <= cfg.startup_grace_s:
            return []
        return [
            Incident(
                _K.STALL,
                Severity.CRITICAL,
                f"no metrics at all {silent / 60:.0f} min after start",
                details={"silent_s": round(silent)},
            )
        ]
    walls = sorted({p.wall for n in w.names() for p in w.points(n)})
    gaps = [b - a for a, b in itertools.pairwise(walls) if b > a]
    usual = median(gaps) if gaps else 0.0
    threshold = max(cfg.stall_min_s, cfg.stall_interval_mult * usual)
    silent = now - last
    if silent <= threshold:
        return []
    return [
        Incident(
            _K.STALL,
            Severity.CRITICAL,
            f"no new metrics for {silent / 60:.1f} min (usually every {usual:.1f} s)",
            step=w.last_step(),
            details={"silent_s": round(silent), "usual_interval_s": round(usual, 2)},
        )
    ]


def detect_gpu_idle(
    samples: Sequence[tuple[float, float]], cfg: DetectorConfig, *, now: float, alive: bool
) -> list[Incident]:
    """``samples`` are (wall, max GPU utilisation %) readings, e.g. from nvidia-smi."""
    if not alive or not samples:
        return []
    window = [(t, u) for t, u in samples if t >= now - cfg.gpu_idle_s]
    if not window or now - min(t for t, _ in window) < cfg.gpu_idle_s * 0.9:
        return []
    if any(u >= cfg.gpu_idle_util for _, u in window):
        return []
    return [
        Incident(
            _K.GPU_IDLE,
            Severity.WARN,
            f"GPU utilisation under {cfg.gpu_idle_util:.0f}% for {cfg.gpu_idle_s / 60:.0f} min "
            "while the run is alive",
            details={"max_util": max(u for _, u in window)},
        )
    ]


# --- crash classification ---------------------------------------------------------------------------------

OOM_PATTERNS = (
    "cuda out of memory",
    "outofmemoryerror",
    "mps backend out of memory",
    "hip out of memory",
    "cudnn_status_alloc_failed",
    "cublas_status_alloc_failed",
    "resource exhausted: oom",
    "defaultcpuallocator: can't allocate memory",
    "not enough memory: you tried to allocate",
)


def classify_exit(returncode: int, log_tail: Sequence[str]) -> Incident | None:
    """Turn a finished process into an incident: OOM, other crash, or None for a clean exit."""
    if returncode == 0:
        return None
    text = "\n".join(log_tail[-80:]).lower()
    host_oom_killed = returncode in (-9, 137) and ("killed" in text or not text.strip())
    if host_oom_killed or any(pat in text for pat in OOM_PATTERNS):
        return Incident(
            _K.OOM,
            Severity.CRITICAL,
            f"out of memory (exit {returncode})",
            details={"returncode": returncode, "host_oom_killer": host_oom_killed},
        )
    last_error = next(
        (ln for ln in reversed(log_tail[-80:]) if "error" in ln.lower() or "exception" in ln.lower()), ""
    )
    return Incident(
        _K.CRASH,
        Severity.CRITICAL,
        f"process exited with code {returncode}" + (f": {last_error[:160]}" if last_error else ""),
        details={"returncode": returncode, "last_error": last_error[:400]},
    )


# --- registry ---------------------------------------------------------------------------------------------

WINDOW_DETECTORS: dict[str, Callable[[MetricWindow, DetectorConfig], list[Incident]]] = {
    _K.NAN_INF.value: detect_nan_inf,
    _K.SPIKE.value: detect_spike,
    _K.DIVERGENCE.value: detect_divergence,
    _K.PLATEAU.value: detect_plateau,
    _K.OVERFIT.value: detect_overfit,
    _K.THROUGHPUT.value: detect_throughput,
}


def run_window_detectors(w: MetricWindow, cfg: DetectorConfig) -> list[Incident]:
    """All metric-based detectors except the ones the user disabled. A NaN masks spike/divergence noise."""
    out: list[Incident] = []
    for kind, fn in WINDOW_DETECTORS.items():
        if kind in cfg.disabled:
            continue
        out.extend(fn(w, cfg))
    if any(i.kind is _K.NAN_INF for i in out):
        nan_metrics = {i.metric for i in out if i.kind is _K.NAN_INF}
        out = [i for i in out if not (i.kind in (_K.SPIKE, _K.DIVERGENCE) and i.metric in nan_metrics)]
    return out
