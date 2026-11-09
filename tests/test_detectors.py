from __future__ import annotations

import math

import pytest

from nightshift.core.detectors import (
    DetectorConfig,
    classify_exit,
    detect_divergence,
    detect_gpu_idle,
    detect_nan_inf,
    detect_overfit,
    detect_plateau,
    detect_spike,
    detect_stall,
    detect_throughput,
    ema,
    robust_z,
    run_window_detectors,
)
from nightshift.core.models import IncidentKind, MetricPoint, MetricWindow, Severity
from tests.fixtures.curves import (
    CLEAN,
    T0,
    add_series,
    exp_decay,
    window_from,
    with_divergence,
    with_nan,
    with_spike,
)

CFG = DetectorConfig()


# --- the reliability promise: clean runs stay silent ------------------------------------------------------


@pytest.mark.parametrize("shape", sorted(CLEAN))
@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("n", [60, 300])
def test_clean_curves_raise_nothing_actionable(shape: str, seed: int, n: int) -> None:
    """Across 6 shapes x 25 seeds x 2 lengths, no detector that can trigger an action fires."""
    w = window_from(CLEAN[shape](n, seed))
    incidents = [i for i in run_window_detectors(w, CFG) if i.kind is not IncidentKind.PLATEAU]
    assert incidents == [], [i.message for i in incidents]


@pytest.mark.parametrize("seed", range(10))
def test_clean_curve_with_val_loss_is_not_overfitting(seed: int) -> None:
    train = exp_decay(300, seed)
    w = window_from(train, "train/loss")
    vals = [t * 1.05 for t in exp_decay(30, seed + 100, noise=0.005, tau=4)]
    add_series(w, vals, "val/loss", every=100, dt=10)
    assert detect_overfit(w, CFG) == []


# --- nan / inf --------------------------------------------------------------------------------------------


def test_nan_fires_with_onset_at_first_bad_point() -> None:
    w = window_from(with_nan(exp_decay(100), 80))
    [inc] = detect_nan_inf(w, CFG)
    assert inc.kind is IncidentKind.NAN_INF and inc.severity is Severity.CRITICAL
    assert inc.onset_step == 810 and inc.step == 1000
    assert inc.details["bad_points"] == 20


def test_inf_grad_norm_fires() -> None:
    w = window_from(exp_decay(50))
    add_series(w, [1.0] * 49 + [math.inf], "grad_norm")
    [inc] = detect_nan_inf(w, CFG)
    assert inc.metric == "grad_norm"


def test_nan_that_recovered_is_silent() -> None:
    vals = exp_decay(100)
    vals[50] = math.nan
    assert detect_nan_inf(window_from(vals), CFG) == []


# --- spike ------------------------------------------------------------------------------------------------


def test_single_spike_warns_and_repeated_spike_is_critical() -> None:
    vals = exp_decay(100)
    w = window_from(with_spike(vals, 99))
    [inc] = detect_spike(w, CFG)
    assert inc.severity is Severity.WARN and inc.step == 1000
    w2 = window_from(with_spike(with_spike(vals, 98), 99))
    [inc2] = detect_spike(w2, CFG)
    assert inc2.severity is Severity.CRITICAL and inc2.details["repeated"]
    assert inc2.onset_step == 990


def test_spike_after_it_passes_is_silent() -> None:
    w = window_from(with_spike(exp_decay(100), 90))
    assert detect_spike(w, CFG) == []


def test_spike_needs_history() -> None:
    w = window_from(with_spike(exp_decay(10), 9))
    assert detect_spike(w, CFG) == []


def test_moderate_outlier_is_not_a_spike() -> None:
    w = window_from(with_spike(exp_decay(100), 99, factor=1.5))
    assert detect_spike(w, CFG) == []


# --- divergence -------------------------------------------------------------------------------------------


def test_divergence_fires_with_onset() -> None:
    w = window_from(with_divergence(exp_decay(200), 120))
    [inc] = detect_divergence(w, CFG)
    assert inc.kind is IncidentKind.DIVERGENCE and inc.severity is Severity.CRITICAL
    assert 1200 < inc.onset_step < 2000
    assert inc.details["points_above"] >= CFG.divergence_patience


def test_divergence_needs_patience() -> None:
    w = window_from(with_divergence(exp_decay(130), 120))  # only 10 points of growth
    assert detect_divergence(w, CFG) == []


def test_short_runs_never_diverge() -> None:
    w = window_from([5.0 + i for i in range(30)])
    assert detect_divergence(w, CFG) == []


# --- plateau ----------------------------------------------------------------------------------------------


def test_plateau_fires_on_flat_tail() -> None:
    vals = exp_decay(100, tau=10) + [1.2] * 400
    [inc] = detect_plateau(window_from(vals), CFG)
    assert inc.severity is Severity.INFO


def test_plateau_silent_while_improving() -> None:
    assert detect_plateau(window_from(exp_decay(100, tau=200)), CFG) == []


def test_plateau_needs_enough_steps() -> None:
    assert detect_plateau(window_from([1.0] * 20), CFG) == []


# --- overfit ----------------------------------------------------------------------------------------------


def test_overfit_fires_when_val_rises_and_train_falls() -> None:
    w = window_from(exp_decay(300, noise=0.0), "train/loss")
    add_series(w, [2.0, 1.8, 1.6, 1.5, 1.55, 1.6, 1.7], "val/loss", every=400, dt=40)
    [inc] = detect_overfit(w, CFG)
    assert inc.kind is IncidentKind.OVERFIT and inc.severity is Severity.WARN
    assert inc.onset_step == 1600


def test_overfit_silent_when_train_also_rises() -> None:
    w = window_from([1.0 + 0.01 * i for i in range(300)], "train/loss")
    add_series(w, [1.5, 1.55, 1.6, 1.7], "val/loss", every=700, dt=70)
    assert detect_overfit(w, CFG) == []


def test_overfit_needs_train_metric() -> None:
    w = window_from([1.5, 1.55, 1.6, 1.7], "val/loss")
    assert detect_overfit(w, CFG) == []


# --- throughput -------------------------------------------------------------------------------------------


def _timed(walls: list[float]) -> MetricWindow:
    w = MetricWindow()
    for i, t in enumerate(walls):
        w.add(MetricPoint((i + 1) * 10, "loss", 2.0, t))
    return w


def test_throughput_drop_fires() -> None:
    walls = [T0 + i * 1.0 for i in range(200)]  # 10 steps/s
    walls += [walls[-1] + (i + 1) * 5.0 for i in range(40)]  # 2 steps/s for 200 s
    [inc] = detect_throughput(_timed(walls), CFG)
    assert inc.kind is IncidentKind.THROUGHPUT
    assert inc.details["recent_steps_per_s"] < 0.5 * inc.details["baseline_steps_per_s"]


def test_steady_throughput_is_silent() -> None:
    assert detect_throughput(_timed([T0 + i * 1.0 for i in range(400)]), CFG) == []


def test_throughput_needs_a_full_recent_window() -> None:
    walls = [T0 + i * 1.0 for i in range(200)] + [T0 + 210.0]
    assert detect_throughput(_timed(walls), CFG) == []


# --- stall ------------------------------------------------------------------------------------------------


def test_stall_fires_after_long_silence() -> None:
    w = _timed([T0 + i * 2.0 for i in range(50)])
    last = T0 + 98.0
    assert detect_stall(w, CFG, now=last + 100, alive=True, started_at=T0) == []
    [inc] = detect_stall(w, CFG, now=last + 400, alive=True, started_at=T0)
    assert inc.kind is IncidentKind.STALL and inc.details["silent_s"] == 400


def test_stall_threshold_scales_with_slow_logging() -> None:
    w = _timed([T0 + i * 60.0 for i in range(20)])  # logs once a minute -> 10 min threshold
    last = T0 + 19 * 60.0
    assert detect_stall(w, CFG, now=last + 500, alive=True, started_at=T0) == []
    assert detect_stall(w, CFG, now=last + 700, alive=True, started_at=T0)


def test_stall_ignores_dead_process_and_startup() -> None:
    w = MetricWindow()
    assert detect_stall(w, CFG, now=T0 + 10_000, alive=False, started_at=T0) == []
    assert detect_stall(w, CFG, now=T0 + 600, alive=True, started_at=T0) == []
    [inc] = detect_stall(w, CFG, now=T0 + 1000, alive=True, started_at=T0)
    assert "no metrics at all" in inc.message


# --- gpu idle ---------------------------------------------------------------------------------------------


def test_gpu_idle() -> None:
    now = T0 + 1000
    idle = [(now - 300 + i * 10, 0.0) for i in range(31)]
    assert detect_gpu_idle(idle, CFG, now=now, alive=True)
    assert detect_gpu_idle(idle, CFG, now=now, alive=False) == []
    busy = [*idle[:-1], (now, 90.0)]
    assert detect_gpu_idle(busy, CFG, now=now, alive=True) == []
    assert detect_gpu_idle(idle[-5:], CFG, now=now, alive=True) == []  # not idle for long enough yet
    assert detect_gpu_idle([], CFG, now=now, alive=True) == []


# --- crash classification ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "RuntimeError: MPS backend out of memory (MPS allocated: 17.2 GB)",
        "RuntimeError: CUDNN_STATUS_ALLOC_FAILED",
    ],
)
def test_oom_is_classified(line: str) -> None:
    inc = classify_exit(1, ["Traceback (most recent call last):", line])
    assert inc is not None and inc.kind is IncidentKind.OOM


def test_host_oom_killer() -> None:
    inc = classify_exit(-9, ["Killed"])
    assert inc is not None and inc.kind is IncidentKind.OOM and inc.details["host_oom_killer"]


def test_other_crash_and_clean_exit() -> None:
    assert classify_exit(0, ["done"]) is None
    inc = classify_exit(2, ["RuntimeError: dataloader worker 3 exited unexpectedly"])
    assert inc is not None and inc.kind is IncidentKind.CRASH
    assert "dataloader" in inc.message


# --- registry & helpers -----------------------------------------------------------------------------------


def test_nan_masks_spike_and_divergence_on_same_metric() -> None:
    vals = with_divergence(exp_decay(200), 120)
    vals[-1] = math.nan
    kinds = {i.kind for i in run_window_detectors(window_from(vals), CFG)}
    assert IncidentKind.NAN_INF in kinds and IncidentKind.DIVERGENCE not in kinds


def test_disabled_detectors_do_not_run() -> None:
    cfg = DetectorConfig(disabled=["nan_inf"])
    assert run_window_detectors(window_from(with_nan(exp_decay(100), 90)), cfg) == []


def test_explicit_metric_selection() -> None:
    w = window_from(with_nan(exp_decay(100), 90), "objective")
    assert detect_nan_inf(w, CFG) == []  # not recognised as a loss by name
    cfg = DetectorConfig(train_metric="objective")
    assert detect_nan_inf(w, cfg)
    assert detect_nan_inf(w, DetectorConfig(train_metric="missing")) == []


def test_config_from_dict_ignores_unknown_keys() -> None:
    cfg = DetectorConfig.from_dict({"spike_z": 4.0, "bogus": 1})
    assert cfg.spike_z == 4.0
    assert DetectorConfig.from_dict(None) == DetectorConfig()


def test_helpers() -> None:
    assert math.isnan(ema([math.nan], 5)[0])
    assert ema([1.0, math.nan, 3.0], 1)[1] == 1.0
    assert robust_z(1.0, [1.0, 1.0, 1.0]) == 0.0
    assert robust_z(2.0, [1.0, 1.0, 1.0]) > 100


def test_spike_inside_a_batch_is_found_not_only_the_last_point() -> None:
    w = window_from(exp_decay(60))
    burst = [44.7, 206.9, 113.9, 188.5, 140.6, 119.8, 97.6, 87.8, 161.0, 98.7, 95.0, 90.0]
    w.extend(MetricPoint(610 + i * 10, "loss", v, T0 + 100) for i, v in enumerate(burst))
    [inc] = detect_spike(w, CFG)
    assert inc.severity is Severity.CRITICAL and inc.onset_step == 610 and inc.value == 206.9
