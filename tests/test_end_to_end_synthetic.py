"""The engine against a real (synthetic) training process: read its log, detect, pick a safe checkpoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from nightshift.core.checkpoints import discover, select_rollback
from nightshift.core.detectors import DetectorConfig, classify_exit, run_window_detectors
from nightshift.core.models import IncidentKind, MetricWindow
from nightshift.core.sources import JsonlSource, LogTailSource, TfEventsSource

TRAIN = Path(__file__).resolve().parents[1] / "examples" / "synthetic" / "train.py"


def _run(tmp: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TRAIN), "--out", str(tmp / "out"), *args],
        cwd=tmp,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_divergent_run_is_detected_and_rollback_predates_it(tmp_path: Path) -> None:
    proc = _run(
        tmp_path, "--lr", "3e-3", "--unstable-after", "600", "--steps", "900",
        "--jsonl", str(tmp_path / "m.jsonl"), "--tb", str(tmp_path / "tb"),
    )  # fmt: skip
    assert proc.returncode == 0
    (tmp_path / "train.log").write_text(proc.stdout)

    for src in (LogTailSource(tmp_path / "train.log"), JsonlSource(tmp_path / "m.jsonl")):
        w = MetricWindow()
        w.extend(src.poll())
        incidents = run_window_detectors(w, DetectorConfig())
        kinds = {i.kind for i in incidents}
        assert IncidentKind.NAN_INF in kinds, kinds
        nan = next(i for i in incidents if i.kind is IncidentKind.NAN_INF)
        assert 600 < nan.onset_step <= 900

    w = MetricWindow()
    w.extend(TfEventsSource(tmp_path / "tb").poll())
    assert any(i.kind is IncidentKind.NAN_INF for i in run_window_detectors(w, DetectorConfig()))

    # the loss started climbing right after step 600, ~30 steps before the first NaN: the incident's onset
    # must point at the climb, not at the NaN, or a checkpoint saved mid-climb would be restored
    w = MetricWindow()
    w.extend(LogTailSource(tmp_path / "train.log").poll())
    nan = next(i for i in run_window_detectors(w, DetectorConfig()) if i.kind is IncidentKind.NAN_INF)
    assert nan.onset_step is not None and nan.onset_step <= 620 < nan.details["first_bad_step"]
    ckpt = select_rollback(
        discover(tmp_path, ["out/ckpt_*.json"]), onset_step=nan.onset_step, onset_wall=None
    )
    assert ckpt is not None and ckpt.step is not None and ckpt.step <= 600

    # and the recovery Nightshift would make (half the lr, resume from that checkpoint) really is stable
    rec = _run(tmp_path, "--lr", "1.5e-3", "--unstable-after", "600", "--steps", "900", "--resume", ckpt.path)
    assert rec.returncode == 0
    (tmp_path / "rec.log").write_text(rec.stdout)
    w = MetricWindow()
    w.extend(LogTailSource(tmp_path / "rec.log").poll())
    assert [i for i in run_window_detectors(w, DetectorConfig()) if i.kind is not IncidentKind.PLATEAU] == []


def test_rollback_skips_poisoned_checkpoints_with_frequent_saves(tmp_path: Path) -> None:
    proc = _run(tmp_path, "--lr", "3e-3", "--unstable-after", "600", "--steps", "700", "--ckpt-every", "10")
    (tmp_path / "train.log").write_text(proc.stdout)
    w = MetricWindow()
    w.extend(LogTailSource(tmp_path / "train.log").poll())
    nan = next(i for i in run_window_detectors(w, DetectorConfig()) if i.kind is IncidentKind.NAN_INF)
    ckpt = select_rollback(
        discover(tmp_path, ["out/ckpt_*.json"]), onset_step=nan.onset_step, onset_wall=None
    )
    assert ckpt is not None and ckpt.step is not None
    assert ckpt.step <= 610, f"restored {ckpt.path}, written after the loss had started climbing"


def test_oom_run_is_classified(tmp_path: Path) -> None:
    proc = _run(tmp_path, "--batch-size", "64", "--oom-batch", "64", "--steps", "50")
    assert proc.returncode == 1
    inc = classify_exit(proc.returncode, (proc.stdout + proc.stderr).splitlines())
    assert inc is not None and inc.kind is IncidentKind.OOM
    ok = _run(tmp_path, "--batch-size", "32", "--grad-accum", "2", "--oom-batch", "64", "--steps", "50")
    assert ok.returncode == 0


def test_crash_and_overfit_modes(tmp_path: Path) -> None:
    crash = _run(tmp_path, "--crash-at", "20", "--steps", "50")
    inc = classify_exit(crash.returncode, crash.stderr.splitlines())
    assert inc is not None and inc.kind is IncidentKind.CRASH

    over = _run(tmp_path, "--steps", "3000", "--overfit-after", "1500")
    w = MetricWindow()
    (tmp_path / "o.log").write_text(over.stdout)
    w.extend(LogTailSource(tmp_path / "o.log").poll())
    assert any(i.kind is IncidentKind.OVERFIT for i in run_window_detectors(w, DetectorConfig()))
