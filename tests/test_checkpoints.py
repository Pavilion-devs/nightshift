from __future__ import annotations

import os
from pathlib import Path

from nightshift.core.checkpoints import Checkpoint, discover, select_rollback, step_from_name


def _touch(p: Path, mtime: float) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    os.utime(p, (mtime, mtime))


def test_step_from_name() -> None:
    assert step_from_name("out/ckpt_1200.pt") == 1200
    assert step_from_name("checkpoint-1200") == 1200
    assert step_from_name("epoch=3-step=1200.ckpt") == 1200
    assert step_from_name("model_v2_step_900.safetensors") == 900
    assert step_from_name("final.pt") is None


def test_discover_finds_common_layouts_and_skips_noise(tmp_path: Path) -> None:
    _touch(tmp_path / "out" / "ckpt_100.pt", 100)
    _touch(tmp_path / "out" / "ckpt_200.pt", 200)
    (tmp_path / "runs" / "checkpoint-300").mkdir(parents=True)
    os.utime(tmp_path / "runs" / "checkpoint-300", (300, 300))
    _touch(tmp_path / ".git" / "objects" / "ckpt_999.pt", 999)
    found = discover(tmp_path)
    assert [c.step for c in found] == [100, 200, 300]


def test_discover_custom_pattern(tmp_path: Path) -> None:
    _touch(tmp_path / "out" / "ckpt_100.json", 100)
    _touch(tmp_path / "out" / "notes.json", 50)
    found = discover(tmp_path, ["out/ckpt_*.json"])
    assert [c.step for c in found] == [100]


CKPTS = [Checkpoint(f"c{s}", s, float(s)) for s in (100, 200, 300, 400, 500)]


def test_rollback_never_uses_a_checkpoint_from_after_onset() -> None:
    assert select_rollback(CKPTS, onset_step=350, onset_wall=None).step == 300
    assert select_rollback(CKPTS, onset_step=300, onset_wall=None).step == 200  # equal step is not "before"
    assert select_rollback(CKPTS, onset_step=None, onset_wall=250.0).step == 200
    # both bounds must hold: step says 400 is fine, but it was written after the anomaly began
    assert select_rollback(CKPTS, onset_step=450, onset_wall=350.0).step == 300


def test_rollback_margin_and_refusals() -> None:
    assert select_rollback(CKPTS, onset_step=450, onset_wall=None, margin_steps=100).step == 300
    assert select_rollback(CKPTS, onset_step=50, onset_wall=None) is None
    assert select_rollback(CKPTS, onset_step=None, onset_wall=None) is None
    assert select_rollback([], onset_step=500, onset_wall=None) is None


def test_rollback_with_stepless_checkpoints_uses_time_only_when_known() -> None:
    ck = [Checkpoint("last.pt", None, 100.0)]
    assert select_rollback(ck, onset_step=500, onset_wall=None) is None
    assert select_rollback(ck, onset_step=500, onset_wall=200.0).path == "last.pt"
    assert ck[0].to_dict()["path"] == "last.pt"
