"""Find a run's checkpoints and pick the one to roll back to.

The one rule that matters: never restore a checkpoint written at or after the moment the anomaly began. A
checkpoint saved while the loss was already exploding would carry the corruption straight into the restart.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATTERNS = (
    "**/ckpt*",
    "**/checkpoint*",
    "**/*.ckpt",
    "**/*.pt",
    "**/*.pth",
    "**/*.safetensors",
)
_INT = re.compile(r"(\d+)")
_SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", "wandb"}


@dataclass(frozen=True)
class Checkpoint:
    path: str
    step: int | None
    mtime: float

    def to_dict(self) -> dict:
        return {"path": self.path, "step": self.step, "mtime": self.mtime}


def step_from_name(path: str) -> int | None:
    """The last integer in the file or folder name: ckpt_1200.pt, checkpoint-1200, epoch=3-step=1200.ckpt."""
    name = Path(path).name
    stem = name.split(".")[0] if not name.startswith(".") else name
    m = re.search(r"step[=_-]?(\d+)", name, re.I)
    if m:
        return int(m.group(1))
    nums = _INT.findall(stem)
    return int(nums[-1]) if nums else None


def discover(
    root: str | os.PathLike[str],
    patterns: tuple[str, ...] | list[str] = DEFAULT_PATTERNS,
    max_items: int = 2000,
) -> list[Checkpoint]:
    """Checkpoints under ``root`` matching any glob pattern, oldest first."""
    root = Path(root)
    seen: dict[str, Checkpoint] = {}
    for pattern in patterns:
        full = pattern if os.path.isabs(pattern) else str(root / pattern)
        for match in glob.iglob(full, recursive=True):
            parts = set(Path(match).parts)
            if parts & _SKIP_DIRS:
                continue
            try:
                mtime = os.stat(match).st_mtime
            except OSError:
                continue
            real = os.path.realpath(match)
            if real not in seen:
                seen[real] = Checkpoint(path=os.path.abspath(match), step=step_from_name(match), mtime=mtime)
            if len(seen) >= max_items:
                break
    return sorted(seen.values(), key=lambda c: (c.mtime, c.step or -1))


def select_rollback(
    checkpoints: list[Checkpoint],
    *,
    onset_step: int | None,
    onset_wall: float | None,
    margin_steps: int = 0,
) -> Checkpoint | None:
    """Newest checkpoint that is provably from before the anomaly began.

    A checkpoint qualifies only if it was written before ``onset_wall`` (when known) AND its step is below
    ``onset_step - margin_steps`` (when both are known). With neither bound known we refuse to guess.
    """
    if onset_step is None and onset_wall is None:
        return None
    ok: list[Checkpoint] = []
    for c in checkpoints:
        if onset_wall is not None and c.mtime >= onset_wall:
            continue
        if onset_step is not None:
            if c.step is None:
                if onset_wall is None:
                    continue  # no way to place it in time
            elif c.step >= onset_step - margin_steps:
                continue
        ok.append(c)
    if not ok:
        return None
    return max(ok, key=lambda c: (c.step if c.step is not None else -1, c.mtime))
