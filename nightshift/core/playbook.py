"""What to do about an incident. Pure decision logic; the supervisor carries the decision out.

The user approves a playbook once, at launch. Everything Nightshift may do on its own is in it: which incidents
trigger which action, how many automatic interventions are allowed, the cooldown, and exactly which parameters
may change and how (``lr *= 0.5`` floored at ``lr_min``; ``batch_size //= 2`` with ``grad_accum *= 2``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

from .models import Incident, IncidentKind, Severity

NOTIFY = "notify"
STOP = "stop"
ROLLBACK_RESTART = "rollback_restart"
OOM_RESTART = "oom_restart"
ACTIONS = (NOTIFY, STOP, ROLLBACK_RESTART, OOM_RESTART)
RESTARTS = (ROLLBACK_RESTART, OOM_RESTART)

DEFAULT_RULES: dict[str, str] = {
    "nan_inf": ROLLBACK_RESTART,
    "divergence": ROLLBACK_RESTART,
    "spike:critical": ROLLBACK_RESTART,
    "spike:warn": NOTIFY,
    "oom": OOM_RESTART,
    "crash": NOTIFY,
    "stall": NOTIFY,
    "overfit": NOTIFY,
    "plateau": NOTIFY,
    "throughput": NOTIFY,
    "gpu_idle": NOTIFY,
}

# notifications for these show up in the pane and the morning report, but never buzz a phone
QUIET_KINDS = ("plateau",)


@dataclass
class Playbook:
    rules: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_RULES))
    max_interventions: int = 3
    cooldown_s: float = 600.0
    lr_param: str = "lr"
    lr_factor: float = 0.5
    lr_min: float = 1e-7
    batch_param: str = "batch_size"
    accum_param: str = "grad_accum"
    rollback_margin_steps: int = 0
    checkpoint_patterns: list[str] = field(default_factory=list)  # empty = common layouts under cwd
    page_on: list[str] = field(default_factory=lambda: ["critical"])  # warnings: pane + morning report

    def __post_init__(self) -> None:
        bad = {k: v for k, v in self.rules.items() if v not in ACTIONS}
        if bad:
            raise ValueError(f"unknown playbook action(s): {bad}")
        if not 0 < self.lr_factor < 1:
            raise ValueError("lr_factor must be between 0 and 1")
        if self.max_interventions < 0:
            raise ValueError("max_interventions must be >= 0")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Playbook:
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        rules = dict(DEFAULT_RULES)
        rules.update(data.pop("rules", None) or {})
        return cls(rules=rules, **{k: v for k, v in data.items() if k in known and k != "rules"})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def action_for(self, inc: Incident) -> str:
        return self.rules.get(f"{inc.kind.value}:{inc.severity.value}") or self.rules.get(
            inc.kind.value, NOTIFY
        )

    def describe(self, params: dict[str, Any]) -> list[str]:
        """Plain-language lines for the approval card: everything the user is agreeing to."""
        auto = sorted({k.split(":")[0] for k, v in self.rules.items() if v in RESTARTS})
        lines = [
            f"Automatic recovery on: {', '.join(auto) or 'nothing (notify only)'}",
            f"At most {self.max_interventions} automatic interventions, {self.cooldown_s / 60:.0f} min apart",
        ]
        if self.lr_param in params:
            lines.append(
                f"Rollback lowers {self.lr_param} ×{self.lr_factor:g} each time (floor {self.lr_min:g})"
            )
        if self.batch_param in params:
            accum = f", {self.accum_param} ×2" if self.accum_param in params else ""
            lines.append(f"Out-of-memory halves {self.batch_param}{accum}")
        lines.append(
            "If recovery is not possible or the budget runs out, the run is stopped and you are paged"
        )
        return lines


@dataclass
class RunState:
    mode: str  # "launch" | "watch"
    params: dict[str, Any]
    interventions: int = 0
    last_intervention_at: float | None = None
    can_resume: bool = True  # the approved command has a {resume} slot


@dataclass
class Decision:
    action: str
    reason: str
    params: dict[str, Any] = field(default_factory=dict)  # params for the restart, when restarting
    page: bool = False
    changes: dict[str, tuple[Any, Any]] = field(default_factory=dict)  # name -> (old, new)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "params": self.params,
            "page": self.page,
            "changes": {k: list(v) for k, v in self.changes.items()},
        }


def decide(pb: Playbook, inc: Incident, state: RunState, now: float) -> Decision:
    action = pb.action_for(inc)
    page = inc.severity.value in pb.page_on and inc.kind.value not in QUIET_KINDS

    if action == NOTIFY:
        return Decision(NOTIFY, "notify only, per playbook", page=page)
    if state.mode != "launch":
        return Decision(NOTIFY, f"would {action}, but this run is watched read-only", page=page)
    if action == STOP:
        return Decision(STOP, "stop, per playbook", page=True)

    # a restart was asked for: check the guard rails, in order
    if state.interventions >= pb.max_interventions:
        return Decision(
            STOP, f"automatic budget used up ({state.interventions}/{pb.max_interventions})", page=True
        )
    if state.last_intervention_at is not None and now - state.last_intervention_at < pb.cooldown_s:
        mins = (now - state.last_intervention_at) / 60
        return Decision(STOP, f"failed again {mins:.0f} min after the last recovery; stopping", page=True)

    params = dict(state.params)
    changes: dict[str, tuple[Any, Any]] = {}
    if action == ROLLBACK_RESTART:
        if not state.can_resume:
            return Decision(STOP, "cannot roll back: the approved command has no {resume} slot", page=True)
        if pb.lr_param in params:
            old = float(params[pb.lr_param])
            new = max(pb.lr_min, old * pb.lr_factor)
            if new >= old:
                return Decision(STOP, f"{pb.lr_param} is already at its floor ({old:g})", page=True)
            params[pb.lr_param] = new
            changes[pb.lr_param] = (old, new)
        return Decision(ROLLBACK_RESTART, "roll back to the last healthy checkpoint", params, page, changes)

    # OOM_RESTART
    if inc.kind is not IncidentKind.OOM:
        return Decision(NOTIFY, f"{action} only applies to out-of-memory", page=page)
    if pb.batch_param not in params:
        return Decision(STOP, f"cannot shrink memory: no {pb.batch_param} in the approved command", page=True)
    old_b = int(params[pb.batch_param])
    if old_b <= 1:
        return Decision(STOP, f"{pb.batch_param} is already 1", page=True)
    params[pb.batch_param] = old_b // 2
    changes[pb.batch_param] = (old_b, old_b // 2)
    if pb.accum_param in params:
        old_a = int(params[pb.accum_param])
        params[pb.accum_param] = old_a * 2
        changes[pb.accum_param] = (old_a, old_a * 2)
    return Decision(OOM_RESTART, "restart with half the batch size", params, page or True, changes)


def is_actionable(inc: Incident) -> bool:
    return inc.severity is not Severity.INFO
