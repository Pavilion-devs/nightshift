"""Read-only views over the store: what the tools, the slash command and the Desktop pane show.

Pure reads, so they work from any process (the dashboard routes may not share memory with the supervisors).
"""

from __future__ import annotations

import math
import time
from typing import Any

from .models import is_train_loss, is_val_loss
from .store import Store


def _num(v: float) -> float | str:
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return round(v, 6)


def run_summary(store: Store, run: dict[str, Any], now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    latest = store.latest_metrics(run["id"])
    train = next(((k, v) for k, v in latest.items() if is_train_loss(k)), None)
    val = next(((k, v) for k, v in latest.items() if is_val_loss(k)), None)
    open_incidents = [i for i in store.incidents(run["id"], limit=20) if i["resolved_at"] is None]
    last_action = next(iter(store.actions(run["id"], limit=1)), None)
    started = run.get("started_at")
    health = "healthy"
    if run["status"] in ("stopped", "failed"):
        health = run["status"]
    elif any(i["severity"] == "critical" for i in open_incidents) or run["status"] == "recovering":
        health = "critical"
    elif any(i["severity"] == "warn" for i in open_incidents):
        health = "warning"
    return {
        "id": run["id"],
        "name": run["name"],
        "mode": run["mode"],
        "status": run["status"],
        "health": health,
        "attempt": run.get("attempt") or 0,
        "interventions": run.get("interventions") or 0,
        "params": run.get("params") or {},
        "note": run.get("note"),
        "uptime_s": round((run.get("ended_at") or now) - started) if started else None,
        "step": train[1]["step"] if train else None,
        "train_loss": {"name": train[0], "value": _num(train[1]["value"])} if train else None,
        "val_loss": {"name": val[0], "value": _num(val[1]["value"])} if val else None,
        "open_incidents": [
            {"id": i["id"], "kind": i["kind"], "severity": i["severity"], "message": i["message"]}
            for i in open_incidents
        ],
        "last_action": {
            "kind": last_action["kind"],
            "message": last_action["message"],
            "at": last_action["at"],
        }
        if last_action
        else None,
    }


def status(store: Store, *, include_finished: int = 5) -> dict[str, Any]:
    runs = store.list_runs(limit=100)
    active = [r for r in runs if r["status"] in ("pending", "approved", "starting", "running", "recovering")]
    done = [r for r in runs if r not in active][:include_finished]
    unacked = store.incidents(unacked_only=True, limit=50)
    return {
        "ok": True,
        "runs": [run_summary(store, r) for r in active + done],
        "pending_approvals": [r["id"] for r in active if r["status"] == "pending"],
        "unacknowledged_incidents": len(unacked),
    }


def run_detail(store: Store, run_id: str, max_points: int = 400) -> dict[str, Any] | None:
    run = store.get_run(run_id)
    if run is None:
        return None
    names = store.metric_names(run_id)
    charted = [n for n in names if is_train_loss(n)][:1] + [n for n in names if is_val_loss(n)][:1]
    series = {
        n: [
            {"step": p["step"], "value": _num(p["value"]), "wall": p["wall"], "attempt": p["attempt"]}
            for p in store.series(run_id, n, max_points)
        ]
        for n in charted
    }
    return {
        **run_summary(store, run),
        "command": run.get("argv"),
        "cwd": run.get("cwd"),
        "series": series,
        "incidents": store.incidents(run_id, limit=50),
        "actions": store.actions(run_id, limit=50),
    }
