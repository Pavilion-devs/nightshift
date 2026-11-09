"""Everything the Nightshift screen shows (dashboard tab, Desktop page and pane), as one JSON-ready dict.

Pure reads over the store, so the dashboard routes can serve it from any process.
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Any

from .models import is_loss, is_train_loss, is_val, is_val_loss
from .report import morning_report
from .store import Store
from .views import run_summary

ACTIVE = ("pending", "approved", "starting", "running", "recovering")
RECOVERIES = ("rollback_restart", "oom_restart")
_NOISE = ("covered by", "notify only")
DAY = 86_400.0


def board(
    store: Store, *, now: float | None = None, window_s: float = DAY, max_points: int = 400
) -> dict[str, Any]:
    now = time.time() if now is None else now
    since = now - window_s
    runs = store.list_runs(limit=100)
    active = [r for r in runs if r["status"] in ACTIVE]
    recent = [r for r in runs if r not in active and (r.get("ended_at") or r["created_at"]) >= since]
    shown = active + recent[: max(0, 8 - len(active))] if active or recent else runs[:3]

    cards = [_run_card(store, r, max_points=max_points) for r in shown]
    incidents = [i for i in store.incidents(since=since, limit=200) if i["severity"] != "info"]
    names = {r["id"]: r["name"] for r in runs}
    report = morning_report(store, since=since, now=now)
    attention = [i for i in incidents if i["resolved_at"] is None and i["severity"] == "critical"]
    return {
        "ok": True,
        "now": now,
        "runs": cards,
        "pending_approvals": [r["id"] for r in active if r["status"] == "pending"],
        "unacknowledged_incidents": sum(
            1 for i in store.incidents(unacked_only=True, limit=200) if i["severity"] != "info"
        ),
        "banner": {
            "incidents": len(incidents),
            "recovered": report["recoveries"],
            "attention": len(attention),
            "pending": sum(1 for r in active if r["status"] == "pending"),
            "window_h": round(window_s / 3600),
        },
        "table": [_incident_row(i, names.get(i["run_id"], i["run_id"])) for i in incidents[:20]],
        "alerts": [
            {"id": i["id"], "run_id": i["run_id"], "kind": i["kind"], "message": i["message"], "at": i["at"]}
            for i in store.incidents(unacked_only=True, limit=20)
            if i["severity"] == "critical"
        ],
        "report": {
            k: report[k] for k in ("runs", "incidents", "recoveries", "stops", "gpu_hours_rescued_estimate")
        },
    }


def _num(v: float | None) -> float | str | None:
    if v is None:
        return None
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return round(v, 6)


def _series(store: Store, run_id: str, name: str, max_points: int) -> list[dict[str, Any]]:
    return [
        {"step": p["step"], "value": _num(p["value"]), "attempt": p["attempt"]}
        for p in store.series(run_id, name, max_points)
        if p["step"] is not None
    ]


def _run_card(store: Store, run: dict[str, Any], *, max_points: int) -> dict[str, Any]:
    rid = run["id"]
    card = run_summary(store, run)
    names = store.metric_names(rid)
    train = next((n for n in names if is_train_loss(n)), None)
    val = next((n for n in names if is_val_loss(n)), None)
    lr = next((n for n in names if n in ("lr", "learning_rate") or n.endswith("/lr")), None)
    card["series"] = {
        "train": _series(store, rid, train, max_points) if train else [],
        "val": _series(store, rid, val, max_points // 2) if val else [],
        "lr": _series(store, rid, lr, max_points // 2) if lr else [],
    }
    card["series_names"] = {"train": train, "val": val, "lr": lr}
    incidents = store.incidents(rid, limit=50)
    actions = store.actions(rid, limit=50)
    card["incidents"] = [_incident_row(i, run["name"]) for i in incidents if i["severity"] != "info"]
    card["markers"] = [
        {"step": i["onset_step"] if i["onset_step"] is not None else i["step"], "kind": i["kind"],
         "severity": i["severity"]}
        for i in incidents
        if i["severity"] == "critical" and (i["onset_step"] is not None or i["step"] is not None)
    ]  # fmt: skip
    card["recoveries"] = [
        {"step": ((a["details"] or {}).get("checkpoint") or {}).get("step"), "kind": a["kind"],
         "message": a["message"], "at": a["at"]}
        for a in actions
        if a["kind"] in RECOVERIES and a["ok"]
    ]  # fmt: skip
    card["did"] = [
        {"kind": a["kind"], "ok": bool(a["ok"]), "message": a["message"], "at": a["at"]}
        for a in reversed(actions)
        if not (a["kind"] == "notify" and a["message"].startswith(_NOISE))
    ][-6:]
    critical = next((i for i in incidents if i["severity"] == "critical"), None)
    card["signals"] = signals(store, rid, critical, names) if critical else []
    card["diagnosis"] = next((i["diagnosis"] for i in incidents if i.get("diagnosis")), None)
    card["summary"] = _summary(run, critical, card["signals"], actions)
    card["log"] = run_log(run, incidents, actions)
    card["playbook"] = {
        "max_interventions": (run.get("playbook") or {}).get("max_interventions", 3),
        "used": run.get("interventions") or 0,
    }
    return card


def signals(store: Store, run_id: str, inc: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    """How far each training metric moved at the incident, against its own recent median.

    Measured on the points logged before the incident was raised (so a later, recovered attempt does not
    blur it), between the anomaly's onset and the detection step."""
    onset = inc["onset_step"] if inc["onset_step"] is not None else inc["step"]
    if onset is None:
        return []
    end = inc["step"] if inc["step"] is not None else onset
    out = []
    for name in names:
        if is_val(name) or name in ("batch_size", "grad_accum", "step", "epoch"):
            continue
        pts = [
            p
            for p in store.series(run_id, name, 5000)
            if p["step"] is not None and p["wall"] <= inc["at"] + 1
        ]
        before = [p["value"] for p in pts if p["step"] < onset and _finite(p["value"])][-20:]
        during = [p["value"] for p in pts if onset <= p["step"] <= end]
        if len(before) < 3 or not during:
            continue
        base = statistics.median(before)
        if not _finite(base) or base == 0:
            continue
        if any(not _finite(v) for v in during):
            out.append({"metric": name, "ratio": None, "text": "NaN", "base": _num(base), "peak": "nan"})
            continue
        peak = max(during, key=lambda v: abs(v / base))
        ratio = abs(peak / base)
        if ratio < 1.5:
            continue
        out.append({"metric": name, "ratio": round(ratio, 2), "text": f"{ratio:.0f}×" if ratio >= 10
                    else f"{ratio:.1f}×", "base": _num(base), "peak": _num(peak)})  # fmt: skip
    out.sort(key=lambda s: (s["ratio"] is not None, -(s["ratio"] or 0)))
    top = out[:3]
    hi = max((math.log10(s["ratio"]) for s in top if s["ratio"]), default=1.0) or 1.0
    for s in top:
        s["frac"] = 1.0 if s["ratio"] is None else max(0.04, min(1.0, math.log10(s["ratio"]) / hi))
    return top


def _finite(v: float | None) -> bool:
    return v is not None and math.isfinite(v)


def _fmt(v: Any) -> str:
    if isinstance(v, str):
        return v
    if v is None:
        return "?"
    return f"{v:.3g}"


def _summary(
    run: dict[str, Any], inc: dict[str, Any] | None, sig: list[dict[str, Any]], actions: list
) -> str:
    """A plain-language account of the latest critical incident, from Nightshift's own records (no LLM)."""
    if inc is None:
        if run["status"] == "pending":
            return "Waiting for your approval. Nothing has started."
        return "No critical incidents. Nightshift stays silent while the run is healthy."
    parts = []
    loss = next((s for s in sig if is_loss(s["metric"])), None)
    if inc["kind"] == "oom":
        parts.append("The process ran out of memory.")
    elif loss and loss["ratio"]:
        parts.append(f"{loss['metric']} jumped to {_fmt(loss['peak'])}, {loss['text']} its recent median "
                     f"of {_fmt(loss['base'])}, at step {inc['onset_step'] or inc['step']}.")  # fmt: skip
    else:
        parts.append(inc["message"][0].upper() + inc["message"][1:] + ".")
    others = [s for s in sig if s is not loss and s["ratio"]]
    if others:
        parts.append(" ".join(f"{s['metric']} rose {s['text']} at the same time." for s in others[:2]))
    fix = next(
        (a for a in actions if a["kind"] in RECOVERIES and a["ok"] and a["incident_id"] == inc["id"]), None
    )
    if fix:
        parts.append(f"Nightshift {fix['message']}.")
    return " ".join(parts)


def _incident_row(i: dict[str, Any], run_name: str) -> dict[str, Any]:
    if i["resolved_at"] is not None:
        state = "recovered"
    elif not i["acked"]:
        state = "new"
    else:
        state = "investigating"
    return {
        "id": i["id"],
        "run_id": i["run_id"],
        "run": run_name,
        "kind": i["kind"],
        "severity": i["severity"],
        "metric": i["metric"] or ("process" if i["kind"] in ("oom", "crash", "stall") else i["kind"]),
        "message": i["message"],
        "step": i["step"],
        "onset_step": i["onset_step"],
        "at": i["at"],
        "resolved_at": i["resolved_at"],
        "state": state,
    }


def run_log(run: dict[str, Any], incidents: list, actions: list) -> list[dict[str, Any]]:
    """The run's story in time order: launch, incidents, what Nightshift did, the end."""
    ev: list[dict[str, Any]] = []
    params = dict(run.get("params") or {})
    for a in sorted(actions, key=lambda a: -a["at"]):  # undo each recovery's changes, newest first
        for k, change in ((a["details"] or {}).get("changes") or {}).items():
            params[k] = change[0]
    if run.get("started_at"):
        detail = ", ".join(f"{k} {v}" for k, v in list(params.items())[:3]) or "approved command"
        ev.append({"at": run["started_at"], "event": "Run launched", "status": "success", "detail": detail})
    latest_info: dict[str, int] = {}
    for i in incidents:  # newest first: keep one note per informational kind
        if i["severity"] == "info":
            latest_info.setdefault(i["kind"], i["id"])
    for i in incidents:
        if i["severity"] == "info" and latest_info.get(i["kind"]) != i["id"]:
            continue
        sev = {"critical": "failed", "warn": "warning"}.get(i["severity"], "info")
        step = f"step {i['step']}" if i["step"] is not None else ""
        ev.append({"at": i["at"], "event": _title(i), "status": sev, "detail": (i["message"] + (
            f" ({step})" if step and step not in i["message"] else ""))})  # fmt: skip
    for a in actions:
        if a["kind"] == "notify" and a["message"].startswith(_NOISE):
            continue
        status = "warning" if a["kind"] in RECOVERIES and a["ok"] else "failed" if not a["ok"] else "info"
        title = {"rollback_restart": "Rolled back", "oom_restart": "Restarted smaller", "stop": "Run stopped",
                 "notify": "You were alerted"}.get(a["kind"], a["kind"])  # fmt: skip
        ev.append({"at": a["at"], "event": title, "status": status, "detail": a["message"]})
    if run.get("ended_at"):
        ok = run["status"] == "finished"
        ev.append({"at": run["ended_at"], "event": "Run finished" if ok else f"Run {run['status']}",
                   "status": "success" if ok else "failed", "detail": run.get("note") or ""})  # fmt: skip
    ev.sort(key=lambda e: e["at"])
    return ev[-12:]


_TITLES = {
    "nan_inf": "NaN in the loss",
    "spike": "Loss spike",
    "divergence": "Loss diverging",
    "oom": "Out of memory",
    "crash": "Process crashed",
    "stall": "Run stalled",
    "overfit": "Overfitting",
    "plateau": "Loss plateau",
    "throughput": "Throughput dropped",
    "gpu_idle": "GPU idle",
}


def _title(i: dict[str, Any]) -> str:
    return _TITLES.get(i["kind"], i["kind"].replace("_", " ").capitalize())
