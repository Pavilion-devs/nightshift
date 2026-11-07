"""Words and pictures: incident alerts, status alerts, the morning report and the run chart."""

from __future__ import annotations

import math
import time
from typing import Any

from .alerts import Alert
from .chart_png import Series, render
from .models import Incident, is_train_loss, is_val_loss
from .playbook import NOTIFY, OOM_RESTART, ROLLBACK_RESTART, STOP, Decision
from .store import Store

_EMOJI = {"nan_inf": "🔥", "divergence": "📈", "spike": "⚡", "oom": "💥", "crash": "💀", "stall": "🧊",
          "overfit": "🔀", "plateau": "➖", "throughput": "🐢", "gpu_idle": "💤"}  # fmt: skip
_LABEL = {"nan_inf": "NaN", "divergence": "divergence", "spike": "loss spike", "oom": "out of memory",
          "crash": "crash", "stall": "stall", "overfit": "overfitting", "plateau": "plateau",
          "throughput": "slowdown", "gpu_idle": "idle GPU"}  # fmt: skip


def alert_for_incident(
    run: dict[str, Any],
    inc: Incident,
    d: Decision,
    result: str,
    ok: bool,
    diagnosis: str | None,
    chart: bytes | None,
) -> Alert:
    kind = inc.kind.value
    where = f" at step {inc.step}" if inc.step is not None else ""
    if d.action in (ROLLBACK_RESTART, OOM_RESTART) and ok:
        outcome = f"✅ {result}"
    elif d.action == STOP:
        outcome = f"🛑 {result}"
    elif d.action == NOTIFY:
        outcome = "👀 no automatic action (notify only)"
    else:
        outcome = f"⚠️ {result}"
    title = f"{_EMOJI.get(kind, '🌙')} {run['name']}: {_LABEL.get(kind, kind)}{where}"
    lines = [inc.message, outcome]
    budget = f"interventions used {run.get('interventions') or 0}"
    if d.action != NOTIFY:
        lines.append(budget)
    if diagnosis:
        lines.append(f"Why: {diagnosis}")
    return Alert(
        run_id=run["id"],
        run_name=run["name"],
        title=title,
        body="\n".join(lines),
        page=d.page,
        kind=kind,
        chart_png=chart,
        details={"incident": inc.to_dict(), "decision": d.to_dict(), "result": result, "ok": ok},
    )


def alert_for_status(store: Store, run: dict[str, Any], status: str, note: str) -> Alert:
    latest = store.latest_metrics(run["id"])
    loss = next((v for k, v in latest.items() if is_train_loss(k)), None)
    tail = (
        f" Final loss {loss['value']:.4g} at step {loss['step']}."
        if loss and math.isfinite(loss["value"])
        else ""
    )
    icon = {"finished": "🏁", "stopped": "🛑", "failed": "❌"}.get(status, "🌙")
    return Alert(
        run_id=run["id"],
        run_name=run["name"],
        title=f"{icon} {run['name']}: {status}",
        body=f"{note}.{tail}",
        page=status != "finished",
        kind=status,
    )


def chart_for_run(store: Store, run: dict[str, Any], width: int = 800, height: int = 400) -> bytes:
    names = store.metric_names(run["id"])
    train = [n for n in names if is_train_loss(n)][:1]
    val = [n for n in names if is_val_loss(n)][:1]
    series: list[Series] = []
    for i, name in enumerate(train + val):
        pts = store.series(run["id"], name, max_points=width)
        series.append(
            Series(
                points=[(p["wall"], p["value"]) for p in pts],
                attempts=[p["attempt"] for p in pts] if i == 0 else [],
                color=None if i == 0 else (158, 206, 106),
            )
        )
    incidents = [i for i in store.incidents(run["id"], limit=50) if i["severity"] != "info"]
    actions = store.actions(run["id"], limit=50)
    finite = [y for s in series for _, y in s.points if math.isfinite(y) and y > 0]
    log_y = bool(finite) and max(finite) / min(finite) > 50
    return render(
        series,
        width=width,
        height=height,
        incidents_x=[i["at"] for i in incidents],
        recoveries_x=[a["at"] for a in actions if a["kind"] in (ROLLBACK_RESTART, OOM_RESTART) and a["ok"]],
        log_y=log_y,
    )


def morning_report(
    store: Store, since: float, now: float | None = None, gpus_per_run: float = 1.0
) -> dict[str, Any]:
    """What happened overnight, as data and as text.

    "GPU-hours rescued" is labelled as an estimate: for each automatic recovery, how long the recovered attempt
    then trained (until the run ended, the next recovery, or now). Without Nightshift that time would have been
    spent on a broken run, or not at all.
    """
    now = time.time() if now is None else now
    runs = [r for r in store.list_runs(limit=200) if (r.get("ended_at") or now) >= since]
    incidents = store.incidents(since=since, limit=500)
    actions = store.actions(since=since, limit=500)
    recoveries = [a for a in actions if a["kind"] in (ROLLBACK_RESTART, OOM_RESTART) and a["ok"]]
    stops = [a for a in actions if a["kind"] == STOP]
    ends = {r["id"]: r.get("ended_at") or now for r in runs}
    saved_h = 0.0
    for a in recoveries:
        later = [b["at"] for b in recoveries if b["run_id"] == a["run_id"] and b["at"] > a["at"]]
        end = min([ends.get(a["run_id"], now), *later])
        saved_h += max(0.0, end - a["at"]) / 3600 * gpus_per_run
    serious = [i for i in incidents if i["severity"] != "info"]
    lines = [
        f"🌅 Nightshift report: {len(runs)} run(s), {len(serious)} incident(s), "
        f"{len(recoveries)} recovered automatically, {len(stops)} stopped."
    ]
    if recoveries:
        lines.append(
            f"≈{saved_h:.1f} GPU-hours of training rescued (estimate: time each recovered attempt ran)."
        )
    for r in runs:
        latest = store.latest_metrics(r["id"])
        loss = next((v for k, v in latest.items() if is_train_loss(k)), None)
        loss_txt = (
            f", loss {loss['value']:.4g} @ step {loss['step']}"
            if loss and math.isfinite(loss["value"])
            else ""
        )
        lines.append(f"• {r['name']}: {r['status']}, {r.get('interventions') or 0} intervention(s){loss_txt}")
    for a in sorted(actions, key=lambda a: a["at"]):
        if a["kind"] == NOTIFY:
            continue
        stamp = time.strftime("%H:%M", time.localtime(a["at"]))
        lines.append(f"  {stamp} {a['kind']}: {a['message']}")
    return {
        "runs": len(runs),
        "incidents": len(serious),
        "recoveries": len(recoveries),
        "stops": len(stops),
        "gpu_hours_rescued_estimate": round(saved_h, 2),
        "text": "\n".join(lines),
    }
