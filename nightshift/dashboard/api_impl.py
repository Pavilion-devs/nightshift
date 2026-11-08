"""Synchronous implementations behind the dashboard routes. Plain functions so they are easy to test."""

from __future__ import annotations

import threading
from typing import Any

from ..adapter.runtime import data_dir
from ..core import views
from ..core.manager import Manager
from ..core.store import Store

_lock = threading.Lock()
_store: Store | None = None


def _get_store() -> Store:
    global _store
    with _lock:
        if _store is None:
            _store = Store(data_dir() / "nightshift.db")
        return _store


def _manager() -> Manager:
    # a Manager that is never started: it only writes decisions for the supervising process to pick up
    return Manager(_get_store(), data_dir=data_dir())


def status() -> dict[str, Any]:
    store = _get_store()
    st = views.status(store)
    for r in st["runs"]:
        series = (
            views.run_detail(store, r["id"], max_points=240)["series"] if r["status"] != "pending" else {}
        )
        r["series"] = series
        r["markers"] = [
            {
                "at": i["at"],
                "step": i["onset_step"] if i["onset_step"] is not None else i["step"],
                "kind": i["kind"],
                "severity": i["severity"],
                "message": i["message"],
            }
            for i in store.incidents(r["id"], limit=30)
            if i["severity"] != "info"
        ]
        actions = store.actions(r["id"], limit=30)
        r["recoveries"] = [
            {"at": a["at"], "step": _ckpt_step(a), "kind": a["kind"], "message": a["message"]}
            for a in actions
            if a["kind"] in ("rollback_restart", "oom_restart") and a["ok"]
        ]
        r["timeline"] = [
            {"at": a["at"], "kind": a["kind"], "ok": a["ok"], "message": a["message"]}
            for a in actions
            if not (a["kind"] == "notify" and a["message"].startswith(("covered by", "notify only")))
        ][:6]
        r["diagnosis"] = next(
            (i["diagnosis"] for i in store.incidents(r["id"], limit=10) if i.get("diagnosis")), None
        )
        if r["status"] == "pending":
            r["card"] = _manager().approval_card(r["id"])
    st["alerts"] = [
        {"id": i["id"], "run_id": i["run_id"], "kind": i["kind"], "message": i["message"], "at": i["at"]}
        for i in store.incidents(unacked_only=True, limit=20)
        if i["severity"] == "critical"
    ]
    return st


def _ckpt_step(action: dict[str, Any]) -> int | None:
    ck = (action.get("details") or {}).get("checkpoint") or {}
    return ck.get("step")


def run(run_id: str) -> dict[str, Any] | None:
    return views.run_detail(_get_store(), run_id)


def approve(run_id: str) -> dict[str, Any]:
    return _manager().approve(run_id)


def deny(run_id: str) -> dict[str, Any]:
    return _manager().deny(run_id)


def stop(run_id: str) -> dict[str, Any]:
    return _manager().request_stop(run_id)


def ack() -> dict[str, Any]:
    return _manager().ack()
