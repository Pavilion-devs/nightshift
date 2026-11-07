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
    st = views.status(_get_store())
    store = _get_store()
    for r in st["runs"]:
        series = (
            views.run_detail(store, r["id"], max_points=240)["series"] if r["status"] != "pending" else {}
        )
        r["series"] = series
        r["markers"] = [
            {"at": i["at"], "kind": i["kind"], "severity": i["severity"]}
            for i in store.incidents(r["id"], limit=30)
            if i["severity"] != "info"
        ]
        r["recoveries"] = [
            {"at": a["at"], "kind": a["kind"], "message": a["message"]}
            for a in store.actions(r["id"], limit=30)
            if a["kind"] in ("rollback_restart", "oom_restart") and a["ok"]
        ]
        if r["status"] == "pending":
            r["card"] = _manager().approval_card(r["id"])
    return st


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
