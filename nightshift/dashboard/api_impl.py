"""Synchronous implementations behind the dashboard routes. Plain functions so they are easy to test."""

from __future__ import annotations

import threading
from typing import Any

from ..adapter.runtime import data_dir
from ..core import board, views
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
    st = board.board(_get_store())
    for r in st["runs"]:
        if r["status"] == "pending":
            r["card"] = _manager().approval_card(r["id"])
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
