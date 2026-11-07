"""Nightshift backend routes, mounted at /api/plugins/nightshift/.

Serves both the web dashboard tab and the Desktop pane (Desktop's ``ctx.rest`` calls these routes).
These routes may run in a different process from the supervisor threads, so they only ever read
Nightshift's SQLite store, never in-memory state.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()

VERSION = "0.1.0"


@router.get("/status")
async def status() -> dict:
    return {"ok": True, "version": VERSION, "runs": []}
