"""Nightshift backend routes, mounted at /api/plugins/nightshift/.

Serves both the web dashboard tab and the Desktop pane (Desktop's ``ctx.rest`` calls these routes).
These routes may run in a different process from the supervisor threads, so they only ever read
Nightshift's SQLite store, never in-memory state. Approve/stop/ack are written to the same store; the
process that supervises runs picks them up within a second.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()

_PKG = "_nightshift_dashboard"


def _engine() -> Any:
    """Load the Nightshift package by path (this file is imported standalone, without a package)."""
    if _PKG not in sys.modules:
        root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            _PKG, root / "__init__.py", submodule_search_locations=[str(root)]
        )
        if spec is None or spec.loader is None:
            raise ImportError("cannot load nightshift")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_PKG] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{_PKG}.dashboard.api_impl")


@router.get("/status")
async def status() -> dict:
    return await asyncio.to_thread(_engine().status)


@router.get("/run/{run_id}")
async def run(run_id: str) -> dict:
    detail = await asyncio.to_thread(_engine().run, run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="no such run")
    return detail


@router.post("/approve/{run_id}")
async def approve(run_id: str) -> dict:
    return await asyncio.to_thread(_engine().approve, run_id)


@router.post("/deny/{run_id}")
async def deny(run_id: str) -> dict:
    return await asyncio.to_thread(_engine().deny, run_id)


@router.post("/stop/{run_id}")
async def stop(run_id: str) -> dict:
    return await asyncio.to_thread(_engine().stop, run_id)


@router.post("/ack")
async def ack() -> dict:
    return await asyncio.to_thread(_engine().ack)
