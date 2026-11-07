"""Hermes wiring. The only module family in Nightshift that touches Hermes APIs."""

from __future__ import annotations

import json
import logging
from typing import Any

from .. import __version__

log = logging.getLogger("nightshift")

STATUS_SCHEMA: dict[str, Any] = {
    "name": "nightshift_status",
    "description": (
        "Show the ML training runs Nightshift is watching: health, latest metrics, open incidents "
        "and actions taken. Read-only."
    ),
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}


def _status_handler(args: dict, **kwargs: Any) -> str:
    try:
        return json.dumps({"ok": True, "version": __version__, "runs": []})
    except Exception as exc:  # tool handlers never raise
        log.exception("nightshift_status failed")
        return json.dumps({"ok": False, "error": str(exc)})


def install(ctx: Any) -> None:
    """Register every Nightshift surface. Must match plugin.yaml's provides_* lists exactly."""
    ctx.register_tool(
        name="nightshift_status",
        toolset="nightshift",
        schema=STATUS_SCHEMA,
        handler=_status_handler,
        description=STATUS_SCHEMA["description"],
        emoji="🌙",
    )
    log.info("nightshift %s registered", __version__)
