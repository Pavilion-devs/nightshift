"""Hermes wiring. Everything registered here must match plugin.yaml's provides_* lists exactly."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .. import __version__
from . import schemas
from .runtime import reset_runtime, start_soon
from .tools import Handlers, make_cli_handler, setup_cli

log = logging.getLogger("nightshift")

SKILL_DIR = Path(__file__).resolve().parent.parent / "skills" / "nightshift"

MORNING_REPORT = {
    "title": "Nightshift morning report",
    "description": "Every morning: what happened to your training runs overnight, incidents, automatic "
    "recoveries and GPU-hours rescued.",
    "category": "daily",
    "schedule_template": "{minute} {hour} * * *",
    "prompt_template": (
        "Call the nightshift_report tool with hours=14 and send its `text` field to the user exactly as "
        "written. Then add at most one line on anything that still needs the user's attention (a stopped or "
        "failed run, a pending approval). If there were no runs, say so in one line."
    ),
    "slots": [
        {"name": "time", "type": "time", "label": "What time?", "default": "08:00", "help": "24h local time"},
        {
            "name": "deliver",
            "type": "enum",
            "label": "Where to deliver?",
            "default": "origin",
            "options": ["origin", "local", "telegram", "discord", "email"],
            "optional": False,
            "strict": False,
            "help": "origin = the chat you set this up from",
        },
    ],
    "tags": ["ml", "training", "daily"],
}


def install(ctx: Any) -> Handlers:
    h = Handlers(ctx)
    handlers = h.tool_handlers()
    for schema in schemas.ALL:
        ctx.register_tool(
            name=schema["name"],
            toolset="nightshift",
            schema=schema,
            handler=handlers[schema["name"]],
            description=schema["description"],
            emoji="🌙",
        )
    ctx.register_hook("pre_llm_call", h.pre_llm_call)
    ctx.register_hook("on_session_start", h.on_session_start)
    ctx.register_command(
        "nightshift",
        h.command,
        description="Training runs: status, approve, stop, report",
        args_hint="[status|approve <id>|stop <id>|report]",
    )
    if hasattr(ctx, "register_cli_command"):
        try:
            ctx.register_cli_command(
                "nightshift", "Nightshift training-run supervisor", setup_cli, make_cli_handler(h)
            )
        except Exception:  # skipped in the plugin host, absent on older hosts
            log.debug("nightshift: CLI command not registered", exc_info=True)
    if hasattr(ctx, "register_skill") and (SKILL_DIR / "SKILL.md").exists():
        try:
            ctx.register_skill("nightshift", SKILL_DIR / "SKILL.md", "Supervise ML training runs overnight")
        except Exception:
            log.debug("nightshift: skill not registered", exc_info=True)
    if hasattr(ctx, "register_automation_blueprint"):
        try:
            ctx.register_automation_blueprint("morning-report", **MORNING_REPORT)
        except Exception:
            log.debug("nightshift: blueprint not registered", exc_info=True)
    if hasattr(ctx, "on_unload"):
        ctx.on_unload(reset_runtime)
    start_soon(ctx)
    log.info("nightshift %s registered", __version__)
    return h
