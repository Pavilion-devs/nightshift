"""Deliver alerts. Every alert is already in Nightshift's database (the Desktop pane and the next chat turn see it);
paging alerts are also pushed to a messaging platform with Hermes's own `hermes send` (or ``send_message``
where a Hermes version provides it as a tool)."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..core.alerts import Alert
from ..core.redact import redact
from ..core.store import Store

log = logging.getLogger("nightshift")

# surfaces that are a person's screen, not a chat we can push to
_LOCAL_SURFACES = {"", "cli", "tui", "desktop", "acp", "local", "api"}


def _image_dir() -> Path:
    """Hermes only attaches images from its own caches, so charts are written there."""
    try:
        from hermes_constants import get_hermes_home

        root = Path(get_hermes_home())
    except ImportError:
        root = Path.home() / ".hermes"
    path = root / "image_cache" / "nightshift"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _scrub(text: str) -> str:
    # Nightshift's own pass; Hermes's redaction helper is not part of the public plugin API (catalog rule 9)
    return redact(text)


class HermesNotifier:
    def __init__(self, ctx: Any, store: Store) -> None:
        self.ctx = ctx
        self.store = store
        self.sent: list[dict[str, Any]] = []  # last deliveries, for status and tests

    def target_for(self, run_id: str) -> str | None:
        try:
            configured = (self.ctx.get_config("alert_target", "") or "").strip()
        except Exception:
            configured = ""
        if configured:
            return None if configured.lower() in ("none", "off", "local") else configured
        run = self.store.get_run(run_id) or {}
        surface = ((run.get("origin") or {}).get("surface") or "").lower()
        return None if surface in _LOCAL_SURFACES else surface

    def send(self, alert: Alert) -> None:
        if not alert.page:
            return  # quiet: pane, next chat turn and the morning report only
        target = self.target_for(alert.run_id)
        if not target:
            return
        message = _scrub(f"{alert.title}\n{alert.body}")
        if alert.chart_png:
            try:
                path = _image_dir() / f"{alert.run_id}-{int(time.time() * 1000)}.png"
                path.write_bytes(alert.chart_png)
                message += f"\nMEDIA:{path}"
            except OSError:
                log.warning("nightshift: could not write the chart image", exc_info=True)
        args = {"action": "send", "target": target, "message": message}
        try:
            result = _as_dict(self.ctx.dispatch_tool("send_message", args))
            if _unknown_tool(result):
                # Hermes v0.23 has no agent-callable send_message tool (toolsets.py says so on purpose).
                # Its documented route for pushing text from outside a conversation is `hermes send`:
                # no LLM, no agent loop, the gateway's own stored credentials. Nightshift never sees them.
                result = hermes_send(target, message)
        except Exception as exc:
            log.warning("nightshift: alert delivery to %s failed: %s", target, exc)
            result = {"error": str(exc)}
        ok = isinstance(result, dict) and not result.get("error") and result.get("success", True) is not False
        if not ok:
            log.warning("nightshift: alert delivery to %s failed: %s", target, result)
        self.sent.append({"target": target, "ok": ok, "title": alert.title, "result": result})
        del self.sent[:-20]


def _as_dict(raw: Any) -> Any:
    return json.loads(raw) if isinstance(raw, str) else raw


def _unknown_tool(result: Any) -> bool:
    return isinstance(result, dict) and "Unknown tool" in str(result.get("error", ""))


def hermes_send(target: str, message: str, timeout_s: float = 60.0) -> dict[str, Any]:
    """Deliver through Hermes's own `hermes send` CLI (argv, no shell; the text goes in on stdin)."""
    exe = shutil.which("hermes")
    argv = [exe] if exe else [sys.executable, "-m", "hermes_cli.main"]
    env = None
    if (
        not exe
    ):  # the plugin host runs Hermes's Python with -I: point it at the Hermes package it already uses
        spec = importlib.util.find_spec("hermes_cli")
        if spec is None or not spec.origin:
            return {"error": "hermes CLI not found"}
        env = {**os.environ, "PYTHONPATH": str(Path(spec.origin).resolve().parent.parent)}
    argv += ["send", "--to", target, "--file", "-", "--json"]
    try:
        proc = subprocess.run(argv, input=message, capture_output=True, text=True, timeout=timeout_s, env=env,
                              check=False)  # fmt: skip
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"hermes send: {exc}"}
    out = proc.stdout.strip()
    try:
        result = json.loads(out.splitlines()[-1]) if out else {}
    except ValueError:
        result = {"output": out[-300:]}
    if proc.returncode != 0:
        return {"error": (proc.stderr.strip() or out)[-300:] or f"hermes send exited {proc.returncode}"}
    return result if isinstance(result, dict) else {"result": result}
