"""Deliver alerts. Every alert is already in Nightshift's database (the Desktop pane and the next chat turn see it);
paging alerts are also pushed to a messaging platform through Hermes's own ``send_message`` tool."""

from __future__ import annotations

import json
import logging
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
                # Hermes registers its built-in tools on the first agent turn; a run approved from the CLI
                # right after a gateway start can alert before that. Load them through the public
                # registry and send with Hermes's own tool (it reads the platform config, not us).
                result = _as_dict(_dispatch_builtin("send_message", args))
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


def _dispatch_builtin(name: str, args: dict[str, Any]) -> Any:
    from tools.registry import discover_builtin_tools, registry  # public plugin API (tools.registry)

    if registry.get_entry(name) is None:
        discover_builtin_tools()
    return registry.dispatch(name, args)
