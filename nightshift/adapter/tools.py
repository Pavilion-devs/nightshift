"""Tool handlers, the /nightshift command, the CLI command and the hooks.

Every handler returns JSON (or text, for the command), never raises, and returns at once: the long work happens
on the supervisor threads. Tools only ever *propose* launches; starting or stopping a process takes a human.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from ..core import views
from ..core.manager import SpecError
from ..core.report import morning_report
from .runtime import get_runtime

log = logging.getLogger("nightshift")

APPROVAL_NOTE = (
    "Nothing has started. Show the user this card (command, directory, what recovery may change) and ask "
    "them to approve it with `/nightshift approve {run_id}` or the Approve button in the Nightshift pane."
)


def _json(obj: Any) -> str:
    return json.dumps(obj, default=str)


def _safe(fn: Callable[..., Any]) -> Callable[..., str]:
    def wrapper(args: dict | None = None, **kwargs: Any) -> str:
        try:
            return _json(fn(args or {}, **kwargs))
        except SpecError as exc:
            return _json({"ok": False, "error": str(exc)})
        except Exception as exc:
            log.exception("nightshift: tool %s failed", fn.__name__)
            return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    wrapper.__name__ = fn.__name__
    return wrapper


class Handlers:
    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self._surfaced: dict[str, set[int]] = {}  # session -> incident ids already mentioned

    @property
    def rt(self):
        return get_runtime(self.ctx)

    def _origin(self) -> dict[str, Any]:
        try:
            return {"surface": self.ctx.session_surface()}
        except Exception:
            return {}

    def _cwd(self, args: dict) -> str | None:
        if args.get("cwd"):
            return args["cwd"]
        try:
            return self.ctx.session_cwd() or None
        except Exception:
            return None

    # -- tools ----------------------------------------------------------------------------------------
    def launch(self, args: dict, **_: Any) -> dict:
        spec = dict(args)
        spec["cwd"] = self._cwd(args)
        card = self.rt.manager.propose(spec, origin=self._origin())
        card["next_step"] = APPROVAL_NOTE.format(run_id=card["run_id"])
        return card

    def watch(self, args: dict, **_: Any) -> dict:
        spec = dict(args)
        spec["cwd"] = self._cwd(args)
        return self.rt.manager.watch(spec, origin=self._origin())

    def status(self, args: dict, **_: Any) -> dict:
        return views.status(self.rt.store)

    def run(self, args: dict, **_: Any) -> dict:
        detail = views.run_detail(self.rt.store, str(args.get("run_id", "")), max_points=60)
        return detail if detail else {"ok": False, "error": f"no run {args.get('run_id')!r}"}

    def report(self, args: dict, **_: Any) -> dict:
        hours = float(args.get("hours") or 12)
        return morning_report(self.rt.store, since=time.time() - hours * 3600)

    def ack(self, args: dict, **_: Any) -> dict:
        return self.rt.manager.ack(args.get("run_id") or None)

    def tool_handlers(self) -> dict[str, Callable[..., str]]:
        return {
            "nightshift_launch": _safe(self.launch),
            "nightshift_watch": _safe(self.watch),
            "nightshift_status": _safe(self.status),
            "nightshift_run": _safe(self.run),
            "nightshift_report": _safe(self.report),
            "nightshift_ack": _safe(self.ack),
        }

    # -- hooks ----------------------------------------------------------------------------------------
    def on_session_start(self, **_: Any) -> None:
        try:
            get_runtime(self.ctx)  # resumes supervision of runs left running by a previous Hermes
        except Exception:
            log.exception("nightshift: could not start")

    def pre_llm_call(self, **kwargs: Any) -> dict[str, str] | None:
        """Mention new, unacknowledged problems once per conversation. Silent while every run is healthy."""
        try:
            rt = get_runtime(self.ctx)
            fresh = [i for i in rt.store.incidents(unacked_only=True, limit=20) if i["severity"] != "info"]
            if not fresh:
                return None
            session = str(kwargs.get("session_id") or kwargs.get("session_key") or "default")
            seen = self._surfaced.setdefault(session, set())
            new = [i for i in fresh if i["id"] not in seen]
            if not new:
                return None
            seen.update(i["id"] for i in new)
            lines = []
            for inc in new[:3]:
                run = rt.store.get_run(inc["run_id"]) or {"name": inc["run_id"]}
                act = next(
                    (a for a in rt.store.actions(inc["run_id"], limit=20) if a["incident_id"] == inc["id"]),
                    None,
                )
                outcome = f" → {act['message']}" if act else ""
                lines.append(f"- {run['name']} ({inc['run_id']}): {inc['message']}{outcome}")
            more = f"\n(+{len(new) - 3} more)" if len(new) > 3 else ""
            text = (
                "[Nightshift] New training-run incident(s) since the user last looked:\n"
                + "\n".join(lines)
                + more
                + "\nMention this briefly if relevant. `/nightshift ack` marks them seen."
            )
            return {"context": text}
        except Exception:
            log.exception("nightshift: pre_llm_call failed")
            return None

    # -- /nightshift ------------------------------------------------------------------------------------
    def command(self, raw: str = "", **_: Any) -> str:
        try:
            return self._command(raw)
        except Exception as exc:
            log.exception("nightshift: /nightshift failed")
            return f"Nightshift: {type(exc).__name__}: {exc}"

    def _command(self, raw: str) -> str:
        parts = (raw or "").split()
        sub = parts[0].lower() if parts else "status"
        arg = parts[1] if len(parts) > 1 else None
        rt = self.rt
        if sub in ("status", "runs"):
            return format_status(views.status(rt.store))
        if sub == "show" and arg:
            detail = views.run_detail(rt.store, arg, max_points=0)
            return format_detail(detail) if detail else f"No run {arg}."
        if sub == "approve" and arg:
            res = rt.manager.approve(arg)
            return f"✅ Approved {arg}: launching now." if res["ok"] else f"Not approved: {res['error']}"
        if sub in ("deny", "cancel") and arg:
            res = rt.manager.deny(arg)
            return f"Declined {arg}." if res["ok"] else f"No pending run {arg}."
        if sub == "stop" and arg:
            res = rt.manager.request_stop(arg)
            return f"🛑 Stopping {arg}." if res["ok"] else f"Not stopped: {res.get('error')}"
        if sub == "ack":
            return f"Marked {rt.manager.ack(arg)['acknowledged']} incident(s) as seen."
        if sub == "report":
            hours = float(arg) if arg else 12.0
            return morning_report(rt.store, since=time.time() - hours * 3600)["text"]
        return (
            "Usage: /nightshift [status | show <id> | approve <id> | deny <id> | stop <id> | ack [id] | "
            "report [hours]]"
        )


def format_status(st: dict) -> str:
    if not st["runs"]:
        return "🌙 Nightshift: no runs yet. Ask Hermes to launch or watch a training run."
    icon = {"healthy": "🟢", "warning": "🟡", "critical": "🔴", "stopped": "⚪", "failed": "❌"}
    lines = ["🌙 Nightshift"]
    for r in st["runs"]:
        loss = r["train_loss"]["value"] if r["train_loss"] else "—"
        step = r["step"] if r["step"] is not None else "—"
        lines.append(
            f"{icon.get(r['health'], '•')} {r['name']} [{r['id']}] {r['status']} · step {step} · loss {loss}"
            f" · {r['interventions']} intervention(s)"
        )
        for inc in r["open_incidents"][:2]:
            lines.append(f"    ↳ {inc['severity']}: {inc['message']}")
        if r["status"] == "pending":
            lines.append(f"    ↳ waiting for you: /nightshift approve {r['id']}")
    return "\n".join(lines)


def format_detail(d: dict) -> str:
    lines = [f"🌙 {d['name']} [{d['id']}] {d['status']} (attempt {d['attempt']})"]
    if d.get("command"):
        lines.append(f"command: {' '.join(d['command'])}")
    lines.append(f"params: {d['params']}")
    for a in reversed(d["actions"][:10]):
        lines.append(f"  {time.strftime('%H:%M', time.localtime(a['at']))} {a['kind']}: {a['message']}")
    for i in d["incidents"][:5]:
        if i.get("diagnosis"):
            lines.append(f"  why ({i['kind']}): {i['diagnosis']}")
    return "\n".join(lines)


# -- `hermes nightshift ...` -----------------------------------------------------------------------------


def setup_cli(parser: Any) -> None:
    sub = parser.add_subparsers(dest="ns_cmd")
    sub.add_parser("status", help="Show watched runs")
    for name in ("approve", "deny", "stop", "show"):
        p = sub.add_parser(name, help=f"{name} a run")
        p.add_argument("run_id")
    rp = sub.add_parser("report", help="Overnight report")
    rp.add_argument("--hours", type=float, default=12.0)


def make_cli_handler(handlers: Handlers) -> Callable[[Any], int]:
    def handle(args: Any) -> int:
        cmd = getattr(args, "ns_cmd", None) or "status"
        raw = cmd
        if getattr(args, "run_id", None):
            raw += f" {args.run_id}"
        if cmd == "report":
            raw += f" {args.hours}"
        print(handlers.command(raw))
        return 0

    return handle
