"""The Hermes adapter against a fake ctx: registrations match the manifest, tools behave, approval is human."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pytest

import nightshift
from nightshift.adapter import runtime as rt_mod
from nightshift.adapter.diagnose import HermesDiagnoser
from nightshift.adapter.notify import HermesNotifier
from nightshift.adapter.register import MORNING_REPORT
from nightshift.core.alerts import Alert
from nightshift.core.store import Store

MANIFEST = Path(nightshift.__file__).parent / "plugin.yaml"
TRAIN = str(Path(__file__).resolve().parents[1] / "examples" / "synthetic" / "train.py")


class FakeLlm:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[list[dict]] = []
        self.fail = fail

    def complete(self, messages: list[dict], **kw: Any) -> Any:
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("no model configured")
        return type("R", (), {"text": "  The lr of 3e-3 is too high after warmup;\n use 1e-3.  "})()


class FakeCtx:
    def __init__(self, cwd: str = "", surface: str = "cli", config: dict | None = None) -> None:
        self.tools: dict[str, dict[str, Any]] = {}
        self.hooks: dict[str, Any] = {}
        self.commands: dict[str, Any] = {}
        self.cli: dict[str, Any] = {}
        self.skills: dict[str, Any] = {}
        self.blueprints: dict[str, Any] = {}
        self.unload: list[Any] = []
        self.dispatched: list[tuple[str, dict]] = []
        self.config = config or {}
        self.llm = FakeLlm()
        self._cwd, self._surface = cwd, surface

    def register_tool(self, name: str, toolset: str, schema: dict, handler: Any, **kw: Any) -> None:
        self.tools[name] = {"toolset": toolset, "schema": schema, "handler": handler, **kw}

    def register_hook(self, name: str, callback: Any, **kw: Any) -> None:
        self.hooks[name] = callback

    def register_command(self, name: str, handler: Any, **kw: Any) -> None:
        self.commands[name] = handler

    def register_cli_command(self, name: str, help: str, setup_fn: Any, handler_fn: Any) -> None:
        self.cli[name] = (setup_fn, handler_fn)

    def register_skill(self, name: str, path: Path, description: str = "") -> None:
        self.skills[name] = path

    def register_automation_blueprint(self, key: str, **kw: Any) -> None:
        self.blueprints[key] = kw

    def on_unload(self, cb: Any) -> None:
        self.unload.append(cb)

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def session_cwd(self) -> str:
        return self._cwd

    def session_surface(self) -> str:
        return self._surface

    def dispatch_tool(self, name: str, args: dict, **kw: Any) -> str:
        self.dispatched.append((name, args))
        return json.dumps({"success": True})


@pytest.fixture
def ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data = tmp_path / "plugin-data"
    data.mkdir()
    monkeypatch.setattr(rt_mod, "data_dir", lambda: data)
    work = tmp_path / "work"
    work.mkdir()
    c = FakeCtx(cwd=str(work), surface="telegram")
    nightshift.register(c)
    yield c
    rt_mod.reset_runtime()


def _call(c: FakeCtx, tool: str, **args: Any) -> dict:
    return json.loads(c.tools[tool]["handler"](args))


def _manifest_list(key: str) -> list[str]:
    text = MANIFEST.read_text()
    inline = re.search(rf"^{key}:\s*\[(.*?)\]\s*$", text, re.M)
    if inline:
        return [s.strip() for s in inline.group(1).split(",") if s.strip()]
    block = re.search(rf"^{key}:\s*\n((?:\s+- .+\n)+)", text, re.M)
    return [ln.strip()[2:].strip() for ln in block.group(1).splitlines()] if block else []


def test_registrations_match_manifest(ctx: FakeCtx) -> None:
    assert sorted(ctx.tools) == sorted(_manifest_list("provides_tools"))
    assert sorted(ctx.hooks) == sorted(_manifest_list("provides_hooks"))
    assert set(ctx.commands) == {"nightshift"} and set(ctx.cli) == {"nightshift"}
    assert ctx.skills["nightshift"].name == "SKILL.md"
    assert ctx.blueprints["morning-report"]["schedule_template"] == MORNING_REPORT["schedule_template"]
    assert ctx.unload and all(t["toolset"] == "nightshift" for t in ctx.tools.values())


def test_tools_never_raise(ctx: FakeCtx) -> None:
    for name in ctx.tools:
        out = json.loads(ctx.tools[name]["handler"]({"run_id": 123, "command": None, "hours": "x"}))
        assert isinstance(out, dict)
    assert _call(ctx, "nightshift_launch", command="")["ok"] is False
    assert _call(ctx, "nightshift_run", run_id="nope")["ok"] is False


def test_agent_can_only_propose_and_the_human_approves(ctx: FakeCtx) -> None:
    card = _call(
        ctx,
        "nightshift_launch",
        command=f"{sys.executable} {TRAIN} --steps 200 --sleep 0.002 --lr {{lr}} --resume={{resume}}",
        params={"lr": 1e-3},
        name="tiny",
    )
    assert card["ok"] and card["status"] == "pending"
    assert f"/nightshift approve {card['run_id']}" in card["next_step"]
    status = _call(ctx, "nightshift_status")
    assert status["pending_approvals"] == [card["run_id"]]
    assert "waiting for you" in ctx.commands["nightshift"]("status")

    reply = ctx.commands["nightshift"](f"approve {card['run_id']}")
    assert reply.startswith("✅ Approved")
    store = rt_mod.get_runtime(ctx).store
    deadline = time.time() + 30
    while time.time() < deadline and store.get_run(card["run_id"])["status"] != "finished":
        time.sleep(0.1)
    run = store.get_run(card["run_id"])
    assert run["status"] == "finished" and run["origin"] == {"surface": "telegram"}
    detail = _call(ctx, "nightshift_run", run_id=card["run_id"])
    assert detail["series"]["loss"] and detail["command"][-1] == "--resume={resume}"
    assert "finished" in ctx.commands["nightshift"](f"show {card['run_id']}")
    report = _call(ctx, "nightshift_report", hours=1)
    assert report["runs"] >= 1 and "Nightshift report" in report["text"]


def test_command_paths(ctx: FakeCtx) -> None:
    cmd = ctx.commands["nightshift"]
    assert "no runs yet" in cmd("")
    assert "Usage" in cmd("bogus")
    assert "Not approved" in cmd("approve missing")
    assert "No pending run" in cmd("deny missing")
    assert "Not stopped" in cmd("stop missing")
    assert "Marked 0" in cmd("ack")
    assert "Nightshift report" in cmd("report 2")
    assert "No run" in cmd("show missing")
    card = _call(ctx, "nightshift_launch", command=f"{sys.executable} {TRAIN}", name="x")
    assert "Declined" in cmd(f"deny {card['run_id']}")


def test_pre_llm_call_mentions_each_incident_once_per_session(ctx: FakeCtx, tmp_path: Path) -> None:
    hook = ctx.hooks["pre_llm_call"]
    assert hook(session_id="s1") is None  # silent while healthy
    log = tmp_path / "work" / "ext.log"
    log.write_text(
        "".join(f"step {i * 10}: loss {3 - i * 0.01:.3f}\n" for i in range(50)) + "step 510: loss nan\n"
    )
    res = _call(ctx, "nightshift_watch", log_path="ext.log", name="ext")
    store = rt_mod.get_runtime(ctx).store
    deadline = time.time() + 15
    while time.time() < deadline and not store.incidents(res["run_id"]):
        time.sleep(0.1)
    out = hook(session_id="s1")
    assert out and "[Nightshift]" in out["context"] and "ext" in out["context"]
    assert hook(session_id="s1") is None  # already mentioned in this conversation
    assert hook(session_id="s2") is not None  # a new conversation hears about it once too
    assert _call(ctx, "nightshift_ack")["acknowledged"] >= 1
    assert hook(session_id="s3") is None


def test_cli_handler(ctx: FakeCtx, capsys: pytest.CaptureFixture[str]) -> None:
    import argparse

    setup, handler = ctx.cli["nightshift"]
    parser = argparse.ArgumentParser()
    setup(parser)
    assert handler(parser.parse_args(["status"])) == 0
    assert handler(parser.parse_args(["report", "--hours", "1"])) == 0
    assert handler(parser.parse_args(["approve", "nope"])) == 0
    out = capsys.readouterr().out
    assert "Nightshift" in out and "Not approved" in out


# --- notifier & diagnoser ------------------------------------------------------------------------------------


def _alert(page: bool = True, chart: bytes | None = b"\x89PNG fake") -> Alert:
    return Alert(run_id="r1", run_name="tiny", title="🔥 tiny: NaN", body="token=abcdef123456 leaked", page=page,
                 kind="nan_inf", chart_png=chart)  # fmt: skip


def test_notifier_pushes_pages_with_chart_and_scrubs_secrets(tmp_path: Path, monkeypatch) -> None:
    import nightshift.adapter.notify as notify_mod

    monkeypatch.setattr(notify_mod, "_image_dir", lambda: tmp_path)
    store = Store(tmp_path / "n.db")
    store.create_run(id="r1", name="tiny", mode="launch", origin={"surface": "telegram"})
    c = FakeCtx()
    n = HermesNotifier(c, store)
    n.send(_alert(page=False))
    assert c.dispatched == []  # quiet alerts never buzz a phone
    n.send(_alert())
    [(tool, args)] = c.dispatched
    assert tool == "send_message" and args["target"] == "telegram"
    assert "abcdef123456" not in args["message"] and "MEDIA:" in args["message"]
    chart_path = args["message"].split("MEDIA:")[1].strip()
    assert Path(chart_path).read_bytes().startswith(b"\x89PNG")


def test_notifier_target_rules(tmp_path: Path) -> None:
    store = Store(tmp_path / "n.db")
    store.create_run(id="r1", name="tiny", mode="launch", origin={"surface": "desktop"})
    assert HermesNotifier(FakeCtx(), store).target_for("r1") is None  # local screen: nothing to push to
    assert HermesNotifier(FakeCtx(config={"alert_target": "discord"}), store).target_for("r1") == "discord"
    assert HermesNotifier(FakeCtx(config={"alert_target": "none"}), store).target_for("r1") is None
    c = FakeCtx(config={"alert_target": "telegram"})
    c.dispatch_tool = lambda *a, **k: json.dumps({"error": "telegram not configured"})  # type: ignore[method-assign]
    n = HermesNotifier(c, store)
    n.send(_alert(chart=None))
    assert n.sent[-1]["ok"] is False


def test_diagnoser() -> None:
    c = FakeCtx()
    text = HermesDiagnoser(c).diagnose(
        {"incident": {"kind": "nan_inf"}, "log_tail": ["api_key=sk-abcdefghijklmnop1234"]}
    )
    assert text == "The lr of 3e-3 is too high after warmup; use 1e-3."
    assert "sk-abcdefghijklmnop1234" not in c.llm.calls[0][1]["content"]
    assert HermesDiagnoser(FakeCtx(config={"diagnose": False})).diagnose({}) is None
    failing = FakeCtx()
    failing.llm = FakeLlm(fail=True)
    assert HermesDiagnoser(failing).diagnose({}) is None
