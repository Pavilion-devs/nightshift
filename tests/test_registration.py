"""register() must register exactly what plugin.yaml declares (catalog rule 6)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import nightshift

MANIFEST = Path(nightshift.__file__).parent / "plugin.yaml"


class FakeCtx:
    def __init__(self) -> None:
        self.tools: dict[str, dict[str, Any]] = {}
        self.hooks: dict[str, Any] = {}
        self.rpc: dict[str, Any] = {}
        self.commands: dict[str, Any] = {}

    def register_tool(self, name: str, toolset: str, schema: dict, handler: Any, **kw: Any) -> None:
        self.tools[name] = {"toolset": toolset, "schema": schema, "handler": handler, **kw}

    def register_hook(self, name: str, callback: Any, **kw: Any) -> None:
        self.hooks[name] = callback

    def register_rpc(self, name: str, handler: Any) -> None:
        self.rpc[name] = handler

    def register_command(self, name: str, handler: Any, **kw: Any) -> None:
        self.commands[name] = handler


def _manifest_list(key: str) -> list[str]:
    """Tiny reader for the two flat YAML lists we need, so tests stay dependency-free."""
    text = MANIFEST.read_text()
    inline = re.search(rf"^{key}:\s*\[(.*?)\]\s*$", text, re.M)
    if inline:
        return [s.strip() for s in inline.group(1).split(",") if s.strip()]
    block = re.search(rf"^{key}:\s*\n((?:\s+- .+\n)+)", text, re.M)
    return [ln.strip()[2:].strip() for ln in block.group(1).splitlines()] if block else []


def test_registrations_match_manifest() -> None:
    ctx = FakeCtx()
    nightshift.register(ctx)
    assert sorted(ctx.tools) == sorted(_manifest_list("provides_tools"))
    assert sorted(ctx.hooks) == sorted(_manifest_list("provides_hooks"))


def test_status_tool_returns_json_and_never_raises() -> None:
    ctx = FakeCtx()
    nightshift.register(ctx)
    out = json.loads(ctx.tools["nightshift_status"]["handler"]({}))
    assert out["ok"] is True
    assert out["version"] == nightshift.__version__


def test_status_rpc_registered() -> None:
    ctx = FakeCtx()
    nightshift.register(ctx)
    assert ctx.rpc["status"]({})["ok"] is True
