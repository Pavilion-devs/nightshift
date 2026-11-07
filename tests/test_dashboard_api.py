"""The routes behind the Desktop pane, loaded the way Hermes loads them: standalone, by file path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import nightshift  # noqa: E402

API = Path(nightshift.__file__).parent / "dashboard" / "plugin_api.py"
TRAIN = str(Path(__file__).resolve().parents[1] / "examples" / "synthetic" / "train.py")


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in [m for m in sys.modules if m.startswith("_nightshift_dashboard")]:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location("hermes_dashboard_plugin_nightshift", API)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    impl = mod._engine()
    runtime = sys.modules["_nightshift_dashboard.adapter.runtime"]
    monkeypatch.setattr(runtime, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(impl, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(impl, "_store", None)
    app = fastapi.FastAPI()
    app.include_router(mod.router)
    yield TestClient(app), impl, tmp_path


def test_status_approve_deny_stop_ack(client) -> None:
    c, impl, tmp = client
    assert c.get("/status").json()["runs"] == []
    mgr = impl._manager()
    card = mgr.propose({"name": "p", "cwd": str(tmp), "command": [sys.executable, TRAIN, "--lr", "{lr}"],
                        "params": {"lr": 1e-3}})  # fmt: skip
    st = c.get("/status").json()
    [run] = st["runs"]
    assert run["status"] == "pending" and run["card"]["approve_with"].endswith(card["run_id"])
    assert c.post(f"/approve/{card['run_id']}").json()["ok"]
    assert (
        impl._get_store().get_run(card["run_id"])["status"] == "approved"
    )  # the supervising process launches
    assert c.post(f"/approve/{card['run_id']}").json()["ok"] is False

    card2 = mgr.propose({"name": "q", "cwd": str(tmp), "command": [sys.executable, TRAIN]})
    assert c.post(f"/deny/{card2['run_id']}").json()["ok"]
    assert c.post(f"/stop/{card['run_id']}").json()["ok"]  # queued for the supervising process
    assert c.post("/stop/missing").json()["ok"] is False
    assert c.post("/ack").json()["ok"]
    assert c.get(f"/run/{card['run_id']}").json()["name"] == "p"
    assert c.get("/run/missing").status_code == 404
