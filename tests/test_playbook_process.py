from __future__ import annotations

import math
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from nightshift.core.chart_png import Series, render
from nightshift.core.models import Incident, IncidentKind, MetricPoint, Severity
from nightshift.core.playbook import (
    NOTIFY,
    OOM_RESTART,
    ROLLBACK_RESTART,
    STOP,
    Playbook,
    RunState,
    decide,
    is_actionable,
)
from nightshift.core.process import (
    ProcessHandle,
    TemplateError,
    fingerprint,
    gpu_utilization,
    parse_command,
    pid_alive,
    placeholders,
    read_tail,
    render_argv,
    template_env,
)
from nightshift.core.redact import redact, redact_lines
from nightshift.core.report import alert_for_status, chart_for_run, morning_report
from nightshift.core.store import Store

NAN = Incident(
    IncidentKind.NAN_INF, Severity.CRITICAL, "loss is nan", metric="loss", step=640, onset_step=610
)
OOM = Incident(IncidentKind.OOM, Severity.CRITICAL, "out of memory")
SPIKE_WARN = Incident(IncidentKind.SPIKE, Severity.WARN, "spike", metric="loss")
PLATEAU = Incident(IncidentKind.PLATEAU, Severity.INFO, "plateau", metric="loss")


def _state(**kw) -> RunState:
    base = {"mode": "launch", "params": {"lr": 3e-3, "batch_size": 64, "grad_accum": 1}}
    base.update(kw)
    return RunState(**base)


# --- playbook ---------------------------------------------------------------------------------------------------


def test_rollback_halves_lr_and_pages() -> None:
    d = decide(Playbook(), NAN, _state(), now=1000)
    assert d.action == ROLLBACK_RESTART and d.page
    assert d.params["lr"] == pytest.approx(1.5e-3) and d.changes == {"lr": (3e-3, 1.5e-3)}


def test_oom_halves_batch_and_doubles_accum() -> None:
    d = decide(Playbook(), OOM, _state(), now=1000)
    assert d.action == OOM_RESTART
    assert d.params["batch_size"] == 32 and d.params["grad_accum"] == 2


def test_guard_rails() -> None:
    pb = Playbook(max_interventions=2, cooldown_s=600)
    assert decide(pb, NAN, _state(interventions=2), 1000).action == STOP
    d = decide(pb, NAN, _state(interventions=1, last_intervention_at=900), 1000)
    assert d.action == STOP and "after the last recovery" in d.reason
    assert decide(pb, NAN, _state(interventions=1, last_intervention_at=100), 1000).action == ROLLBACK_RESTART
    assert decide(pb, NAN, _state(can_resume=False), 1000).action == STOP
    assert decide(pb, NAN, _state(params={"lr": 1e-7}), 1000).action == STOP  # already at the floor
    assert (
        decide(pb, NAN, _state(params={}), 1000).action == ROLLBACK_RESTART
    )  # rollback without an lr change
    assert decide(pb, OOM, _state(params={"batch_size": 1}), 1000).action == STOP
    assert decide(pb, OOM, _state(params={"lr": 1.0}), 1000).action == STOP
    d = decide(pb, OOM, _state(params={"batch_size": 8}), 1000)
    assert d.params == {"batch_size": 4} and "grad_accum" not in d.changes


def test_watch_mode_and_quiet_kinds() -> None:
    d = decide(Playbook(), NAN, _state(mode="watch"), 1000)
    assert d.action == NOTIFY and "read-only" in d.reason and d.page
    assert decide(Playbook(), SPIKE_WARN, _state(), 1000).page is False  # warnings do not wake anyone
    assert decide(Playbook(), PLATEAU, _state(), 1000).page is False
    assert not is_actionable(PLATEAU) and is_actionable(NAN)


def test_custom_rules_and_validation() -> None:
    pb = Playbook.from_dict({"rules": {"nan_inf": "stop", "overfit": "oom_restart"}, "max_interventions": 5})
    assert decide(pb, NAN, _state(), 1000).action == STOP
    over = Incident(IncidentKind.OVERFIT, Severity.WARN, "overfit")
    assert decide(pb, over, _state(), 1000).action == NOTIFY  # oom_restart only applies to OOM
    assert pb.to_dict()["max_interventions"] == 5
    with pytest.raises(ValueError):
        Playbook(rules={"nan_inf": "explode"})
    with pytest.raises(ValueError):
        Playbook(lr_factor=2.0)
    with pytest.raises(ValueError):
        Playbook(max_interventions=-1)
    lines = Playbook().describe({"lr": 1e-3, "batch_size": 8, "grad_accum": 1})
    assert any("lr ×0.5" in ln for ln in lines) and any("grad_accum ×2" in ln for ln in lines)
    assert any(
        "notify only" in ln for ln in Playbook(rules=dict.fromkeys(["nan_inf"], "notify")).describe({})
    )


# --- templating & processes -------------------------------------------------------------------------------------


def test_render_argv() -> None:
    tpl = ["python", "train.py", "--lr", "{lr}", "--resume={resume}", "--bs={batch_size}"]
    assert placeholders(tpl) == {"lr", "resume", "batch_size"}
    assert render_argv(tpl, {"lr": 0.0015, "batch_size": 32}) == [
        "python",
        "train.py",
        "--lr",
        "0.0015",
        "--bs=32",
    ]
    assert render_argv(tpl, {"lr": 1e-4, "batch_size": 32}, resume="out/c.pt")[4] == "--resume=out/c.pt"
    two_words = ["python", "t.py", "--resume", "{resume}", "--lr", "{lr}"]
    assert render_argv(two_words, {"lr": 1}) == ["python", "t.py", "--lr", "1"]
    assert render_argv(two_words, {"lr": 1}, resume="c.pt") == [
        "python",
        "t.py",
        "--resume",
        "c.pt",
        "--lr",
        "1",
    ]
    with pytest.raises(TemplateError, match="no value for lr"):
        render_argv(tpl, {"batch_size": 1})
    with pytest.raises(TemplateError, match="bad placeholder"):
        placeholders(["{0}"])
    assert parse_command("python 'my train.py' --x 1") == ["python", "my train.py", "--x", "1"]
    with pytest.raises(TemplateError):
        parse_command("")
    env = template_env({"lr": 1e-3}, "c.pt", "abc", 2)
    assert env == {
        "NIGHTSHIFT_LR": "0.001",
        "NIGHTSHIFT_RUN_ID": "abc",
        "NIGHTSHIFT_ATTEMPT": "2",
        "NIGHTSHIFT_RESUME": "c.pt",
    }


def test_launch_poll_stop_kills_children_too(tmp_path: Path) -> None:
    script = tmp_path / "parent.py"
    script.write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    h = ProcessHandle.launch(
        [sys.executable, str(script)], cwd=str(tmp_path), log_path=str(tmp_path / "l.log")
    )
    deadline = time.time() + 10
    while time.time() < deadline and not (tmp_path / "l.log").read_text().strip():
        time.sleep(0.05)
    child = int((tmp_path / "l.log").read_text().split()[0])
    assert h.poll() == (True, None) and h.is_ours() and pid_alive(child)
    assert h.stop(grace_s=3)
    time.sleep(0.2)
    assert not pid_alive(child), "the dataloader-style child must die with the group"
    assert h.poll()[0] is False
    assert h.stop() is True  # idempotent


def test_sigterm_ignoring_process_gets_sigkill(tmp_path: Path) -> None:
    script = tmp_path / "stubborn.py"
    script.write_text(
        "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nprint('ok', flush=True)\ntime.sleep(60)\n"
    )
    h = ProcessHandle.launch(
        [sys.executable, str(script)], cwd=str(tmp_path), log_path=str(tmp_path / "s.log")
    )
    deadline = time.time() + 10
    while time.time() < deadline and "ok" not in (tmp_path / "s.log").read_text():
        time.sleep(0.05)
    assert h.stop(grace_s=0.5)
    assert h.popen.returncode == -signal.SIGKILL


def test_reattach_refuses_recycled_pids(tmp_path: Path) -> None:
    h = ProcessHandle.launch([sys.executable, "-c", "import time; time.sleep(30)"], cwd=str(tmp_path),
                             log_path=str(tmp_path / "x.log"))  # fmt: skip
    try:
        same = ProcessHandle.reattach(h.pid, h.pgid, h.proc_id, h.log_path)
        assert same.is_ours() and same.poll() == (True, None)
        impostor = ProcessHandle.reattach(
            h.pid, h.pgid, "Mon Jan  1 00:00:00 1990 something-else", h.log_path
        )
        assert not impostor.is_ours()
        assert impostor.stop() is True and pid_alive(h.pid)  # refused to signal a process that is not ours
    finally:
        os.killpg(h.pgid, signal.SIGKILL)
        h.popen.wait()
    gone = ProcessHandle.reattach(h.pid, h.pgid, h.proc_id, h.log_path)
    assert not gone.is_ours() and fingerprint(h.pid) is None


def test_reattach_survives_a_launcher_that_re_execs_itself(tmp_path: Path) -> None:
    """python.org's macOS Python, torchrun and shell wrappers exec into another command line after launch."""
    code = "import os, sys, time; time.sleep(0.3); os.execv(sys.executable, [sys.executable, '-c', 'import time; time.sleep(30)'])"  # noqa: E501
    h = ProcessHandle.launch(
        [sys.executable, "-c", code], cwd=str(tmp_path), log_path=str(tmp_path / "x.log")
    )
    try:
        time.sleep(1.0)  # the command line has changed by now; the start time has not
        same = ProcessHandle.reattach(h.pid, h.pgid, h.proc_id, h.log_path)
        assert same.is_ours()
        legacy = ProcessHandle.reattach(h.pid, h.pgid, f"{h.proc_id} /old/python -c x", h.log_path)
        assert legacy.is_ours()
    finally:
        os.killpg(h.pgid, signal.SIGKILL)
        h.popen.wait()


def test_read_tail_and_gpu(tmp_path: Path) -> None:
    f = tmp_path / "log"
    f.write_text("".join(f"line {i}\n" for i in range(500)) + "bar 1\rbar 2\n")
    tail = read_tail(str(f), max_lines=5)
    assert tail[-1] == "bar 2" and len(tail) == 5
    assert read_tail(str(tmp_path / "missing")) == []
    util = gpu_utilization()
    assert util is None or 0 <= util <= 100


# --- redaction ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "secret",
    [
        "sk-abcdefghijklmnopqrstuv",
        "ghp_abcdefghijklmnopqrstuvwxyz",
        "hf_abcdefghijklmnopqrstuvwx",
        "AKIAABCDEFGHIJKLMNOP",
        "123456789:AAabcdefghijklmnopqrstuvwxyz0123456",
        "xoxb-1234567890-abcdef",
    ],
)
def test_redacts_tokens(secret: str) -> None:
    assert secret not in redact(f"logging in with {secret} now")


def test_redacts_assignments_and_urls() -> None:
    out = redact_lines(
        [
            "WANDB_API_KEY=abcd1234efgh",
            'password: "hunter22"',
            "https://user:pw@host/x",
            "Bearer abc.def.ghijklmnopqrstu",
        ]
    )
    assert (
        "abcd1234efgh" not in out[0]
        and "hunter22" not in out[1]
        and "pw@" not in out[2]
        and "ghijk" not in out[3]
    )
    assert redact("loss=2.31 step=10") == "loss=2.31 step=10"


# --- charts & reports --------------------------------------------------------------------------------------------


def test_render_png_shapes() -> None:
    png = render([Series([(i, 2 + math.sin(i)) for i in range(50)] + [(50, math.nan)], attempts=[0] * 25 + [1] * 26)],
                 incidents_x=[30], recoveries_x=[35], log_y=False)  # fmt: skip
    assert png.startswith(b"\x89PNG\r\n\x1a\n") and b"IEND" in png
    assert render([]).startswith(b"\x89PNG")
    assert render([Series([(1, 5.0)])], log_y=True, x_is_time=False).startswith(b"\x89PNG")
    assert render([Series([(1, math.inf), (2, 0.0)])], log_y=True).startswith(b"\x89PNG")


def test_reports(tmp_path: Path) -> None:
    store = Store(tmp_path / "n.db")
    rid = store.create_run(name="tiny", mode="launch", status="finished", ended_at=5000.0)
    store.add_metrics(rid, [MetricPoint(i, "loss", 3 - i * 0.01, 1000.0 + i) for i in range(100)])
    store.add_metrics(
        rid, [MetricPoint(i, "val_loss", 3.1 - i * 0.01, 1000.0 + i) for i in range(0, 100, 10)]
    )
    iid = store.add_incident(rid, NAN, at=1100.0)
    store.add_action(rid, ROLLBACK_RESTART, True, "rolled back to step 600", incident_id=iid, at=1200.0)
    store.add_action(rid, NOTIFY, True, "noted", at=1300.0)
    rep = morning_report(store, since=0, now=10_000)
    assert rep["recoveries"] == 1 and rep["incidents"] == 1
    assert rep["gpu_hours_rescued_estimate"] == pytest.approx((5000 - 1200) / 3600, abs=0.01)
    assert "rolled back to step 600" in rep["text"] and "tiny: finished" in rep["text"]
    assert chart_for_run(store, store.get_run(rid)).startswith(b"\x89PNG")
    alert = alert_for_status(store, store.get_run(rid), "finished", "training finished")
    assert not alert.page and "Final loss" in alert.body
    assert alert_for_status(store, store.get_run(rid), "failed", "boom").page
