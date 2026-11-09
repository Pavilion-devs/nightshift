"""Real processes, real files, real signals: the whole night shift, minus Hermes."""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import pytest

from nightshift.core.alerts import RecordingNotifier
from nightshift.core.manager import Manager, SpecError
from nightshift.core.process import pid_alive
from nightshift.core.store import Store
from nightshift.core.supervisor import SupervisorConfig

TRAIN = str(Path(__file__).resolve().parents[1] / "examples" / "synthetic" / "train.py")
FAST = SupervisorConfig(poll_s=0.05, lease_s=2.0, stop_grace_s=3.0)


class Diag:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def diagnose(self, ctx: dict) -> str:
        self.calls.append(ctx)
        return "learning rate too high for this stage of training"


@pytest.fixture
def env(tmp_path: Path):
    store = Store(tmp_path / "data" / "nightshift.db")
    notes = RecordingNotifier()
    diag = Diag()
    mgr = Manager(
        store, data_dir=tmp_path / "data", notifier=notes, diagnoser=diag, sup_cfg=FAST, housekeeping_s=0.05
    )
    work = tmp_path / "work"
    work.mkdir()
    yield mgr, store, notes, diag, work
    mgr.shutdown(wait=True)
    for run in store.list_runs():
        if run.get("pgid") and pid_alive(run["pid"]):
            os.killpg(run["pgid"], 9)


def _wait(store: Store, run_id: str, statuses: tuple[str, ...], timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = store.get_run(run_id)
        if run and run["status"] in statuses:
            return run
        time.sleep(0.05)
    raise AssertionError(f"run never reached {statuses}: {store.get_run(run_id)}")


def _cmd(*extra: str) -> list[str]:
    return [sys.executable, TRAIN, "--out", "out", "--sleep", "0.003", *extra]


def test_divergence_is_rolled_back_with_lower_lr_and_the_run_finishes(env) -> None:
    mgr, store, notes, diag, work = env
    card = mgr.propose(
        {
            "name": "tiny",
            "cwd": str(work),
            "command": _cmd("--lr", "{lr}", "--steps", "900", "--unstable-after", "600", "--resume={resume}"),
            "params": {"lr": 3e-3},
        }
    )
    assert card["status"] == "pending" and "--resume" not in card["command"]
    assert any("Automatic recovery" in line for line in card["playbook"])
    mgr.housekeeping()
    assert store.get_run(card["run_id"])["status"] == "pending"  # nothing runs without approval

    assert mgr.approve(card["run_id"])["ok"]
    mgr.start()
    run = _wait(store, card["run_id"], ("finished", "stopped", "failed"))
    assert run["status"] == "finished", run["note"]
    assert run["interventions"] == 1 and run["attempt"] == 1
    assert run["params"]["lr"] == pytest.approx(1.5e-3)
    assert run["restored_from"].endswith(("ckpt_600.json", "ckpt_500.json"))

    [rb] = [a for a in store.actions(card["run_id"]) if a["kind"] == "rollback_restart"]
    assert rb["ok"] and rb["details"]["checkpoint"]["step"] <= 600
    assert "lr 0.003 → 0.0015" in rb["message"]
    paged = [a for a in notes.alerts if a.page]
    assert len(paged) == 1, [a.title for a in paged]  # one buzz for one event, not one per detector
    [page] = paged
    assert page.kind in ("nan_inf", "divergence", "spike")  # usually caught as it climbs, before NaN
    assert page.details["decision"]["action"] == "rollback_restart"
    assert page.chart_png and page.chart_png.startswith(b"\x89PNG")
    assert "Why: learning rate too high" in page.body
    assert diag.calls and diag.calls[0]["log_tail"]
    assert notes.alerts[-1].kind == "finished" and not notes.alerts[-1].page
    # the restarted attempt's own metrics are healthy and stored separately
    attempts = {p["attempt"] for p in store.series(card["run_id"], "loss", 5000)}
    assert attempts == {0, 1}


def test_oom_restarts_with_half_the_batch(env) -> None:
    mgr, store, notes, _, work = env
    card = mgr.propose(
        {
            "name": "oomy",
            "cwd": str(work),
            "command": _cmd("--steps", "300", "--batch-size", "{batch_size}", "--grad-accum", "{grad_accum}",
                            "--oom-batch", "64", "--resume={resume}"),
            "params": {"batch_size": 64, "grad_accum": 1},
        }
    )  # fmt: skip
    mgr.approve(card["run_id"])
    mgr.start()
    run = _wait(store, card["run_id"], ("finished", "stopped", "failed"))
    assert run["status"] == "finished", run["note"]
    assert run["params"] == {"batch_size": 32, "grad_accum": 2}
    [act] = [a for a in store.actions(card["run_id"]) if a["kind"] == "oom_restart"]
    assert "batch_size 64 → 32" in act["message"] and "grad_accum 1 → 2" in act["message"]
    assert any(a.kind == "oom" and a.page for a in notes.alerts)


def test_budget_is_enforced_then_the_run_is_stopped(env) -> None:
    mgr, store, notes, _, work = env
    card = mgr.propose(
        {
            "name": "hopeless",
            "cwd": str(work),
            "command": _cmd("--lr", "{lr}", "--lr-critical", "1e-5", "--unstable-after", "150", "--steps", "3000",
                            "--ckpt-every", "50", "--resume={resume}"),
            "params": {"lr": 3e-3},
            "playbook": {"max_interventions": 2, "cooldown_s": 0},
        }
    )  # fmt: skip
    mgr.approve(card["run_id"])
    mgr.start()
    run = _wait(store, card["run_id"], ("finished", "stopped", "failed"))
    assert run["status"] == "stopped"
    assert run["interventions"] == 2
    assert "budget used up" in run["note"]
    assert not pid_alive(run["pid"])
    assert notes.alerts[-1].page


def test_cooldown_stops_a_run_that_fails_again_right_after_recovery(env) -> None:
    mgr, store, _, _, work = env
    card = mgr.propose(
        {
            "name": "relapse",
            "cwd": str(work),
            "command": _cmd("--lr", "{lr}", "--lr-critical", "1e-5", "--unstable-after", "150", "--steps", "3000",
                            "--ckpt-every", "50", "--resume={resume}"),
            "params": {"lr": 3e-3},
            "playbook": {"cooldown_s": 3600},
        }
    )  # fmt: skip
    mgr.approve(card["run_id"])
    mgr.start()
    run = _wait(store, card["run_id"], ("finished", "stopped", "failed"))
    assert run["status"] == "stopped" and run["interventions"] == 1
    assert "after the last recovery" in run["note"]


def test_no_resume_slot_means_stop_not_restart_from_scratch(env) -> None:
    mgr, store, _notes, _, work = env
    card = mgr.propose(
        {
            "name": "noresume",
            "cwd": str(work),
            "command": _cmd("--lr", "{lr}", "--steps", "3000", "--unstable-after", "200"),
            "params": {"lr": 3e-3},
        }
    )
    mgr.approve(card["run_id"])
    mgr.start()
    run = _wait(store, card["run_id"], ("finished", "stopped", "failed"))
    assert run["status"] == "stopped" and run["interventions"] == 0
    assert "{resume}" in run["note"]


def test_user_stop_kills_the_whole_group(env) -> None:
    mgr, store, _, _, work = env
    card = mgr.propose({"name": "long", "cwd": str(work), "command": _cmd("--steps", "100000")})
    mgr.approve(card["run_id"])
    mgr.start()
    run = _wait(store, card["run_id"], ("running",))
    assert mgr.request_stop(card["run_id"])["ok"]
    run = _wait(store, card["run_id"], ("stopped",))
    assert not pid_alive(run["pid"])
    assert store.actions(card["run_id"])[0]["message"] == "stopped at your request"


def test_crash_with_notify_only_marks_the_run_failed(env) -> None:
    mgr, store, notes, _, work = env
    card = mgr.propose(
        {"name": "crashy", "cwd": str(work), "command": _cmd("--crash-at", "30", "--steps", "500")}
    )
    mgr.approve(card["run_id"])
    mgr.start()
    run = _wait(store, card["run_id"], ("failed", "finished", "stopped"))
    assert run["status"] == "failed" and "dataloader" in run["note"]
    assert any(a.kind == "crash" and a.page for a in notes.alerts)


def test_another_process_adopts_a_run_after_a_crash(tmp_path: Path) -> None:
    """Hermes dies mid-run: the trainer keeps going and the next Hermes picks it up from the database."""
    store = Store(tmp_path / "db" / "n.db")
    work = tmp_path / "w"
    work.mkdir()
    a = Manager(
        store, data_dir=tmp_path / "db", sup_cfg=SupervisorConfig(poll_s=0.05, lease_s=0.5), owner="A"
    )
    card = a.propose(
        {"name": "survivor", "cwd": str(work), "command": _cmd("--lr", "{lr}", "--steps", "900",
                                                               "--unstable-after", "600", "--resume={resume}"),
         "params": {"lr": 3e-3}}
    )  # fmt: skip
    a.approve(card["run_id"])
    a.housekeeping()
    pid = store.get_run(card["run_id"])["pid"]
    a.shutdown(wait=True)  # "Hermes crashed": nobody is watching, the trainer runs on
    assert pid_alive(pid)

    b_store = Store(tmp_path / "db" / "n.db")
    b = Manager(
        b_store, data_dir=tmp_path / "db", sup_cfg=SupervisorConfig(poll_s=0.05, lease_s=0.5), owner="B"
    )
    time.sleep(0.6)  # A's lease lapses
    b.start()
    try:
        run = _wait(b_store, card["run_id"], ("finished", "stopped", "failed"))
        assert run["status"] == "finished" and run["interventions"] == 1 and run["owner"] in (None, "B")
    finally:
        b.shutdown(wait=True)


def test_two_managers_never_both_supervise(tmp_path: Path) -> None:
    store = Store(tmp_path / "n.db")
    work = tmp_path / "w"
    work.mkdir()
    a = Manager(store, data_dir=tmp_path, sup_cfg=FAST, owner="A")
    b = Manager(Store(tmp_path / "n.db"), data_dir=tmp_path, sup_cfg=FAST, owner="B")
    card = a.propose({"name": "one", "cwd": str(work), "command": _cmd("--steps", "100000")})
    a.approve(card["run_id"])
    a.housekeeping()
    b.housekeeping()
    try:
        assert card["run_id"] in a.supervisors and card["run_id"] not in b.supervisors
    finally:
        a.request_stop(card["run_id"])
        a.housekeeping()
        a.shutdown(wait=True)
        b.shutdown(wait=True)


def test_watch_mode_only_alerts(env, tmp_path: Path) -> None:
    mgr, store, notes, _, work = env
    log = work / "external.log"
    log.write_text("".join(f"step {i * 10}: loss {3.0 - i * 0.01:.4f}\n" for i in range(60)))
    res = mgr.watch({"name": "external", "cwd": str(work), "log_path": "external.log"})
    mgr.start()
    _wait(store, res["run_id"], ("running",))
    with log.open("a") as fh:
        fh.write("step 610: loss nan\n")
    deadline = time.time() + 10
    while time.time() < deadline and not any(a.kind == "nan_inf" for a in notes.alerts):
        time.sleep(0.05)
    [alert] = [a for a in notes.alerts if a.kind == "nan_inf"]
    assert "watched read-only" in str(alert.details["decision"]["reason"])
    assert store.get_run(res["run_id"])["status"] == "running"


def test_spec_validation(env) -> None:
    mgr, _, _, _, work = env
    with pytest.raises(SpecError, match="params has no value"):
        mgr.propose({"cwd": str(work), "command": _cmd("--lr", "{lr}")})
    with pytest.raises(SpecError, match="does not exist"):
        mgr.propose({"cwd": str(work / "nope"), "command": "python x.py"})
    with pytest.raises(SpecError, match="cannot find the program"):
        mgr.propose({"cwd": str(work), "command": "definitely-not-a-binary --x"})
    with pytest.raises(SpecError, match="empty command"):
        mgr.propose({"cwd": str(work), "command": ""})
    with pytest.raises(SpecError, match="invalid playbook"):
        mgr.propose({"cwd": str(work), "command": _cmd(), "playbook": {"rules": {"nan_inf": "explode"}}})
    with pytest.raises(SpecError, match="needs at least one source"):
        mgr.watch({"cwd": str(work)})
    with pytest.raises(SpecError, match="never controls processes"):
        mgr.watch({"cwd": str(work), "log_path": "x.log", "pid": 123})
    with pytest.raises(SpecError, match="each source needs"):
        mgr.watch({"cwd": str(work), "sources": [{"type": "wandb", "path": "x"}]})


def test_deny_and_approve_errors(env) -> None:
    mgr, store, _, _, work = env
    card = mgr.propose({"name": "x", "cwd": str(work), "command": _cmd()})
    assert mgr.deny(card["run_id"])["ok"]
    assert store.get_run(card["run_id"])["status"] == "cancelled"
    assert not mgr.approve(card["run_id"])["ok"]
    assert not mgr.approve("missing")["ok"]
    assert not mgr.request_stop("missing")["ok"]
    assert not mgr.request_stop(card["run_id"])["ok"]
    card2 = mgr.propose({"name": "y", "cwd": str(work), "command": _cmd()})
    assert mgr.request_stop(card2["run_id"])["status"] == "cancelled"  # stopping a proposal declines it
    assert math.isfinite(mgr.ack()["acknowledged"])


def test_forget_only_removes_finished_runs(env) -> None:
    mgr, store, _, _, _ = env
    done = store.create_run(name="old", mode="launch", status="finished")
    store.add_metrics(done, [])
    live = store.create_run(name="live", mode="launch", status="running")
    assert mgr.forget(live)["ok"] is False and store.get_run(live)
    assert mgr.forget(done) == {"ok": True, "forgotten": 1} and store.get_run(done) is None
    store.create_run(name="old2", mode="launch", status="stopped")
    assert mgr.forget("finished")["forgotten"] == 1
    assert [r["name"] for r in store.list_runs()] == ["live"]


def _poll_until(cond, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline and not cond():
        time.sleep(0.05)


def test_spike_in_the_middle_of_a_poll_is_caught(env, tmp_path: Path) -> None:
    """A fast trainer logs many lines between two polls: a blow-up followed by a high plateau must still be a
    spike, raised at the step it began (regression: only the last point of each poll used to be checked)."""
    mgr, store, _, _, work = env
    log = work / "fast.log"
    log.write_text("".join(f"step {i * 10}: loss {3.0 - i * 0.02:.4f}\n" for i in range(60)))
    res = mgr.watch({"name": "fast", "cwd": str(work), "log_path": "fast.log"})
    mgr.start()
    _wait(store, res["run_id"], ("running",))
    burst = [44.7, 206.9, 113.9, 188.5, 140.6, 119.8, 97.6, 87.8, 161.0, 98.7, 95.0, 90.0]
    with log.open("a") as fh:  # one write, so the whole burst lands in a single poll
        fh.write("".join(f"step {600 + i * 10}: loss {v}\n" for i, v in enumerate(burst)))
    _poll_until(lambda: any(i["kind"] == "spike" for i in store.incidents(res["run_id"])))
    [spike] = [i for i in store.incidents(res["run_id"]) if i["kind"] == "spike"]
    assert spike["severity"] == "critical" and spike["onset_step"] == 600


def test_a_warning_that_repeats_escalates_to_critical(env, tmp_path: Path) -> None:
    mgr, store, notes, _, work = env
    log = work / "slow.log"
    log.write_text("".join(f"step {i * 10}: loss {3.0 - i * 0.02:.4f}\n" for i in range(60)))
    res = mgr.watch({"name": "slow", "cwd": str(work), "log_path": "slow.log"})
    mgr.start()
    _wait(store, res["run_id"], ("running",))
    with log.open("a") as fh:
        fh.write("step 600: loss 40.0\n")
    _poll_until(lambda: any(i["kind"] == "spike" for i in store.incidents(res["run_id"])))
    with log.open("a") as fh:
        fh.write("step 610: loss 90.0\n")
    _poll_until(lambda: any(i["severity"] == "critical" for i in store.incidents(res["run_id"])))
    sev = [i["severity"] for i in store.incidents(res["run_id"]) if i["kind"] == "spike"]
    assert sorted(sev) == ["critical", "warn"], sev
    assert any(a.kind == "spike" and a.page for a in notes.alerts)  # the critical one pages
