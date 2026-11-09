"""The data behind the Nightshift screen: banner counts, signals, the run's story, the incidents table."""

from __future__ import annotations

from pathlib import Path

from nightshift.core.board import board, signals
from nightshift.core.models import Incident, IncidentKind, MetricPoint, Severity
from nightshift.core.store import Store

T0 = 1_800_000_000.0


def _rolled_back_run(store: Store) -> str:
    rid = store.create_run(name="tinygpt", mode="launch", status="finished", cwd="/w",
                           params={"lr": 0.0015, "batch_size": 32}, started_at=T0, ended_at=T0 + 900)  # fmt: skip
    healthy = [MetricPoint(s, "loss", 3.0 - s / 400, T0 + s / 10) for s in range(10, 610, 10)]
    healthy += [MetricPoint(s, "grad_norm", 0.4, T0 + s / 10) for s in range(10, 610, 10)]
    healthy += [MetricPoint(s, "lr", 0.003, T0 + s / 10) for s in range(10, 610, 10)]
    burst = [MetricPoint(610, "loss", 44.7, T0 + 61), MetricPoint(620, "loss", 206.9, T0 + 62),
             MetricPoint(610, "grad_norm", 14.0, T0 + 61), MetricPoint(620, "grad_norm", 68.8, T0 + 62)]  # fmt: skip
    store.add_metrics(rid, healthy + burst, attempt=0)
    after = [MetricPoint(s, "loss", 1.6 - s / 10_000, T0 + 100 + s / 10) for s in range(600, 1510, 10)]
    store.add_metrics(rid, after, attempt=1)
    inc = Incident(IncidentKind.SPIKE, Severity.CRITICAL, "loss spiked to 206.9 at step 620", metric="loss",
                   step=620, value=206.9, onset_step=610)  # fmt: skip
    iid = store.add_incident(rid, inc, at=T0 + 63)
    store.add_action(rid, "rollback_restart", True, "rolled back to step 600, lr 0.003 → 0.0015", incident_id=iid,
                     details={"changes": {"lr": [0.003, 0.0015]}, "checkpoint": {"step": 600}}, at=T0 + 64)  # fmt: skip
    store.resolve_incident(iid, at=T0 + 70)
    return rid


def test_board_tells_the_story_of_a_rolled_back_run(tmp_path: Path) -> None:
    store = Store(tmp_path / "n.db")
    _rolled_back_run(store)
    b = board(store, now=T0 + 1000)
    assert b["banner"] == {"incidents": 1, "recovered": 1, "attention": 0, "pending": 0, "window_h": 24}
    [card] = b["runs"]
    assert card["markers"] == [{"step": 610, "kind": "spike", "severity": "critical"}]
    assert card["recoveries"][0]["step"] == 600
    assert {p["attempt"] for p in card["series"]["train"]} == {0, 1} and card["series"]["lr"]
    top = card["signals"][0]
    assert top["metric"] == "grad_norm" and top["text"] == "172×" and top["frac"] == 1.0
    assert "loss jumped to 207" in card["summary"] and "rolled back to step 600" in card["summary"]
    events = [(e["event"], e["status"]) for e in card["log"]]
    assert events == [("Run launched", "success"), ("Loss spike", "failed"), ("Rolled back", "warning"),
                      ("Run finished", "success")]  # fmt: skip
    assert card["log"][0]["detail"].startswith("lr 0.003")  # the parameters it was launched with
    [row] = b["table"]
    assert row["state"] == "recovered" and row["run"] == "tinygpt" and row["step"] == 620


def test_old_incidents_leave_the_banner_and_table(tmp_path: Path) -> None:
    store = Store(tmp_path / "n.db")
    _rolled_back_run(store)
    b = board(store, now=T0 + 3 * 86_400)
    assert b["banner"]["incidents"] == 0 and b["table"] == []
    assert b["runs"]  # the latest runs stay on screen even when nothing happened recently


def test_signals_need_a_baseline_and_ignore_val_and_settings(tmp_path: Path) -> None:
    store = Store(tmp_path / "n.db")
    rid = store.create_run(name="x", mode="watch", status="running", cwd="/w")
    store.add_metrics(rid, [MetricPoint(10, "loss", 2.0, T0), MetricPoint(20, "loss", 50.0, T0 + 1)])
    inc = {"onset_step": 20, "step": 20, "at": T0 + 2}
    assert signals(store, rid, inc, ["loss", "val_loss", "batch_size"]) == []  # two points is no baseline


def test_unresolved_critical_needs_attention_and_info_is_not_counted(tmp_path: Path) -> None:
    store = Store(tmp_path / "n.db")
    rid = store.create_run(name="x", mode="launch", status="running", cwd="/w", started_at=T0)
    store.add_incident(
        rid, Incident(IncidentKind.STALL, Severity.CRITICAL, "no new metrics for 9 min"), at=T0 + 5
    )
    store.add_incident(rid, Incident(IncidentKind.PLATEAU, Severity.INFO, "loss has plateaued"), at=T0 + 6)
    b = board(store, now=T0 + 10)
    assert b["banner"]["attention"] == 1 and b["unacknowledged_incidents"] == 1
    assert [r["state"] for r in b["table"]] == ["new"] and b["runs"][0]["incidents"][0]["metric"] == "process"
