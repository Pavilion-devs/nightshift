from __future__ import annotations

import math
import threading
from pathlib import Path

from nightshift.core.models import Incident, IncidentKind, MetricPoint, Severity
from nightshift.core.store import SCHEMA_VERSION, Store, downsample


def test_migrations_are_idempotent(tmp_path: Path) -> None:
    s = Store(tmp_path / "ns.db")
    assert s.schema_version() == SCHEMA_VERSION
    s.close()
    s2 = Store(tmp_path / "ns.db")  # second open must not re-run or fail
    assert s2.schema_version() == SCHEMA_VERSION
    rows = s2.conn().execute("SELECT COUNT(*) AS n FROM schema_version").fetchone()["n"]
    assert rows == SCHEMA_VERSION


def test_runs_roundtrip_json_fields(tmp_path: Path) -> None:
    s = Store(tmp_path / "ns.db")
    rid = s.create_run(
        name="tiny", mode="launch", argv=["python", "train.py", "--lr", "{lr}"], params={"lr": 3e-3}
    )
    run = s.get_run(rid)
    assert run["argv"][-1] == "{lr}" and run["params"] == {"lr": 3e-3} and run["status"] == "pending"
    s.update_run(rid, status="running", pid=123, params={"lr": 1.5e-3})
    assert s.get_run(rid)["params"]["lr"] == 1.5e-3
    assert s.bump_interventions(rid) == 1 and s.bump_interventions(rid) == 2
    assert [r["id"] for r in s.list_runs(["running"])] == [rid]
    assert s.list_runs(["finished"]) == []
    assert len(s.list_runs()) == 1
    assert s.get_run("missing") is None


def test_metrics_preserve_nan_and_inf(tmp_path: Path) -> None:
    s = Store(tmp_path / "ns.db")
    rid = s.create_run(name="r", mode="watch")
    pts = [
        MetricPoint(1, "loss", 2.0, 1.0),
        MetricPoint(2, "loss", math.nan, 2.0),
        MetricPoint(3, "loss", math.inf, 3.0),
    ]
    assert s.add_metrics(rid, pts, attempt=1) == 3
    assert s.add_metrics(rid, []) == 0
    series = s.series(rid, "loss")
    assert series[0]["value"] == 2.0 and math.isnan(series[1]["value"]) and math.isinf(series[2]["value"])
    assert series[0]["attempt"] == 1
    assert s.metric_names(rid) == ["loss"]
    latest = s.latest_metrics(rid)
    assert math.isinf(latest["loss"]["value"]) and latest["loss"]["step"] == 3
    loaded = s.load_points(rid)
    assert [p.step for p in loaded] == [1, 2, 3]
    assert s.load_points(rid, attempt=0) == []
    assert len(s.load_points(rid, attempt=1, limit=2)) == 2


def test_negative_inf_and_prune(tmp_path: Path) -> None:
    s = Store(tmp_path / "ns.db")
    rid = s.create_run(name="r", mode="watch")
    s.add_metrics(rid, [MetricPoint(i, "loss", -math.inf if i == 0 else 1.0, float(i)) for i in range(10)])
    assert s.series(rid, "loss")[0]["value"] == -math.inf
    assert s.prune_metrics(rid, keep_last=4) == 6
    assert [p.step for p in s.load_points(rid)] == [6, 7, 8, 9]


def test_incidents_and_actions(tmp_path: Path) -> None:
    s = Store(tmp_path / "ns.db")
    rid = s.create_run(name="r", mode="launch")
    inc = Incident(
        IncidentKind.NAN_INF, Severity.CRITICAL, "loss is nan", metric="loss", step=640, onset_step=620
    )
    iid = s.add_incident(rid, inc, at=100.0)
    s.set_diagnosis(iid, "lr too high")
    [row] = s.incidents(rid)
    assert row["kind"] == "nan_inf" and row["diagnosis"] == "lr too high" and row["onset_step"] == 620
    assert s.incidents(rid, unacked_only=True)
    assert s.ack_incidents(rid) == 1 and s.incidents(rid, unacked_only=True) == []
    s.add_incident(rid, inc, at=200.0)
    assert s.ack_incidents() == 1
    assert len(s.incidents(since=150.0)) == 1 and len(s.incidents()) == 2
    s.resolve_incident(iid, at=300.0)
    assert next(r for r in s.incidents() if r["id"] == iid)["resolved_at"] == 300.0
    aid = s.add_action(
        rid, "rollback_restart", True, "restored ckpt_600", incident_id=iid, details={"lr": 1.5e-3}
    )
    [act] = s.actions(rid)
    assert act["id"] == aid and act["ok"] is True and act["details"] == {"lr": 1.5e-3}
    assert s.actions(since=act["at"] + 1) == []
    assert len(s.actions()) == 1


def test_one_connection_per_thread(tmp_path: Path) -> None:
    s = Store(tmp_path / "ns.db")
    rid = s.create_run(name="r", mode="watch")
    errors: list[Exception] = []

    def writer(k: int) -> None:
        try:
            s.add_metrics(rid, [MetricPoint(i, f"m{k}", float(i), float(i)) for i in range(200)])
        except Exception as e:  # pragma: no cover - only on failure
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(s.metric_names(rid)) == 4


def test_downsample_keeps_spikes_and_nans() -> None:
    pts = [{"step": i, "value": 1.0, "wall": float(i)} for i in range(10_000)]
    pts[4321]["value"] = 50.0
    pts[7777]["value"] = math.nan
    out = downsample(pts, 300)
    assert len(out) <= 320
    assert any(p["value"] == 50.0 for p in out)
    assert any(math.isnan(p["value"]) for p in out)
    assert out[0]["step"] == 0 and out[-1]["step"] == 9999
    assert downsample(pts[:10], 300) == pts[:10]
