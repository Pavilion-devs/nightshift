"""Durable state: runs, metrics, incidents and actions in one SQLite file (WAL mode).

Readers in other processes (the dashboard routes Desktop calls) see the same data, so nothing the UI shows
depends on in-memory state. One connection per thread; migrations are versioned and idempotent.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import Incident, MetricPoint

SCHEMA_VERSION = 1

_MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE IF NOT EXISTS runs (
        id            TEXT PRIMARY KEY,
        name          TEXT NOT NULL,
        mode          TEXT NOT NULL,              -- 'launch' (we own the process) | 'watch' (notify-only)
        status        TEXT NOT NULL,         -- pending|running|recovering|stopped|finished|failed
        cwd           TEXT,
        argv_json     TEXT,                       -- approved command template
        params_json   TEXT,                       -- current template params (lr, batch_size, ...)
        playbook_json TEXT,
        sources_json  TEXT,
        detector_json TEXT,
        log_path      TEXT,
        pid           INTEGER,
        pgid          INTEGER,
        proc_id       TEXT,                       -- start time + command, guards against pid reuse
        exit_code     INTEGER,
        interventions INTEGER NOT NULL DEFAULT 0,
        created_at    REAL NOT NULL,
        started_at    REAL,
        updated_at    REAL NOT NULL,
        ended_at      REAL,
        origin_json   TEXT                        -- where the run was requested from, for alerts
    );
    CREATE TABLE IF NOT EXISTS metrics (
        run_id  TEXT NOT NULL,
        name    TEXT NOT NULL,
        step    INTEGER NOT NULL,
        value   REAL,                             -- NULL encodes NaN; +-1e308 clamp encodes +-inf
        wall    REAL NOT NULL,
        attempt INTEGER NOT NULL DEFAULT 0        -- bumps on every restart so the chart can show segments
    );
    CREATE INDEX IF NOT EXISTS metrics_run_name_step ON metrics(run_id, name, step);
    CREATE TABLE IF NOT EXISTS incidents (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      TEXT NOT NULL,
        kind        TEXT NOT NULL,
        severity    TEXT NOT NULL,
        metric      TEXT,
        step        INTEGER,
        onset_step  INTEGER,
        message     TEXT NOT NULL,
        details_json TEXT,
        diagnosis   TEXT,
        at          REAL NOT NULL,
        acked       INTEGER NOT NULL DEFAULT 0,
        resolved_at REAL
    );
    CREATE INDEX IF NOT EXISTS incidents_run ON incidents(run_id, at);
    CREATE TABLE IF NOT EXISTS actions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      TEXT NOT NULL,
        incident_id INTEGER,
        kind        TEXT NOT NULL,
        ok          INTEGER NOT NULL,
        message     TEXT NOT NULL,
        details_json TEXT,
        at          REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS actions_run ON actions(run_id, at);
    """,
}

_RUN_JSON_FIELDS = ("argv", "params", "playbook", "sources", "detector", "origin")
_INF = 1e308


def _enc(v: float) -> float | None:
    if math.isnan(v):
        return None
    if math.isinf(v):
        return _INF if v > 0 else -_INF
    return v


def _dec(v: float | None) -> float:
    if v is None:
        return math.nan
    if v >= _INF:
        return math.inf
    if v <= -_INF:
        return -math.inf
    return v


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._migrate()

    # -- connection ---------------------------------------------------------------------------------------
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA busy_timeout=10000")
            self._local.conn = c
        return c

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None

    def _migrate(self) -> None:
        c = self.conn()
        c.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        row = c.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        current = row["v"] or 0
        for version in sorted(v for v in _MIGRATIONS if v > current):
            c.execute("BEGIN IMMEDIATE")
            try:
                # re-check inside the write lock: another process may have migrated meanwhile
                again = c.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()["v"] or 0
                if again < version:
                    for stmt in _split_sql(_MIGRATIONS[version]):
                        c.execute(stmt)
                    c.execute("INSERT INTO schema_version(version) VALUES (?)", (version,))
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise

    def schema_version(self) -> int:
        return self.conn().execute("SELECT MAX(version) AS v FROM schema_version").fetchone()["v"] or 0

    # -- runs ---------------------------------------------------------------------------------------------
    def create_run(self, *, name: str, mode: str, status: str = "pending", **fields: Any) -> str:
        run_id = fields.pop("id", None) or uuid.uuid4().hex[:8]
        now = time.time()
        row: dict[str, Any] = {
            "id": run_id,
            "name": name,
            "mode": mode,
            "status": status,
            "created_at": now,
            "updated_at": now,
        }
        row.update(self._encode_run_fields(fields))
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        self.conn().execute(f"INSERT INTO runs ({cols}) VALUES ({marks})", tuple(row.values()))
        return run_id

    def update_run(self, run_id: str, **fields: Any) -> None:
        row = self._encode_run_fields(fields)
        row["updated_at"] = time.time()
        sets = ", ".join(f"{k} = ?" for k in row)
        self.conn().execute(f"UPDATE runs SET {sets} WHERE id = ?", (*row.values(), run_id))

    def bump_interventions(self, run_id: str) -> int:
        c = self.conn()
        c.execute(
            "UPDATE runs SET interventions = interventions + 1, updated_at = ? WHERE id = ?",
            (time.time(), run_id),
        )
        return c.execute("SELECT interventions FROM runs WHERE id = ?", (run_id,)).fetchone()["interventions"]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.conn().execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return self._decode_run(row) if row else None

    def list_runs(self, statuses: Iterable[str] | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if statuses:
            st = list(statuses)
            marks = ", ".join("?" for _ in st)
            q = f"SELECT * FROM runs WHERE status IN ({marks}) ORDER BY created_at DESC LIMIT ?"
            rows = self.conn().execute(q, (*st, limit)).fetchall()
        else:
            rows = (
                self.conn()
                .execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
                .fetchall()
            )
        return [self._decode_run(r) for r in rows]

    @staticmethod
    def _encode_run_fields(fields: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in fields.items():
            if k in _RUN_JSON_FIELDS:
                out[f"{k}_json"] = json.dumps(v) if v is not None else None
            else:
                out[k] = v
        return out

    @staticmethod
    def _decode_run(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for k in _RUN_JSON_FIELDS:
            raw = d.pop(f"{k}_json", None)
            d[k] = json.loads(raw) if raw else None
        return d

    # -- metrics ------------------------------------------------------------------------------------------
    def add_metrics(self, run_id: str, points: Iterable[MetricPoint], attempt: int = 0) -> int:
        rows = [(run_id, p.name, p.step, _enc(p.value), p.wall, attempt) for p in points]
        if rows:
            self.conn().executemany(
                "INSERT INTO metrics (run_id, name, step, value, wall, attempt) VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def metric_names(self, run_id: str) -> list[str]:
        rows = (
            self.conn()
            .execute("SELECT DISTINCT name FROM metrics WHERE run_id = ? ORDER BY name", (run_id,))
            .fetchall()
        )
        return [r["name"] for r in rows]

    def series(self, run_id: str, name: str, max_points: int = 600) -> list[dict[str, Any]]:
        """The metric over time, downsampled to ``max_points`` while keeping every spike and NaN visible."""
        rows = (
            self.conn()
            .execute(
                "SELECT step, value, wall, attempt FROM metrics WHERE run_id = ? AND name = ? "
                "ORDER BY wall, step",
                (run_id, name),
            )
            .fetchall()
        )
        pts = [
            {"step": r["step"], "value": _dec(r["value"]), "wall": r["wall"], "attempt": r["attempt"]}
            for r in rows
        ]
        return downsample(pts, max_points)

    def latest_metrics(self, run_id: str) -> dict[str, dict[str, Any]]:
        rows = (
            self.conn()
            .execute(
                """SELECT m.name, m.step, m.value, m.wall FROM metrics m
               JOIN (SELECT name, MAX(rowid) AS rid FROM metrics WHERE run_id = ? GROUP BY name) last
               ON m.rowid = last.rid""",
                (run_id,),
            )
            .fetchall()
        )
        return {r["name"]: {"step": r["step"], "value": _dec(r["value"]), "wall": r["wall"]} for r in rows}

    def load_points(self, run_id: str, attempt: int | None = None, limit: int = 10_000) -> list[MetricPoint]:
        """Recent points in arrival order, for rebuilding a detector window after a restart."""
        if attempt is None:
            q = "SELECT * FROM (SELECT rowid, * FROM metrics WHERE run_id = ? ORDER BY rowid DESC LIMIT ?)"
            rows = self.conn().execute(q + " ORDER BY rowid", (run_id, limit)).fetchall()
        else:
            q = (
                "SELECT * FROM (SELECT rowid, * FROM metrics WHERE run_id = ? AND attempt = ? "
                "ORDER BY rowid DESC LIMIT ?)"
            )
            rows = self.conn().execute(q + " ORDER BY rowid", (run_id, attempt, limit)).fetchall()
        return [MetricPoint(r["step"], r["name"], _dec(r["value"]), r["wall"]) for r in rows]

    def prune_metrics(self, run_id: str, keep_last: int = 200_000) -> int:
        c = self.conn()
        cur = c.execute(
            """DELETE FROM metrics WHERE run_id = ? AND rowid NOT IN
               (SELECT rowid FROM metrics WHERE run_id = ? ORDER BY rowid DESC LIMIT ?)""",
            (run_id, run_id, keep_last),
        )
        return cur.rowcount

    # -- incidents & actions ------------------------------------------------------------------------------
    def add_incident(self, run_id: str, inc: Incident, at: float | None = None) -> int:
        cur = self.conn().execute(
            """INSERT INTO incidents
               (run_id, kind, severity, metric, step, onset_step, message, details_json, at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                inc.kind.value,
                inc.severity.value,
                inc.metric,
                inc.step,
                inc.onset_step,
                inc.message,
                json.dumps(inc.details, default=str),
                at if at is not None else time.time(),
            ),
        )
        return int(cur.lastrowid)

    def set_diagnosis(self, incident_id: int, text: str) -> None:
        self.conn().execute("UPDATE incidents SET diagnosis = ? WHERE id = ?", (text, incident_id))

    def resolve_incident(self, incident_id: int, at: float | None = None) -> None:
        self.conn().execute(
            "UPDATE incidents SET resolved_at = ? WHERE id = ? AND resolved_at IS NULL",
            (at if at is not None else time.time(), incident_id),
        )

    def ack_incidents(self, run_id: str | None = None) -> int:
        if run_id:
            cur = self.conn().execute(
                "UPDATE incidents SET acked = 1 WHERE run_id = ? AND acked = 0", (run_id,)
            )
        else:
            cur = self.conn().execute("UPDATE incidents SET acked = 1 WHERE acked = 0")
        return cur.rowcount

    def incidents(
        self,
        run_id: str | None = None,
        *,
        unacked_only: bool = False,
        since: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        where, args = [], []
        if run_id:
            where.append("run_id = ?")
            args.append(run_id)
        if unacked_only:
            where.append("acked = 0")
        if since is not None:
            where.append("at >= ?")
            args.append(since)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        rows = (
            self.conn()
            .execute(f"SELECT * FROM incidents {clause} ORDER BY at DESC LIMIT ?", (*args, limit))
            .fetchall()
        )
        out = []
        for r in rows:
            d = dict(r)
            d["details"] = json.loads(d.pop("details_json") or "{}")
            out.append(d)
        return out

    def add_action(
        self,
        run_id: str,
        kind: str,
        ok: bool,
        message: str,
        *,
        incident_id: int | None = None,
        details: dict[str, Any] | None = None,
        at: float | None = None,
    ) -> int:
        cur = self.conn().execute(
            """INSERT INTO actions (run_id, incident_id, kind, ok, message, details_json, at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                incident_id,
                kind,
                int(ok),
                message,
                json.dumps(details or {}, default=str),
                at if at is not None else time.time(),
            ),
        )
        return int(cur.lastrowid)

    def actions(
        self, run_id: str | None = None, *, since: float | None = None, limit: int = 100
    ) -> list[dict]:
        where, args = [], []
        if run_id:
            where.append("run_id = ?")
            args.append(run_id)
        if since is not None:
            where.append("at >= ?")
            args.append(since)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        rows = (
            self.conn()
            .execute(f"SELECT * FROM actions {clause} ORDER BY at DESC LIMIT ?", (*args, limit))
            .fetchall()
        )
        out = []
        for r in rows:
            d = dict(r)
            d["ok"] = bool(d["ok"])
            d["details"] = json.loads(d.pop("details_json") or "{}")
            out.append(d)
        return out


def _split_sql(script: str) -> list[str]:
    """Split a migration into statements (comments may contain ';'); ``executescript`` would COMMIT."""
    stmts, buf = [], ""
    for line in script.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            if buf.strip():
                stmts.append(buf.strip())
            buf = ""
    if buf.strip():
        stmts.append(buf.strip())
    return stmts


def downsample(points: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    """Bucket the series and keep each bucket's first, min, max and any non-finite point, in order.

    Plain striding hides exactly what this product exists to show (a one-point spike, a NaN), so extremes and
    non-finite values always survive.
    """
    n = len(points)
    if n <= max_points or max_points < 4:
        return points
    buckets = max(1, max_points // 3)
    size = math.ceil(n / buckets)
    keep: set[int] = {0, n - 1}
    for start in range(0, n, size):
        idx = range(start, min(n, start + size))
        finite = [i for i in idx if math.isfinite(points[i]["value"])]
        keep.add(start)
        for i in idx:
            if not math.isfinite(points[i]["value"]):
                keep.add(i)
        if finite:
            keep.add(min(finite, key=lambda i: points[i]["value"]))
            keep.add(max(finite, key=lambda i: points[i]["value"]))
    return [points[i] for i in sorted(keep)]
