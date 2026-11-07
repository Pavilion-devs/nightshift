"""The engine facade: proposals, approvals, launches, re-attachment and user commands.

Every user decision goes through SQLite (``runs.status`` and the ``commands`` table), so the Desktop pane, a
slash command and an agent tool all take effect the same way, from whichever process they ran in. One
housekeeping thread per Hermes process picks the decisions up; leases make sure exactly one process acts.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from .alerts import Diagnoser, Notifier, NullDiagnoser, NullNotifier
from .clock import Clock, SystemClock
from .detectors import DetectorConfig
from .playbook import Playbook
from .process import (
    RESUME,
    ProcessHandle,
    TemplateError,
    parse_command,
    placeholders,
    render_argv,
    template_env,
)
from .report import alert_for_status
from .store import Store
from .supervisor import TERMINAL, RunSupervisor, SupervisorConfig

log = logging.getLogger("nightshift")

ACTIVE = ("starting", "running", "recovering")
_SOURCE_TYPES = ("log", "jsonl", "csv", "tfevents")


class SpecError(ValueError):
    pass


class Manager:
    def __init__(
        self,
        store: Store,
        *,
        data_dir: str | os.PathLike[str],
        clock: Clock | None = None,
        notifier: Notifier | None = None,
        diagnoser: Diagnoser | None = None,
        owner: str | None = None,
        sup_cfg: SupervisorConfig | None = None,
        housekeeping_s: float = 1.0,
    ) -> None:
        self.store = store
        self.data_dir = Path(data_dir)
        self.clock = clock or SystemClock()
        self.notifier = notifier or NullNotifier()
        self.diagnoser = diagnoser or NullDiagnoser()
        self.owner = owner or f"{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.sup_cfg = sup_cfg or SupervisorConfig()
        self.housekeeping_s = housekeeping_s
        self.supervisors: dict[str, RunSupervisor] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- proposals (agent) ------------------------------------------------------------------------------
    def propose(self, spec: dict[str, Any], *, origin: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate a launch request and park it as ``pending``. Nothing runs until a human approves it."""
        clean = self._validate(spec, launch=True)
        run_id = self.store.create_run(status="pending", mode="launch", origin=origin, **clean)
        return self.approval_card(run_id)

    def watch(self, spec: dict[str, Any], *, origin: dict[str, Any] | None = None) -> dict[str, Any]:
        """Attach read-only to a run started elsewhere. Reads files only, never touches a process."""
        clean = self._validate(spec, launch=False)
        run_id = self.store.create_run(status="approved", mode="watch", origin=origin, **clean)
        return {"ok": True, "run_id": run_id, "mode": "watch", "note": "watching read-only; alerts only"}

    def approval_card(self, run_id: str) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        pb = Playbook.from_dict(run.get("playbook"))
        first = render_argv(run["argv"], run.get("params") or {}, resume=None)
        return {
            "ok": True,
            "run_id": run_id,
            "status": run["status"],
            "name": run["name"],
            "cwd": run["cwd"],
            "command": " ".join(first),
            "command_template": run["argv"],
            "params": run.get("params") or {},
            "sources": run.get("sources") or [{"type": "log", "path": "(the process's own output)"}],
            "playbook": pb.describe(run.get("params") or {}),
            "approve_with": f"/nightshift approve {run_id}",
        }

    def _validate(self, spec: dict[str, Any], *, launch: bool) -> dict[str, Any]:
        name = str(spec.get("name") or "").strip()
        cwd = os.path.abspath(os.path.expanduser(str(spec.get("cwd") or os.getcwd())))
        if not os.path.isdir(cwd):
            raise SpecError(f"working directory does not exist: {cwd}")
        sources = []
        for s in spec.get("sources") or []:
            if not isinstance(s, dict) or s.get("type") not in _SOURCE_TYPES or not s.get("path"):
                raise SpecError(f"each source needs a type in {_SOURCE_TYPES} and a path: {s!r}")
            sources.append({k: s[k] for k in ("type", "path", "pattern") if k in s})
        try:
            playbook = Playbook.from_dict(spec.get("playbook")).to_dict()
        except (TypeError, ValueError) as exc:
            raise SpecError(f"invalid playbook: {exc}") from exc
        if spec.get("checkpoint_patterns"):
            playbook["checkpoint_patterns"] = list(spec["checkpoint_patterns"])
        detector = {**DetectorConfig.from_dict(spec.get("detector")).__dict__}
        out: dict[str, Any] = {"cwd": cwd, "sources": sources, "playbook": playbook, "detector": detector}

        if not launch:
            if not sources and not spec.get("log_path"):
                raise SpecError("watching needs at least one source (a log, JSONL, CSV or TensorBoard path)")
            log_path = spec.get("log_path")
            if log_path:
                lp = os.path.expanduser(str(log_path))
                out["log_path"] = lp if os.path.isabs(lp) else os.path.join(cwd, lp)
            if spec.get("pid"):
                raise SpecError("watch mode never controls processes; leave out pid")
            out["name"] = name or Path(cwd).name
            return out

        try:
            argv = parse_command(spec.get("command") or "")
        except (TemplateError, ValueError) as exc:
            raise SpecError(str(exc)) from exc
        params = dict(spec.get("params") or {})
        try:
            needed = placeholders(argv) - {RESUME}
        except TemplateError as exc:
            raise SpecError(str(exc)) from exc
        missing = needed - params.keys()
        if missing:
            raise SpecError(f"command uses {{{', '.join(sorted(missing))}}} but params has no value for it")
        exe = argv[0]
        if not (shutil.which(exe) or os.path.exists(os.path.join(cwd, exe))):
            raise SpecError(f"cannot find the program {exe!r} (from {cwd})")
        out.update({"argv": argv, "params": params, "name": name or Path(cwd).name})
        return out

    # -- human decisions (slash command, Desktop, CLI) ---------------------------------------------------
    def approve(self, run_id: str) -> dict[str, Any]:
        if self.store.transition(run_id, ["pending"], "approved"):
            return {
                "ok": True,
                "run_id": run_id,
                "status": "approved",
                "note": "launching within a second or two",
            }
        run = self.store.get_run(run_id)
        return {
            "ok": False,
            "error": f"no pending run {run_id!r}" if run is None else f"run is {run['status']}",
        }

    def deny(self, run_id: str) -> dict[str, Any]:
        ok = self.store.transition(
            run_id, ["pending"], "cancelled", note="declined", ended_at=self.clock.now()
        )
        return {"ok": ok, "run_id": run_id, "status": "cancelled" if ok else None}

    def request_stop(self, run_id: str) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        if run is None:
            return {"ok": False, "error": f"no run {run_id!r}"}
        if run["status"] == "pending":
            return self.deny(run_id)
        if run["status"] in TERMINAL:
            return {"ok": False, "error": f"run is already {run['status']}"}
        self.store.add_command(run_id, "stop")
        return {"ok": True, "run_id": run_id, "note": "stopping within a second or two"}

    def ack(self, run_id: str | None = None) -> dict[str, Any]:
        return {"ok": True, "acknowledged": self.store.ack_incidents(run_id)}

    # -- background ------------------------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="nightshift:manager", daemon=True)
            self._thread.start()

    def shutdown(self, wait: bool = False) -> None:
        """Stop watching. Training processes keep running; another Hermes process can pick them up."""
        self._stop.set()
        with self._lock:
            sups = list(self.supervisors.values())
        for sup in sups:
            sup.stop_watching()
        if wait:
            for sup in sups:
                sup.join(5)
            if self._thread:
                self._thread.join(5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.housekeeping()
            except Exception:
                log.exception("nightshift: housekeeping failed")
            self._stop.wait(self.housekeeping_s)
        self.store.close()

    def housekeeping(self) -> None:
        """One pass: launch approved runs, adopt orphaned ones, apply user commands, forget finished ones."""
        lease = self.sup_cfg.lease_s
        for run in self.store.list_runs(["approved"], limit=50):
            if self.store.claim(run["id"], self.owner, lease) and self.store.transition(
                run["id"], ["approved"], "starting"
            ):
                self._start_run(run["id"])
        with self._lock:
            mine = set(self.supervisors)
        for run in self.store.list_runs(ACTIVE, limit=200):
            if run["id"] not in mine and self.store.claim(run["id"], self.owner, lease):
                log.info("nightshift: adopting run %s (%s)", run["id"], run["status"])
                self._supervise(run["id"], handle=None)
        with self._lock:
            sups = dict(self.supervisors)
        for cmd in self.store.pending_commands(sups):
            sup = sups[cmd["run_id"]]
            if cmd["kind"] == "stop":
                result = sup.request_stop("stopped at your request")
            elif cmd["kind"] == "ack":
                result = f"acknowledged {self.store.ack_incidents(cmd['run_id'])}"
            else:
                result = f"unknown command {cmd['kind']!r}"
            self.store.finish_command(cmd["id"], result)
        with self._lock:
            for rid, sup in list(self.supervisors.items()):
                if sup.done:
                    del self.supervisors[rid]

    def _start_run(self, run_id: str) -> None:
        run = self.store.get_run(run_id)
        assert run is not None
        if run["mode"] == "watch":
            self.store.update_run(run_id, status="running", started_at=self.clock.now())
            self._supervise(run_id, handle=None)
            return
        log_path = str(self.data_dir / "runs" / run_id / "attempt-0.log")
        try:
            argv = render_argv(run["argv"], run.get("params") or {}, resume=None)
            handle = ProcessHandle.launch(
                argv,
                cwd=run["cwd"],
                log_path=log_path,
                env_extra=template_env(run.get("params") or {}, None, run_id, 0),
            )
        except Exception as exc:
            log.exception("nightshift: launch failed for %s", run_id)
            self.store.update_run(
                run_id, status="failed", note=f"launch failed: {exc}", ended_at=self.clock.now()
            )
            self.notifier.send(
                alert_for_status(self.store, self.store.get_run(run_id), "failed", f"launch failed: {exc}")
            )
            return
        self.store.update_run(
            run_id,
            status="running",
            pid=handle.pid,
            pgid=handle.pgid,
            proc_id=handle.proc_id,
            log_path=log_path,
            started_at=self.clock.now(),
            note="attempt 0",
        )
        self._supervise(run_id, handle=handle)

    def _supervise(self, run_id: str, handle: ProcessHandle | None) -> None:
        cfg = SupervisorConfig(**{**self.sup_cfg.__dict__, "logs_dir": str(self.data_dir / "runs" / run_id)})
        sup = RunSupervisor(
            run_id,
            self.store,
            clock=self.clock,
            notifier=self.notifier,
            diagnoser=self.diagnoser,
            owner=self.owner,
            cfg=cfg,
            handle=handle,
        )
        with self._lock:
            self.supervisors[run_id] = sup
        sup.start()
