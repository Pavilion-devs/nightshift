"""One supervisor per run: poll metrics, detect, decide, act, tell the user. Runs on its own thread.

Order of operations on an incident is deliberate: act first (every second a diverged run keeps going burns GPU),
then diagnose (an LLM call may take a while), then notify with both.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .alerts import Alert, Diagnoser, Notifier
from .checkpoints import DEFAULT_PATTERNS, Checkpoint, discover, select_rollback
from .clock import Clock
from .detectors import DetectorConfig, classify_exit, detect_gpu_idle, detect_stall, run_window_detectors
from .models import Incident, IncidentKind, MetricWindow, Severity, is_safe_name
from .playbook import (
    NOTIFY,
    OOM_RESTART,
    RESTARTS,
    ROLLBACK_RESTART,
    STOP,
    Decision,
    Playbook,
    RunState,
    decide,
)
from .process import (
    RESUME,
    ProcessHandle,
    gpu_utilization,
    placeholders,
    read_tail,
    render_argv,
    template_env,
)
from .redact import redact_lines
from .report import alert_for_incident, alert_for_status, chart_for_run
from .sources import LogTailSource, MetricSource, make_source
from .store import Store

log = logging.getLogger("nightshift")

TERMINAL = ("finished", "stopped", "failed", "cancelled")


@dataclass
class SupervisorConfig:
    poll_s: float = 2.0
    lease_s: float = 30.0
    gpu_sample_s: float = 30.0
    stop_grace_s: float = 15.0
    max_diagnoses: int = 3
    logs_dir: str = ""


class RunSupervisor:
    def __init__(
        self,
        run_id: str,
        store: Store,
        *,
        clock: Clock,
        notifier: Notifier,
        diagnoser: Diagnoser,
        owner: str,
        cfg: SupervisorConfig,
        handle: ProcessHandle | None = None,
    ) -> None:
        self.run_id = run_id
        self.store = store
        self.clock = clock
        self.notifier = notifier
        self.diagnoser = diagnoser
        self.owner = owner
        self.cfg = cfg
        self.handle = handle
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.lock = threading.RLock()  # one actor at a time: the tick, or a user's stop request
        self._open: dict[tuple[str, str], int] = {}
        self._gpu: list[tuple[float, float]] = []
        self._last_gpu = 0.0
        self._diagnoses = 0
        self.done = False
        run = self._run()
        self.playbook = Playbook.from_dict(run.get("playbook"))
        self.det_cfg = DetectorConfig.from_dict(run.get("detector"))
        self._init_sources(run, fresh=False)

    # -- lifecycle ------------------------------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name=f"nightshift:{self.run_id}", daemon=True)
        self._thread.start()

    def stop_watching(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def _loop(self) -> None:
        while not self._stop.is_set() and not self.done:
            try:
                with self.lock:
                    self.tick()
            except Exception:  # never let one bad tick kill the night shift
                log.exception("nightshift: supervisor tick failed for %s", self.run_id)
            self._stop.wait(self.cfg.poll_s)
        self.store.release(self.run_id, self.owner)
        self.store.close()

    # -- state ----------------------------------------------------------------------------------------
    def _run(self) -> dict[str, Any]:
        run = self.store.get_run(self.run_id)
        if run is None:
            raise KeyError(self.run_id)
        return run

    def _init_sources(self, run: dict[str, Any], *, fresh: bool) -> None:
        cwd = run.get("cwd") or os.getcwd()
        extra: list[MetricSource] = []
        for spec in run.get("sources") or []:
            spec = dict(spec)
            if not os.path.isabs(os.path.expanduser(spec.get("path", ""))):
                spec["path"] = os.path.join(cwd, spec["path"])
            extra.append(make_source(spec, clock=self.clock))
        if not fresh or not hasattr(self, "extra_sources"):
            self.extra_sources = extra
        self.log_source = LogTailSource(run["log_path"], clock=self.clock) if run.get("log_path") else None
        self.log_metrics = not self.extra_sources  # with explicit sources, the log is only for crash context
        self.window = MetricWindow()
        if not fresh:
            self.window.extend(self.store.load_points(self.run_id, attempt=run.get("attempt") or 0))
        self._open.clear()

    # -- one tick -------------------------------------------------------------------------------------
    def tick(self) -> None:
        run = self._run()
        if run["status"] in TERMINAL:
            self.done = True
            return
        now = self.clock.now()
        if not self.store.claim(self.run_id, self.owner, self.cfg.lease_s, now=now):
            log.info("nightshift: lost the lease on %s; another Hermes process is watching it", self.run_id)
            self.done = True
            return

        points = []
        if self.log_source is not None:
            log_points = self.log_source.poll()
            if self.log_metrics:
                points.extend(log_points)
        for src in self.extra_sources:
            points.extend(src.poll())
        points = [p for p in points if is_safe_name(p.name)]
        if points:
            self.store.add_metrics(self.run_id, points, attempt=run.get("attempt") or 0)
            self.window.extend(points)

        alive, rc = self._process_state(run)
        incidents = run_window_detectors(self.window, self.det_cfg)
        started = (
            self.handle.started_at if self.handle and self.handle.started_at else (run["started_at"] or now)
        )
        if "stall" not in self.det_cfg.disabled:
            incidents += detect_stall(self.window, self.det_cfg, now=now, alive=alive, started_at=started)
        if (
            alive
            and "gpu_idle" not in self.det_cfg.disabled
            and now - self._last_gpu >= self.cfg.gpu_sample_s
        ):
            self._last_gpu = now
            util = gpu_utilization()
            if util is not None:
                self._gpu.append((now, util))
                self._gpu = [s for s in self._gpu if s[0] >= now - 2 * self.det_cfg.gpu_idle_s]
        incidents += detect_gpu_idle(self._gpu, self.det_cfg, now=now, alive=alive)

        if not alive and run["mode"] == "launch":
            tail = self._log_tail(run)
            exit_inc = classify_exit(rc, tail) if rc is not None else _classify_unknown_exit(tail)
            if exit_inc is None:
                self._finish(run, "finished", "training finished")
                return
            incidents.append(exit_inc)

        pid_before = self.handle.pid if self.handle else None
        self._handle_incidents(run, incidents, now, alive)
        relaunched = self.handle is not None and self.handle.pid != pid_before
        if not alive and run["mode"] == "launch" and not self.done and not relaunched:
            # it died and the playbook chose not to restart it: the run is over (already notified above)
            self._finish(
                self._run(), "failed", incidents[-1].message if incidents else "exited", notify=False
            )

    def _process_state(self, run: dict[str, Any]) -> tuple[bool, int | None]:
        if run["mode"] != "launch":
            if run.get("pid"):
                handle = self.handle or ProcessHandle.reattach(
                    run["pid"], run.get("pgid") or run["pid"], run.get("proc_id"), run.get("log_path") or ""
                )
                self.handle = handle
                return handle.is_ours(), None
            return True, None  # attached to files only: treat as alive, stall detection still applies
        if self.handle is None:
            if not run.get("pid"):
                return False, None
            self.handle = ProcessHandle.reattach(
                run["pid"], run.get("pgid") or run["pid"], run.get("proc_id"), run.get("log_path") or ""
            )
        return self.handle.poll()

    def _log_tail(self, run: dict[str, Any]) -> list[str]:
        return redact_lines(read_tail(run["log_path"])) if run.get("log_path") else []

    # -- incidents ------------------------------------------------------------------------------------
    def _handle_incidents(
        self, run: dict[str, Any], incidents: list[Incident], now: float, alive: bool
    ) -> None:
        current = {i.key for i in incidents}
        for key, iid in list(self._open.items()):
            if key not in current:
                self.store.resolve_incident(iid, at=now)
                del self._open[key]

        restarted = False
        # most severe first, so one restart handles everything that tick
        for inc in sorted(incidents, key=lambda i: -i.severity.rank):
            if inc.key in self._open:
                continue
            iid = self.store.add_incident(self.run_id, inc, at=now)
            self._open[inc.key] = iid
            if restarted:
                self.store.add_action(
                    self.run_id, NOTIFY, True, "covered by the recovery above", incident_id=iid
                )
                continue
            run = self._run()
            decision = decide(self.playbook, inc, self._state(run), now)
            result, ok, extra = self._execute(run, inc, decision, alive)
            self.store.add_action(
                self.run_id,
                decision.action,
                ok,
                result,
                incident_id=iid,
                details={**decision.to_dict(), **extra},
            )
            restarted = decision.action in RESTARTS and ok
            diagnosis = (
                self._diagnose(run, inc, decision, result) if inc.severity is Severity.CRITICAL else None
            )
            if diagnosis:
                self.store.set_diagnosis(iid, diagnosis)
            self._notify(inc, decision, result, ok, diagnosis, extra)
            if self.done:
                return

    def _state(self, run: dict[str, Any]) -> RunState:
        return RunState(
            mode=run["mode"],
            params=dict(run.get("params") or {}),
            interventions=run.get("interventions") or 0,
            last_intervention_at=run.get("last_intervention_at"),
            can_resume=RESUME in placeholders(run.get("argv") or []),
        )

    def _execute(
        self, run: dict[str, Any], inc: Incident, d: Decision, alive: bool
    ) -> tuple[str, bool, dict[str, Any]]:
        if d.action == NOTIFY:
            return (d.reason, True, {})
        if d.action == STOP:
            self._stop_process()
            self._finish(run, "stopped", d.reason, notify=False)
            return (f"stopped the run: {d.reason}", True, {})
        # restarts
        self.store.update_run(self.run_id, status="recovering")
        self._stop_process()
        ckpt = self._pick_checkpoint(run, inc, d)
        if d.action == ROLLBACK_RESTART and ckpt is None:
            when = f"step {inc.onset_step}" if inc.onset_step is not None else "the incident"
            reason = f"no checkpoint from before {when}; stopped instead of restarting from scratch"
            self._finish(run, "stopped", reason, notify=False)
            d.action, d.page = STOP, True
            return (reason, False, {})
        try:
            self._relaunch(run, d.params, ckpt)
        except Exception as exc:  # launching can fail (bad cwd, missing binary): stop and page
            log.exception("nightshift: relaunch failed for %s", self.run_id)
            self._finish(run, "failed", f"restart failed: {exc}", notify=False)
            d.page = True
            return (f"restart failed: {exc}", False, {})
        changes = ", ".join(f"{k} {_fmt(a)} → {_fmt(b)}" for k, (a, b) in d.changes.items())
        where = f"step {ckpt.step}" if ckpt and ckpt.step is not None else (ckpt.path if ckpt else "scratch")
        verb = "rolled back to" if d.action == ROLLBACK_RESTART else "restarted from"
        text = f"{verb} {where}" + (f", {changes}" if changes else "")
        return (text, True, {"checkpoint": ckpt.to_dict() if ckpt else None})

    def _pick_checkpoint(self, run: dict[str, Any], inc: Incident, d: Decision) -> Checkpoint | None:
        if RESUME not in placeholders(run.get("argv") or []):
            return None
        patterns = self.playbook.checkpoint_patterns or list(DEFAULT_PATTERNS)
        ckpts = discover(run.get("cwd") or os.getcwd(), patterns)
        if d.action == OOM_RESTART:
            # memory errors do not corrupt weights: the newest checkpoint of this attempt is fine
            return max(ckpts, key=lambda c: (c.step if c.step is not None else -1, c.mtime), default=None)
        return select_rollback(
            ckpts,
            onset_step=inc.onset_step,
            onset_wall=inc.onset_wall,
            margin_steps=self.playbook.rollback_margin_steps,
        )

    def _relaunch(self, run: dict[str, Any], params: dict[str, Any], ckpt: Checkpoint | None) -> None:
        attempt = (run.get("attempt") or 0) + 1
        resume = ckpt.path if ckpt else None
        argv = render_argv(run["argv"], params, resume=resume)
        log_path = str(Path(self.cfg.logs_dir or Path(run["log_path"]).parent) / f"attempt-{attempt}.log")
        handle = ProcessHandle.launch(
            argv,
            cwd=run["cwd"],
            log_path=log_path,
            env_extra=template_env(params, resume, self.run_id, attempt),
        )
        self.handle = handle
        now = self.clock.now()
        self.store.update_run(
            self.run_id,
            status="running",
            params=params,
            attempt=attempt,
            pid=handle.pid,
            pgid=handle.pgid,
            proc_id=handle.proc_id,
            log_path=log_path,
            last_intervention_at=now,
            restored_from=resume,
            note=f"attempt {attempt}",
        )
        self.store.bump_interventions(self.run_id)
        for iid in self._open.values():
            self.store.resolve_incident(iid, at=now)  # recovered: the new attempt starts healthy
        self._init_sources(self._run(), fresh=True)

    def request_stop(self, reason: str) -> str:
        """A user asked (via /nightshift, the Desktop pane or an approved agent proposal) to stop the run."""
        with self.lock:
            run = self._run()
            if run["status"] in TERMINAL:
                return f"already {run['status']}"
            self._stop_process()
            self.store.add_action(self.run_id, STOP, True, reason)
            self._finish(run, "stopped", reason)
            return "stopped"

    def _stop_process(self) -> None:
        if self.handle is not None:
            self.handle.stop(self.cfg.stop_grace_s)

    def _finish(self, run: dict[str, Any], status: str, note: str, *, notify: bool = True) -> None:
        now = self.clock.now()
        self.store.update_run(self.run_id, status=status, ended_at=now, note=note)
        for iid in self._open.values():
            self.store.resolve_incident(iid, at=now)  # a run that is over has nothing open any more
        self._open.clear()
        self.done = True
        if notify:
            self.notifier.send(alert_for_status(self.store, self._run(), status, note))

    def _diagnose(self, run: dict[str, Any], inc: Incident, d: Decision, result: str) -> str | None:
        if self._diagnoses >= self.cfg.max_diagnoses:
            return None
        self._diagnoses += 1
        recent = {}
        for name in self.window.names()[:8]:
            vals = self.window.values(name)[-40:]
            recent[name] = [round(v, 5) if math.isfinite(v) else str(v) for v in vals]
        ctx = {
            "run": {"name": run["name"], "argv": run.get("argv"), "params": run.get("params")},
            "incident": inc.to_dict(),
            "action": {"decision": d.to_dict(), "result": result},
            "recent_metrics": recent,
            "log_tail": self._log_tail(run)[-40:],
        }
        try:
            return self.diagnoser.diagnose(ctx)
        except Exception:
            log.exception("nightshift: diagnosis failed")
            return None

    def _notify(
        self, inc: Incident, d: Decision, result: str, ok: bool, diagnosis: str | None, extra: dict[str, Any]
    ) -> None:
        run = self._run()
        chart = None
        if d.page:
            try:
                chart = chart_for_run(self.store, run)
            except Exception:
                log.exception("nightshift: chart rendering failed")
        self.notifier.send(alert_for_incident(run, inc, d, result, ok, diagnosis, chart))


def _classify_unknown_exit(tail: list[str]) -> Incident | None:
    """A re-attached process we are not the parent of: no exit code, so read the log's last words."""
    text = "\n".join(tail[-40:]).lower()
    if "traceback" in text or "error" in text or "killed" in text:
        inc = classify_exit(1, tail)
        if inc is not None:
            inc.details["returncode"] = None
            inc.message = inc.message.replace("exit 1", "exit code unknown").replace(
                "exited with code 1", "exited with an error"
            )
        return inc
    return None


def _fmt(v: Any) -> str:
    return f"{v:.3g}" if isinstance(v, float) else str(v)


__all__ = ["TERMINAL", "Alert", "IncidentKind", "RunSupervisor", "SupervisorConfig"]
