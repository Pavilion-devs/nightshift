"""Launching, watching and stopping the training process. Only ever signals process groups we created.

Safety properties:
  * argv lists only, never a shell;
  * every process starts in its own session (its own process group), so a stop reaches the trainer and its
    dataloader workers, and nothing else;
  * before any signal we re-check that the pid is still the process we started (start time + command), so a
    recycled pid after a reboot can never be killed by mistake.
"""

from __future__ import annotations

import os
import shlex
import signal
import string
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

RESUME = "resume"


class TemplateError(ValueError):
    pass


def placeholders(argv: list[str]) -> set[str]:
    names: set[str] = set()
    for arg in argv:
        for _, field, _, _ in string.Formatter().parse(arg):
            if field is not None:
                if not field.isidentifier():
                    raise TemplateError(f"bad placeholder {{{field}}} in {arg!r}")
                names.add(field)
    return names


def parse_command(command: str | list[str]) -> list[str]:
    argv = shlex.split(command) if isinstance(command, str) else [str(a) for a in command]
    if not argv:
        raise TemplateError("empty command")
    return argv


def render_argv(template: list[str], params: dict[str, object], resume: str | None = None) -> list[str]:
    """Fill ``{name}`` placeholders. Arguments mentioning ``{resume}`` are dropped when there is no checkpoint,
    so ``--resume={resume}`` disappears on a fresh start instead of becoming ``--resume=``."""
    values = {k: _fmt(v) for k, v in params.items()}
    out: list[str] = []
    for i, arg in enumerate(template):
        names = placeholders([arg])
        if RESUME in names:
            if resume is None:
                # `--resume {resume}` (two words): drop the flag along with its missing value
                prev = template[i - 1] if i else ""
                if (
                    arg.strip() == "{resume}"
                    and out
                    and prev.startswith("-")
                    and "{" not in prev
                    and "=" not in prev
                ):
                    out.pop()
                continue
            values[RESUME] = resume
        missing = names - values.keys()
        if missing:
            raise TemplateError(f"no value for {', '.join(sorted(missing))} in {arg!r}")
        out.append(arg.format(**values))
    return out


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def template_env(params: dict[str, object], resume: str | None, run_id: str, attempt: int) -> dict[str, str]:
    """The same values as environment variables, for trainers that prefer reading env over flags."""
    env = {f"NIGHTSHIFT_{k.upper()}": _fmt(v) for k, v in params.items()}
    env["NIGHTSHIFT_RUN_ID"] = run_id
    env["NIGHTSHIFT_ATTEMPT"] = str(attempt)
    if resume is not None:
        env["NIGHTSHIFT_RESUME"] = resume
    return env


def fingerprint(pid: int) -> str | None:
    """Start time + command of ``pid`` as reported by ``ps``; None if it is not running."""
    try:
        res = subprocess.run(
            ["ps", "-o", "lstart=,command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = res.stdout.strip()
    return line or None


def is_zombie(pid: int) -> bool:
    """Exited but not yet reaped. Counts as dead: it uses no GPU and cannot be signalled."""
    try:
        res = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return res.stdout.strip().startswith("Z")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return not is_zombie(pid)


@dataclass
class ProcessHandle:
    pid: int
    pgid: int
    proc_id: str | None
    log_path: str
    popen: subprocess.Popen | None = None  # only when we spawned it in this process
    started_at: float = 0.0

    @classmethod
    def launch(
        cls, argv: list[str], *, cwd: str, log_path: str, env_extra: dict[str, str] | None = None
    ) -> ProcessHandle:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, **(env_extra or {}), "PYTHONUNBUFFERED": "1"}
        with open(log_path, "ab", buffering=0) as log:
            popen = subprocess.Popen(
                argv,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
                close_fds=True,
            )
        return cls(
            pid=popen.pid,
            pgid=popen.pid,  # start_new_session: the child leads its own group
            proc_id=fingerprint(popen.pid),
            log_path=log_path,
            popen=popen,
            started_at=time.time(),
        )

    @classmethod
    def reattach(cls, pid: int, pgid: int, proc_id: str | None, log_path: str) -> ProcessHandle:
        return cls(pid=pid, pgid=pgid, proc_id=proc_id, log_path=log_path)

    def is_ours(self) -> bool:
        """Still the very process we started (guards against pid reuse)."""
        if self.popen is not None:
            return self.popen.poll() is None
        if not pid_alive(self.pid):
            return False
        current = fingerprint(self.pid)
        return current is not None and (self.proc_id is None or current == self.proc_id)

    def poll(self) -> tuple[bool, int | None]:
        """(alive, exit code). Exit code is None for a re-attached process whose parent we are not."""
        if self.popen is not None:
            rc = self.popen.poll()
            return (rc is None, rc)
        return (self.is_ours(), None)

    def stop(self, grace_s: float = 15.0) -> bool:
        """SIGTERM the process group, wait, then SIGKILL. Returns True once nothing of it is left running."""
        if not self.is_ours():
            return True
        try:
            os.killpg(self.pgid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        deadline = time.time() + grace_s
        while time.time() < deadline:
            if self.popen is not None:
                if self.popen.poll() is not None and not _group_alive(self.pgid):
                    return True
            elif not _group_alive(self.pgid) or not pid_alive(self.pid):
                return True
            time.sleep(0.1)
        try:
            os.killpg(self.pgid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except PermissionError:
            # macOS refuses to signal a group whose only members are zombies: nothing is left running
            return not pid_alive(self.pid)
        if self.popen is not None:
            try:
                self.popen.wait(timeout=5)
            except subprocess.TimeoutExpired:
                return False
        return not _group_alive(self.pgid)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_tail(path: str, max_bytes: int = 32_768, max_lines: int = 120) -> list[str]:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            data = fh.read()
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    return [ln for seg in lines for ln in seg.split("\r") if ln.strip()][-max_lines:]


def gpu_utilization() -> float | None:
    """Max utilisation across NVIDIA GPUs, or None when nvidia-smi is unavailable (e.g. on a Mac)."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    vals = [float(x) for x in res.stdout.split() if x.replace(".", "", 1).isdigit()]
    return max(vals) if res.returncode == 0 and vals else None
