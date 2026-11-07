"""One Nightshift engine per Hermes process, created lazily and shared by tools, hooks and commands."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from ..core.manager import Manager
from ..core.store import Store
from ..core.supervisor import SupervisorConfig
from .diagnose import HermesDiagnoser
from .notify import HermesNotifier

log = logging.getLogger("nightshift")

_lock = threading.Lock()
_runtime: Runtime | None = None


def data_dir() -> Path:
    try:
        from plugins.plugin_storage import plugin_data_dir

        return Path(plugin_data_dir("nightshift"))
    except ImportError:  # outside Hermes (tests, scripts)
        path = Path.home() / ".hermes" / "plugin-data" / "nightshift"
        path.mkdir(parents=True, exist_ok=True)
        return path


def db_path() -> Path:
    return data_dir() / "nightshift.db"


class Runtime:
    def __init__(self, ctx: Any, store: Store | None = None, root: Path | None = None) -> None:
        self.ctx = ctx
        root = root or data_dir()
        self.store = store or Store(root / "nightshift.db")
        poll = float(_setting(ctx, "poll_seconds", 2.0) or 2.0)
        self.manager = Manager(
            self.store,
            data_dir=root,
            notifier=HermesNotifier(ctx, self.store),
            diagnoser=HermesDiagnoser(ctx),
            sup_cfg=SupervisorConfig(
                poll_s=max(0.5, poll), max_diagnoses=int(_setting(ctx, "max_diagnoses", 3))
            ),
        )
        self._started = False

    def ensure_started(self) -> None:
        if not self._started:
            self._started = True
            self.manager.start()

    def shutdown(self) -> None:
        self.manager.shutdown(wait=False)


def get_runtime(ctx: Any) -> Runtime:
    global _runtime
    with _lock:
        if _runtime is None:
            _runtime = Runtime(ctx)
        _runtime.ensure_started()
        return _runtime


def reset_runtime() -> None:
    """Stop the engine (plugin unload). Training processes keep running and are re-adopted next time."""
    global _runtime
    with _lock:
        if _runtime is not None:
            _runtime.shutdown()
        _runtime = None


def _setting(ctx: Any, key: str, default: Any) -> Any:
    try:
        value = ctx.get_config(key, default)
    except Exception:
        return default
    return default if value is None else value
