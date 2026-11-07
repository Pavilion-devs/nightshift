"""Incident diagnosis with the user's own model via ``ctx.llm``. Only called for critical incidents, after the
recovery has already happened, and capped per run by the supervisor."""

from __future__ import annotations

import json
import logging
from typing import Any

from ..core.redact import redact

log = logging.getLogger("nightshift")

SYSTEM = (
    "You are a senior ML training engineer on call. A training run just had an incident and an automatic "
    "recovery may already have been applied. From the metrics, the log tail and the command, give the most "
    "likely root cause in one sentence, then one concrete change to prevent it next time. Plain text, at most "
    "two sentences, no preamble, no markdown. If the evidence is thin, say what you would check first."
)


class HermesDiagnoser:
    def __init__(self, ctx: Any, max_tokens: int = 160, timeout: float = 45.0) -> None:
        self.ctx = ctx
        self.max_tokens = max_tokens
        self.timeout = timeout

    def enabled(self) -> bool:
        try:
            return bool(self.ctx.get_config("diagnose", True))
        except Exception:
            return True

    def diagnose(self, context: dict[str, Any]) -> str | None:
        if not self.enabled():
            return None
        payload = redact(json.dumps(context, default=str)[:12_000])
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": payload}]
        try:
            result = self.ctx.llm.complete(messages, max_tokens=self.max_tokens, timeout=self.timeout)
        except Exception as exc:  # no model configured, trust policy, network: recovery already happened
            log.info("nightshift: diagnosis skipped: %s", exc)
            return None
        text = getattr(result, "text", None)
        if text is None and isinstance(result, dict):
            text = result.get("text")
        text = " ".join(str(text or "").split())
        return text[:500] or None
