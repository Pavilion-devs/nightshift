"""Mask secrets in log lines before they are stored, sent to an LLM or pushed to a phone.

The adapter additionally runs Hermes's own ``agent.redact.redact_sensitive_text``; this is the engine's
dependency-free first pass.
"""

from __future__ import annotations

import re

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(sk|rk|pk)-[A-Za-z0-9_\-]{16,}"), "[REDACTED_KEY]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "[REDACTED_HF_TOKEN]"),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"), "[REDACTED_SLACK_TOKEN]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"\b\d{8,10}:[A-Za-z0-9_\-]{30,}"), "[REDACTED_BOT_TOKEN]"),
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-~+/]{16,}=*"), r"\1 [REDACTED]"),
    (
        re.compile(
            r"(?i)\b([a-z0-9_]*?(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret|password|passwd|token))"
            r"(\s*[=:]\s*)(['\"]?)[^\s'\",]{6,}\3"
        ),
        r"\1\2\3[REDACTED]\3",
    ),
    (re.compile(r"(?i)(https?://)[^/\s:@]+:[^/\s@]+@"), r"\1[REDACTED]@"),
]


def redact(text: str) -> str:
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


def redact_lines(lines: list[str]) -> list[str]:
    return [redact(ln) for ln in lines]
