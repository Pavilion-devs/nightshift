"""Nightshift: Hermes takes the night shift on your ML training runs.

The engine lives in ``nightshift.core`` and never imports Hermes. Everything that touches Hermes lives in
``nightshift.adapter``; ``register(ctx)`` is the Hermes entrypoint.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"


def register(ctx: Any) -> None:
    """Hermes entrypoint: wire tools, hooks, commands and RPC methods into the host."""
    from .adapter.register import install

    install(ctx)
