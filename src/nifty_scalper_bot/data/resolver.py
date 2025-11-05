"""Resolver façade for instrument metadata lookups."""

from __future__ import annotations

from nifty_scalper_bot.data.instruments import InstrumentResolver

# --- injected: retry helper for network calls (idempotent insertion) ----------
import random
import threading

def _with_retries(fn, *, retries: int = 3, backoff: float = 0.2, max_backoff: float = 5.0):
    """
    Call fn() with retries. fn may raise; retries uses exponential backoff with jitter.
    Returns fn() result or raises last exception.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - network error paths
            last_exc = exc
            sleep = min(max_backoff, backoff * (2 ** (attempt - 1))) * (0.5 + random.random() * 0.5)
            time.sleep(sleep)
    # final raise
    raise last_exc
# --- end injected helper ---------------------------------------------------

__all__ = ["InstrumentResolver"]
