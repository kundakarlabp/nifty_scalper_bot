"""Session readiness adapter for startup diagnostics."""

from __future__ import annotations

from typing import Any, Callable


def adapt_compute_live_readiness(original: Callable[..., tuple[bool, list[str]]]) -> Callable[..., tuple[bool, list[str]]]:
    """Return an adapter that keeps option quote checks quiet outside session."""

    def wrapped(**kwargs: Any) -> tuple[bool, list[str]]:
        if bool(kwargs.get("live_mode")) and not bool(kwargs.get("market_open")):
            min_bars = int(kwargs.get("option_exec_min_bars") or 1)
            adjusted = dict(kwargs)
            adjusted["ce_quote_ready"] = True
            adjusted["pe_quote_ready"] = True
            adjusted["ce_bars"] = max(int(adjusted.get("ce_bars") or 0), min_bars)
            adjusted["pe_bars"] = max(int(adjusted.get("pe_bars") or 0), min_bars)
            return original(**adjusted)
        return original(**kwargs)

    return wrapped


def apply_app_patch(app_module: Any) -> None:
    """Install the adapter on a loaded app module."""

    current = getattr(app_module, "compute_live_readiness", None)
    if not callable(current) or getattr(current, "_session_readiness_adapted", False):
        return
    wrapped = adapt_compute_live_readiness(current)
    setattr(wrapped, "_session_readiness_adapted", True)
    setattr(app_module, "compute_live_readiness", wrapped)


__all__ = ["adapt_compute_live_readiness", "apply_app_patch"]
