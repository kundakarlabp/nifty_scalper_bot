"""Keep StrategyRunner's frozen gate aligned with its dynamic active universe."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from nifty_scalper_bot.utils.symbols import normalize_symbol


def apply_patches() -> None:
    """Install the dynamic-universe admission fix once."""
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    if getattr(StrategyRunner, "_dynamic_universe_safety_installed", False):
        return

    original: Callable[..., bool] = StrategyRunner._validate_symbol_for_cycle

    @wraps(original)
    def validate_symbol_for_cycle(self: Any, symbol: str) -> bool:
        normalized = normalize_symbol(str(symbol or ""))
        admitted = False
        if normalized and bool(getattr(self, "_universe_dynamic_mode", False)):
            lock = getattr(self, "_lock", None)
            if lock is not None:
                with lock:
                    active = normalized in getattr(self, "_active_symbols", set())
                    frozen = getattr(self, "_frozen_universe", None)
                    if active and isinstance(frozen, set) and normalized not in frozen:
                        frozen.add(normalized)
                        admitted = True
        if admitted:
            self._logger.info(
                "DYNAMIC_ACTIVE_SYMBOL_ADMITTED symbol=%s reason=active_universe_authority",
                normalized,
                extra={
                    "event": "DYNAMIC_ACTIVE_SYMBOL_ADMITTED",
                    "symbol": normalized,
                    "reason": "active_universe_authority",
                },
            )
        return original(self, normalized or symbol)

    StrategyRunner._dynamic_universe_safety_original_validate = original
    StrategyRunner._validate_symbol_for_cycle = validate_symbol_for_cycle
    StrategyRunner._dynamic_universe_safety_installed = True


__all__ = ["apply_patches"]
