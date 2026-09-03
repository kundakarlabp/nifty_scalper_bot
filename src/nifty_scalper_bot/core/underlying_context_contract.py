"""Canonical underlying-context contract for option strategy evaluation.

This module keeps the option execution domain connected to the active NIFTY
spot/futures context without creating another history owner.

Ownership:
- MarketDataManager remains the sole completed OHLC owner.
- DataHub exposes the active basket/quotes as a facade.
- Strategy context carries only symbol identities/metadata.
- ORB/SMC read completed bars from the shared IndicatorEngine.
- OrderManager is deliberately not involved.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import os
from typing import Any

from nifty_scalper_bot.utils.symbols import normalize_symbol

_DEFAULT_OHLC_CAPACITY = 500
_MIN_CONTEXT_SESSION_CAPACITY = 400


def _safe_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        parsed = int(default)
    return max(1, parsed)


def configured_ohlc_capacity() -> int:
    """Return completed-OHLC retention independently of raw tick retention."""
    configured = _safe_positive_int(
        os.getenv("MDM_OHLC_CACHE_LEN", str(_DEFAULT_OHLC_CAPACITY)),
        _DEFAULT_OHLC_CAPACITY,
    )
    return max(_MIN_CONTEXT_SESSION_CAPACITY, configured)


def _resize_deque(value: Any, capacity: int) -> deque[Any]:
    rows = list(value or [])
    return deque(rows[-capacity:], maxlen=capacity)


def ensure_mdm_ohlc_capacity(manager: Any) -> int:
    """Upgrade an existing MDM instance's completed-history capacity in place.

    Raw tick retention is intentionally untouched. Existing completed history is
    preserved and CandleEngine instances are enlarged without inventing bars.
    """
    capacity = configured_ohlc_capacity()
    current = _safe_positive_int(getattr(manager, "_ohlc_cache_len", 0), 1)
    if current < capacity:
        setattr(manager, "_ohlc_cache_len", capacity)

    ohlc = getattr(manager, "_ohlc", None)
    if isinstance(ohlc, Mapping):
        # defaultdict/deque projection is compatibility state only; CandleEngine
        # remains authoritative. Preserve the mapping factory when possible.
        factory = getattr(ohlc, "default_factory", None)
        for symbol in list(ohlc.keys()):
            try:
                ohlc[symbol] = _resize_deque(ohlc[symbol], capacity)
            except Exception:  # noqa: BLE001 - capacity adaptation is best-effort
                continue
        if factory is not None:
            try:
                ohlc.default_factory = lambda: deque(maxlen=capacity)
            except Exception:  # noqa: BLE001
                pass

    engines = getattr(manager, "_engines", None)
    if isinstance(engines, Mapping):
        for engine in list(engines.values()):
            try:
                engine_capacity = int(getattr(engine, "max_bars", 0) or 0)
            except (TypeError, ValueError):
                engine_capacity = 0
            if engine_capacity >= capacity:
                continue
            completed = getattr(engine, "_completed_candles", None)
            if completed is not None:
                try:
                    engine._completed_candles = _resize_deque(completed, capacity)
                except Exception:  # noqa: BLE001
                    continue
            try:
                engine.max_bars = capacity
            except Exception:  # noqa: BLE001
                pass
    return capacity


def install_mdm_ohlc_capacity_contract() -> bool:
    """Decouple MDM completed-OHLC capacity from raw tick cache length."""
    try:
        from nifty_scalper_bot.data.market_data_manager import MarketDataManager
    except Exception:
        return False

    marker = "_underlying_ohlc_capacity_contract_installed"
    if bool(getattr(MarketDataManager, marker, False)):
        return True

    original_init = MarketDataManager.__init__
    original_get = MarketDataManager.get_ohlc_bars

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        capacity = configured_ohlc_capacity()
        self._ohlc_cache_len = capacity
        # The original constructor couples _ohlc to the raw-tick cache. Replace
        # only completed-history projection storage; raw ticks retain their
        # original MDM_TICK_CACHE_LEN bound.
        existing = getattr(self, "_ohlc", None)
        if existing is not None:
            try:
                from collections import defaultdict

                replacement = defaultdict(lambda: deque(maxlen=capacity))
                for symbol, rows in dict(existing).items():
                    replacement[symbol] = _resize_deque(rows, capacity)
                self._ohlc = replacement
            except Exception:  # noqa: BLE001
                pass

    def get_ohlc_bars(
        self: Any, symbol: str, *, limit: int | None = None
    ) -> list[Any]:
        ensure_mdm_ohlc_capacity(self)
        return list(original_get(self, symbol, limit=limit) or [])

    def history_capacity_for(
        self: Any,
        _symbol: str | None = None,
        *,
        role: str | None = None,
        interval: str = "1minute",
    ) -> int:
        del role, interval
        return ensure_mdm_ohlc_capacity(self)

    setattr(__init__, "_underlying_ohlc_capacity_adapted", True)
    setattr(get_ohlc_bars, "_underlying_ohlc_capacity_adapted", True)
    MarketDataManager.__init__ = __init__  # type: ignore[method-assign]
    MarketDataManager.get_ohlc_bars = get_ohlc_bars  # type: ignore[method-assign]
    MarketDataManager.history_capacity_for = history_capacity_for  # type: ignore[attr-defined]
    setattr(MarketDataManager, marker, True)
    return True


def _basket_from_data_hub(data_hub: Any | None) -> Mapping[str, Any]:
    if data_hub is None:
        return {}
    getter = getattr(data_hub, "get_active_contract_basket", None)
    if callable(getter):
        try:
            basket = getter()
        except Exception:  # noqa: BLE001 - context stays fail-closed
            basket = None
        if isinstance(basket, Mapping):
            return basket
    mdm = getattr(data_hub, "_mdm", None)
    getter = getattr(mdm, "get_active_contract_basket", None)
    if callable(getter):
        try:
            basket = getter()
        except Exception:  # noqa: BLE001
            basket = None
        if isinstance(basket, Mapping):
            return basket
    return {}


def resolve_active_underlying_symbols(
    data_hub: Any | None,
    runner_context: Mapping[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    """Return canonical spot/futures identities for option strategy context."""
    context = runner_context or {}
    basket = _basket_from_data_hub(data_hub)

    raw_spot = (
        context.get("spot_symbol")
        or basket.get("spot_symbol")
        or "NSE:NIFTY"
    )
    raw_future = (
        context.get("futures_symbol")
        or basket.get("futures_symbol")
        or basket.get("future_symbol")
    )
    spot = normalize_symbol(str(raw_spot or "")) or None
    future = normalize_symbol(str(raw_future or "")) or None
    return spot, future


# Install at import time because strategy_context_builder imports this contract
# during StrategyManager module construction, before the production MDM instance
# is normally created. Existing instances are still upgraded lazily on reads.
install_mdm_ohlc_capacity_contract()


__all__ = [
    "configured_ohlc_capacity",
    "ensure_mdm_ohlc_capacity",
    "install_mdm_ohlc_capacity_contract",
    "resolve_active_underlying_symbols",
]
