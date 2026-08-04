from datetime import date
from typing import Any

from nifty_scalper_bot.strategies.signal_generator import StrategyManager


class _PM:
    def get_position(self, _symbol: str):
        return None


class _IE:
    def get_indicators(self, _symbol: str, _names: list[str]):
        return {}


class _MDM:
    def get_active_nifty_future_symbol_cached(self):
        return "NFO:NIFTY26JUNFUT"


class _Hub:
    def __init__(self, active: str | None = "NFO:NIFTY26JUNFUT") -> None:
        self._mdm = _MDM()
        self.calls: list[str] = []
        self._active = active

    def get_active_futures_symbol(self):
        return self._active

    def get_quote(self, symbol: str, allow_pull: bool = False):
        self.calls.append(symbol)
        if symbol == "NSE:NIFTY":
            return {"ltp": 24000.0}
        if symbol == "NFO:NIFTY26JUNFUT":
            return {"vwap": 24100.0, "ltp": 24120.0}
        return {}


def _manager(futures_symbol: str | None = None, hub: Any | None = None) -> StrategyManager:
    return StrategyManager([], _IE(), _PM(), data_hub=hub, futures_symbol=futures_symbol)


def test_signal_generator_does_not_construct_calendar_month_may_future_after_expiry() -> None:
    hub = _Hub()
    manager = _manager(hub=hub)
    assert manager._resolve_active_futures_symbol_for_metrics() == "NFO:NIFTY26JUNFUT"
    assert "NFO:NIFTY26MAYFUT" not in hub.calls


def test_augment_futures_metrics_uses_mdm_active_future_only() -> None:
    hub = _Hub()
    manager = _manager(hub=hub)
    indicators: dict[str, Any] = {}
    manager._augment_futures_metrics(indicators)
    assert "NFO:NIFTY26JUNFUT" in hub.calls
    assert "NFO:NIFTY26MAYFUT" not in hub.calls


def test_configured_bare_nifty_does_not_generate_nfo_niftyfut() -> None:
    manager = _manager(futures_symbol="NIFTY")
    assert manager._futures_symbol is None
    assert manager._resolve_active_futures_symbol_for_metrics() != "NFO:NIFTYFUT"


def test_resolve_active_future_falls_back_to_configured_runtime_future() -> None:
    hub = _Hub(active=None)
    manager = _manager(futures_symbol="NFO:NIFTY26JUNFUT", hub=hub)
    assert manager._resolve_active_futures_symbol_for_metrics() == "NFO:NIFTY26JUNFUT"


def test_augment_futures_metrics_tries_configured_runtime_future_when_hub_unresolved() -> None:
    hub = _Hub(active=None)
    manager = _manager(futures_symbol="NFO:NIFTY26JUNFUT", hub=hub)
    indicators: dict[str, Any] = {}
    manager._augment_futures_metrics(indicators)
    assert "NFO:NIFTY26JUNFUT" in hub.calls
    assert "NFO:NIFTY26MAYFUT" not in hub.calls


class _VolumeHub:
    def __init__(
        self,
        quotes: list[dict[str, float]],
        active: str = "NFO:NIFTY26JUNFUT",
    ) -> None:
        self._quotes = list(quotes)
        self._active = active

    def get_active_futures_symbol(self):
        return self._active

    def get_quote(self, symbol: str, allow_pull: bool = False):
        del allow_pull
        if symbol == "NSE:NIFTY":
            return {"ltp": 24000.0, "vwap": 24000.0}
        if symbol == self._active and self._quotes:
            return self._quotes.pop(0)
        return {}


def test_futures_volume_ignores_trade_quantity_and_keeps_baseline() -> None:
    symbol = "NFO:NIFTY26JUNFUT"
    hub = _VolumeHub(
        [
            {"volume_traded_today": 1000.0},
            {"last_quantity": 10.0},
            {"volume_traded_today": 1050.0},
        ],
        active=symbol,
    )
    manager = _manager(futures_symbol=symbol, hub=hub)

    first: dict[str, Any] = {}
    manager._augment_futures_metrics(first)
    assert manager._last_futures_volume == 1000.0
    assert list(manager._futures_volume_history) == []
    assert first["futures_volume_source"] == "volume_traded_today"
    assert first["futures_volume_trusted"] is True

    # Per-trade quantity must not replace the cumulative session-volume baseline.
    trade_quantity_only: dict[str, Any] = {}
    manager._augment_futures_metrics(trade_quantity_only)
    assert manager._last_futures_volume == 1000.0
    assert list(manager._futures_volume_history) == []
    assert trade_quantity_only["futures_volume"] is None
    assert trade_quantity_only["futures_volume_trusted"] is False

    next_cumulative: dict[str, Any] = {}
    manager._augment_futures_metrics(next_cumulative)
    assert manager._last_futures_volume == 1050.0
    assert list(manager._futures_volume_history) == [50.0]
    assert next_cumulative["futures_volume_ratio"] == 1.0


def test_futures_volume_rejects_same_session_decrease_without_false_spike() -> None:
    symbol = "NFO:NIFTY26JUNFUT"
    hub = _VolumeHub(
        [
            {"volume_traded": 1000.0},
            {"volume_traded": 1100.0},
            {"volume_traded": 900.0},
            {"volume_traded": 1150.0},
        ],
        active=symbol,
    )
    manager = _manager(futures_symbol=symbol, hub=hub)

    manager._augment_futures_metrics({})
    manager._augment_futures_metrics({})

    decreased: dict[str, Any] = {}
    manager._augment_futures_metrics(decreased)
    assert manager._last_futures_volume == 1100.0
    assert list(manager._futures_volume_history) == [100.0]
    assert decreased["futures_volume"] is None
    assert decreased["futures_volume_trusted"] is False

    recovered: dict[str, Any] = {}
    manager._augment_futures_metrics(recovered)
    assert manager._last_futures_volume == 1150.0
    assert list(manager._futures_volume_history) == [100.0, 50.0]
    assert recovered["futures_volume_ratio"] == 50.0 / 75.0


def test_futures_volume_resets_baseline_when_contract_changes() -> None:
    symbol = "NFO:NIFTY26JULFUT"
    hub = _VolumeHub([{"volume_traded_today": 125.0}], active=symbol)
    manager = _manager(futures_symbol=symbol, hub=hub)
    manager._last_futures_volume = 9000.0
    manager._last_futures_volume_symbol = "NFO:NIFTY26JUNFUT"
    manager._last_futures_volume_date = date.today()
    manager._futures_volume_history.append(75.0)

    indicators: dict[str, Any] = {}
    manager._augment_futures_metrics(indicators)

    assert manager._last_futures_volume == 125.0
    assert manager._last_futures_volume_symbol == symbol
    assert list(manager._futures_volume_history) == []
    assert indicators["futures_volume_ratio"] is None
    assert indicators["futures_volume_trusted"] is True
