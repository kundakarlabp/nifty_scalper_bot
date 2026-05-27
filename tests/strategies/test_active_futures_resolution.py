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
    def __init__(self) -> None:
        self._mdm = _MDM()
        self.calls: list[str] = []

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
