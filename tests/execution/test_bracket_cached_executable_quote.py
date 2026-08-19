from __future__ import annotations

from unittest.mock import Mock

from nifty_scalper_bot.execution.bracket_manager import BracketManager


class _CachedMarketData:
    def __init__(self, tick: dict[str, object]) -> None:
        self.tick = dict(tick)

    def get_latest_tick(self, _symbol: str) -> dict[str, object]:
        return dict(self.tick)


def _manager(market_data: _CachedMarketData) -> BracketManager:
    order_manager = Mock()
    order_manager.is_live_mode.return_value = False
    return BracketManager(order_manager=order_manager, market_data=market_data)


def test_runtime_on_tick_recovers_same_tick_executable_quote_from_ssot() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    manager = _manager(
        _CachedMarketData(
            {
                "symbol": symbol,
                "ltp": 99.50,
                "bid": 99.25,
                "ask": 99.55,
                "timestamp": 1_000.0,
                "source": "ws",
            }
        )
    )

    manager.on_tick(symbol, 99.50, exchange_ts=1_000.0, defer_submission=True)

    assert manager._exit_quotes[symbol][:2] == (99.25, 99.55)


def test_runtime_on_tick_does_not_mix_quote_from_different_tick() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    manager = _manager(
        _CachedMarketData(
            {
                "symbol": symbol,
                "ltp": 99.70,
                "bid": 99.45,
                "ask": 99.75,
                "timestamp": 1_001.0,
                "source": "ws",
            }
        )
    )

    manager.on_tick(symbol, 99.50, exchange_ts=1_000.0, defer_submission=True)

    assert symbol not in manager._exit_quotes
