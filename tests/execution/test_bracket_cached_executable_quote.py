from __future__ import annotations

from datetime import datetime, timezone
from time import time
from unittest.mock import Mock

from nifty_scalper_bot.execution.bracket_manager import BracketManager
from nifty_scalper_bot.execution.bracket_core import BracketState


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


def test_runtime_on_tick_uses_explicit_full_tick_when_cache_has_advanced() -> None:
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
    full_tick = {
        "symbol": symbol,
        "ltp": 99.50,
        "bid": 99.25,
        "ask": 99.55,
        "timestamp": 1_000.0,
        "source": "ws",
    }

    manager.on_tick(
        symbol,
        99.50,
        exchange_ts=1_000.0,
        defer_submission=True,
        tick=full_tick,
    )

    assert manager._exit_quotes[symbol][:2] == (99.25, 99.55)


def test_runtime_on_tick_falls_back_to_same_tick_cache_when_tick_has_no_depth() -> None:
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
    projected_tick = {
        "symbol": symbol,
        "ltp": 99.50,
        "timestamp": 1_000.0,
        "source": "ws",
    }

    manager.on_tick(
        symbol,
        99.50,
        exchange_ts=1_000.0,
        defer_submission=True,
        tick=projected_tick,
    )

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


def test_runtime_on_tick_matches_cached_quote_using_received_at_fallback() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    received_at = datetime.fromtimestamp(1_000.0, tz=timezone.utc)
    manager = _manager(
        _CachedMarketData(
            {
                "symbol": symbol,
                "ltp": 99.50,
                "bid": 99.25,
                "ask": 99.55,
                "received_at": received_at,
                "source": "ws",
            }
        )
    )

    manager.on_tick(symbol, 99.50, exchange_ts=1_000.0, defer_submission=True)

    assert manager._exit_quotes[symbol][:2] == (99.25, 99.55)


def test_shadow_records_full_lot_bid_only_with_same_tick_exchange_proof() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    manager = object.__new__(BracketManager)
    manager._exit_quote_max_age = 2.0
    manager._exit_quotes = {symbol: (105.0, 105.2, time())}
    stamp = datetime.now(timezone.utc)
    tick = {
        "symbol": symbol,
        "ltp": 105.1,
        "bid": 105.0,
        "ask": 105.2,
        "source": "ws_full",
        "source_timestamp_valid": True,
        "timestamp_source": "exchange_timestamp",
        "timestamp": stamp,
        "depth": {
            "buy": [{"price": 105.0, "quantity": 65}],
            "sell": [{"price": 105.2, "quantity": 65}],
        },
    }

    def sample(depth_qty: int, event_offset: float) -> dict:
        bracket = BracketState(
            entry_order_id="shadow-test",
            symbol=symbol,
            side="BUY",
            quantity=65,
            remaining_quantity=65,
            entry_price=100.0,
            sl_trigger_price=95.0,
            tp_trigger_price=120.0,
        )
        tick["depth"]["buy"][0]["quantity"] = depth_qty
        manager._record_trade_quality_shadow(
            bracket, 105.1, tick=tick,
            exchange_ts=stamp.timestamp() + event_offset,
        )
        return bracket.trailing_config["_research_executable_path"][-1]

    qualified = sample(65, 0.0)
    assert qualified["source"] == "bid"
    assert qualified["full_lot_bid_proven"] is True
    assert qualified["market_age_ms"] is not None
    assert qualified["required_qty"] == 65

    too_thin = sample(20, 0.0)
    assert too_thin["bid"] == 105.0
    assert too_thin["full_lot_bid_proven"] is False

    wrong_tick = sample(65, 1.0)
    assert wrong_tick["full_lot_bid_proven"] is False
    assert wrong_tick["bid"] == 105.0

    tick["source_timestamp_valid"] = False
    stale_clock = sample(65, 0.0)
    assert stale_clock["full_lot_bid_proven"] is False


