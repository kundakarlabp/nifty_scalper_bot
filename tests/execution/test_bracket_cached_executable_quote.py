from __future__ import annotations

from datetime import datetime, timezone
from time import time
from unittest.mock import Mock

from nifty_scalper_bot.execution.bracket_core import BracketState
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


def _active_manager_for_quote_identity() -> tuple[BracketManager, str]:
    symbol = "NFO:NIFTY26AUG24050PE"
    manager = _manager(
        _CachedMarketData(
            {
                "symbol": symbol,
                "ltp": 111.0,
                "bid": 110.0,
                "ask": 112.0,
                "timestamp": 1_002.0,
            }
        )
    )
    manager.register_virtual_bracket(
        "quote-identity",
        symbol,
        "BUY",
        65,
        110.0,
        100.0,
        140.0,
        activate_immediately=True,
    )
    manager._fire_exits_batch = Mock()
    return manager, symbol


def test_new_depthless_tick_cannot_reuse_old_bid_to_mask_hard_stop() -> None:
    manager, symbol = _active_manager_for_quote_identity()
    manager._exit_quotes[symbol] = (108.0, 108.2, time())

    manager.on_tick(
        symbol,
        99.0,
        exchange_ts=1_001.0,
        defer_submission=True,
        tick={"symbol": symbol, "ltp": 99.0, "timestamp": 1_001.0},
    )

    assert symbol not in manager._exit_quotes
    manager._fire_exits_batch.assert_called_once()
    action = manager._fire_exits_batch.call_args.args[0][0][1]
    assert action["type"] == "SL"
    assert action["trigger_price_source"] == "ltp"


def test_new_depthless_tick_cannot_reuse_old_bid_to_fake_take_profit() -> None:
    manager, symbol = _active_manager_for_quote_identity()
    manager._exit_quotes[symbol] = (141.0, 141.2, time())

    manager.on_tick(
        symbol,
        110.0,
        exchange_ts=1_001.0,
        defer_submission=True,
        tick={"symbol": symbol, "ltp": 110.0, "timestamp": 1_001.0},
    )

    assert symbol not in manager._exit_quotes
    manager._fire_exits_batch.assert_not_called()


def test_tick_from_different_option_cannot_contaminate_protective_quote() -> None:
    manager, symbol = _active_manager_for_quote_identity()

    manager.on_tick(
        symbol,
        108.0,
        exchange_ts=1_001.0,
        defer_submission=True,
        tick={
            "symbol": "NFO:NIFTY26AUG24050CE",
            "ltp": 108.0,
            "bid": 95.0,
            "ask": 96.0,
            "timestamp": 1_001.0,
        },
    )

    assert symbol not in manager._exit_quotes
    manager._fire_exits_batch.assert_not_called()


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
            bracket,
            105.1,
            tick=tick,
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


def test_rejected_prefill_tick_cannot_poison_watchdog_exit_quote() -> None:
    """The core pre-fill tick veto must also protect the cached BBO."""
    symbol = "NFO:NIFTY26AUG24050PE"
    manager = _manager(_CachedMarketData({}))
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)
    manager.register_virtual_bracket(
        order_id="prefill-quote-guard",
        symbol=symbol,
        side="BUY",
        qty=65,
        price=110.0,
        sl=100.0,
        tp=130.0,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("prefill-quote-guard")
    assert bracket is not None
    bracket.entry_fill_ts = time()
    bracket.last_ltp = 110.0
    bracket.previous_ltp = 110.0
    manager._exit_quotes[symbol] = (110.0, 110.2, time())
    delayed_event = bracket.entry_fill_ts - 2.0

    manager.on_tick(
        symbol,
        90.0,
        exchange_ts=delayed_event,
        tick={
            "symbol": symbol,
            "ltp": 90.0,
            "bid": 89.9,
            "ask": 90.2,
            "exchange_timestamp": delayed_event,
        },
    )

    assert bracket.last_ltp == 110.0
    assert manager._executable_exit_price(bracket, 110.0) == (110.0, "bid")
    assert manager._sl_crossed(bracket, bracket.last_ltp) is False

    # The public event adapter previously wrote the quote even earlier,
    # before it forwarded this same event to BoundBracketManager.on_tick.
    manager.on_tick_event(
        {
            "symbol": symbol,
            "ltp": 90.0,
            "bid": 89.9,
            "ask": 90.2,
            "exchange_timestamp": delayed_event,
        }
    )
    assert bracket.last_ltp == 110.0
    assert manager._executable_exit_price(bracket, 110.0) == (110.0, "bid")
    assert manager._sl_crossed(bracket, bracket.last_ltp) is False
