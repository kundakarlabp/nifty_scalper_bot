from __future__ import annotations

import time

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


class DummyBroker:
    def get_quote(self, _symbol: str) -> dict[str, float]:
        return {"ltp": 0.0}


class DummyWebSocket:
    on_tick = None

    def subscribe_tokens(self, _tokens, mode: str = 'ltp') -> None:
        return None

    def set_tokens(self, _tokens) -> bool:
        return True

    def unsubscribe_tokens(self, _tokens) -> None:
        return None

    def connection_state(self):
        return None

    def is_connected(self) -> bool:
        return True


def _seed_tick(manager: MarketDataManager, symbol: str, age_seconds: float) -> None:
    now = time.time()
    manager._store_tick(
        symbol,
        {
            'symbol': symbol,
            'ltp': 100.0,
            'last_price': 100.0,
            'timestamp': now - age_seconds,
        },
    )
    manager._last_tick_time[symbol] = now - age_seconds
    manager._last_tick_wallclock[symbol] = now - age_seconds


def _build_manager() -> MarketDataManager:
    manager = MarketDataManager(DummyBroker(), DummyWebSocket())
    manager._tick_stale_threshold_ms = 5_000
    manager.set_readiness_requirements(
        spot_symbol='NSE:NIFTY',
        futures_symbol='NFO:NIFTYFUT',
        atm_ce_symbol='NFO:NIFTYATMCE',
        atm_pe_symbol='NFO:NIFTYATMPE',
        option_symbols=['NFO:NIFTYATMCE', 'NFO:NIFTYATMPE'],
    )
    return manager


def test_spot_stale_but_trading_feed_healthy() -> None:
    manager = _build_manager()
    _seed_tick(manager, 'NSE:NIFTY', age_seconds=9.0)
    _seed_tick(manager, 'NFO:NIFTYFUT', age_seconds=0.5)
    _seed_tick(manager, 'NFO:NIFTYATMCE', age_seconds=0.5)
    _seed_tick(manager, 'NFO:NIFTYATMPE', age_seconds=0.5)

    health = manager.trading_feed_health(max_age_ms=5_000)

    assert health['spot_fresh'] is False
    assert health['futures_fresh'] is True
    assert health['options_fresh'] is True
    assert health['trading_feed_healthy'] is True


def test_futures_stale_marks_trading_feed_unhealthy() -> None:
    manager = _build_manager()
    _seed_tick(manager, 'NSE:NIFTY', age_seconds=0.5)
    _seed_tick(manager, 'NFO:NIFTYFUT', age_seconds=9.0)
    _seed_tick(manager, 'NFO:NIFTYATMCE', age_seconds=0.5)
    _seed_tick(manager, 'NFO:NIFTYATMPE', age_seconds=0.5)

    health = manager.trading_feed_health(max_age_ms=5_000)

    assert health['futures_fresh'] is False
    assert health['trading_feed_healthy'] is False


def test_options_stale_marks_trading_feed_unhealthy() -> None:
    manager = _build_manager()
    _seed_tick(manager, 'NSE:NIFTY', age_seconds=0.5)
    _seed_tick(manager, 'NFO:NIFTYFUT', age_seconds=0.5)
    _seed_tick(manager, 'NFO:NIFTYATMCE', age_seconds=9.0)
    _seed_tick(manager, 'NFO:NIFTYATMPE', age_seconds=0.5)

    health = manager.trading_feed_health(max_age_ms=5_000)

    assert health['options_fresh'] is False
    assert health['trading_feed_healthy'] is False


def test_spot_stale_only_does_not_flip_trading_feed_health() -> None:
    manager = _build_manager()
    _seed_tick(manager, 'NSE:NIFTY', age_seconds=7.0)
    _seed_tick(manager, 'NFO:NIFTYFUT', age_seconds=0.2)
    _seed_tick(manager, 'NFO:NIFTYATMCE', age_seconds=0.2)
    _seed_tick(manager, 'NFO:NIFTYATMPE', age_seconds=0.2)

    assert manager.trading_feed_health(max_age_ms=5_000)['trading_feed_healthy'] is True


def test_trading_feed_health_handles_tuple_fields_without_calling() -> None:
    manager = _build_manager()
    manager._readiness_requirements["options"] = (  # noqa: SLF001
        "NFO:NIFTYATMCE",
        "NFO:NIFTYATMPE",
    )
    _seed_tick(manager, "NFO:NIFTYFUT", age_seconds=0.2)
    _seed_tick(manager, "NFO:NIFTYATMCE", age_seconds=0.2)
    _seed_tick(manager, "NFO:NIFTYATMPE", age_seconds=0.2)

    health = manager.trading_feed_health(max_age_ms=5_000)

    assert isinstance(health, dict)
    assert "futures_fresh" in health
    assert "options_fresh" in health
    assert "trading_feed_healthy" in health
    assert health["futures_fresh"] is True
    assert health["options_fresh"] is True
    assert health["trading_feed_healthy"] is True


def test_trading_feed_health_never_raises_when_requirements_malformed() -> None:
    manager = _build_manager()
    manager._readiness_requirements = {  # noqa: SLF001
        "options": ("NFO:XCE", "NFO:XPE"),
        "futures": ("NFO:FUT",),
        "spot": ("NSE:NIFTY",),
    }

    health = manager.trading_feed_health(max_age_ms=5_000)

    assert health["trading_feed_healthy"] is False
    assert health["error_type"] == "TypeError"
    assert "futures_symbol_type" in health["error"] or "spot_symbol_type" in health["error"]
