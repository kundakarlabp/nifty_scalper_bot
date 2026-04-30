from __future__ import annotations

import time

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


class DummyBroker:
    def get_quote(self, _symbol: str) -> dict[str, float]:
        return {'ltp': 0.0}


class DummyWebSocket:
    on_tick = None


def test_spot_token_updates_canonical_nse_nifty_freshness() -> None:
    manager = MarketDataManager(DummyBroker(), DummyWebSocket())
    before = manager.symbol_data_age_ms('NSE:NIFTY')

    manager._process_queued_tick(
        {
            'instrument_token': 256265,
            'last_price': 22500.0,
            'timestamp': time.time(),
        }
    )

    after = manager.symbol_data_age_ms('NSE:NIFTY')
    assert after < before
    assert manager.get_latest_tick('NSE:NIFTY') is not None


def test_spot_alias_resolves_to_same_canonical_symbol_age() -> None:
    manager = MarketDataManager(DummyBroker(), DummyWebSocket())
    manager._process_queued_tick(
        {
            'instrument_token': 256265,
            'last_price': 22510.0,
            'timestamp': time.time(),
        }
    )

    canonical_age = manager.symbol_data_age_ms('NSE:NIFTY')
    alias_age = manager.symbol_data_age_ms('nifty')

    assert alias_age >= 0
    assert abs(canonical_age - alias_age) < 100


def test_spot_aliases_and_snapshot_for_token_lookup() -> None:
    manager = MarketDataManager(DummyBroker(), DummyWebSocket())
    ltp = 24177.65
    manager._process_queued_tick(
        {
            'instrument_token': 256265,
            'last_price': ltp,
            'timestamp': time.time(),
        }
    )

    assert manager.symbol_has_tick('NSE:NIFTY') is True
    assert manager.symbol_has_tick(256265) is True
    assert manager.symbol_has_tick('NIFTY') is True
    assert manager.symbol_data_age_ms_or_none('NSE:NIFTY') is not None
    assert manager.symbol_data_age_ms_or_none(256265) is not None
    snap = manager.get_symbol_snapshot('NSE:NIFTY')
    assert snap.ltp == ltp

