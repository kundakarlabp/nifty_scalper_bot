from __future__ import annotations

import logging

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


class DummyBroker: ...


class DummyWs: ...


class Bus:
    running = True


def test_mdm_delivery_diagnostic_reflects_bus(caplog) -> None:
    mdm = MarketDataManager(DummyBroker(), DummyWs())
    mdm.bus = Bus()
    caplog.set_level(logging.INFO)

    mdm._emit_tick('NSE:NIFTY', {'symbol': 'NSE:NIFTY', 'ltp': 100.0}, source='ws')  # noqa: SLF001

    rec = next(r for r in caplog.records if 'MDM_TICK_EMITTED_FIRST' in r.message)
    assert 'direct_delivered=False' in rec.message
    assert 'bus_running=True' in rec.message
    assert ' delivered=' not in rec.message
