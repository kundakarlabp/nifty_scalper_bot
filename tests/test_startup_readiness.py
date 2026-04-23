from __future__ import annotations

from pathlib import Path

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


class DummyBroker:
    def get_quote(self, _symbol: str) -> dict[str, float]:
        return {'ltp': 0.0}


class DummyWebSocket:
    on_tick = None


def _manager() -> MarketDataManager:
    manager = MarketDataManager(DummyBroker(), DummyWebSocket())
    manager.set_readiness_requirements(
        spot_symbol='NSE:NIFTY',
        futures_symbol='NFO:NIFTYFUT',
        atm_ce_symbol='NFO:NIFTYATMCE',
        atm_pe_symbol='NFO:NIFTYATMPE',
        option_symbols=['NFO:NIFTYATMCE', 'NFO:NIFTYATMPE'],
    )
    return manager


def test_hard_ready_with_missing_spot_is_allowed_as_degraded_spot() -> None:
    manager = _manager()
    bars = {
        'NFO:NIFTYFUT': 120,
        'NFO:NIFTYATMCE': 120,
        'NFO:NIFTYATMPE': 120,
        'NSE:NIFTY': 0,
    }

    readiness = manager._readiness_state(
        bars, min_bars=50, requirements=manager._readiness_requirements
    )

    assert readiness['hard_ready'] is True
    assert readiness['spot_ready'] is False
    assert readiness['missing_hard'] == []


def test_timeout_with_hard_requirements_met_reports_degraded_readiness() -> None:
    manager = _manager()
    manager._last_readiness_state = {
        'hard_ready': True,
        'spot_ready': False,
        'missing_hard': [],
    }

    snapshot = manager.readiness_state_snapshot()

    assert snapshot['hard_ready'] is True
    assert snapshot['spot_ready'] is False


def test_timeout_without_hard_requirements_reports_incomplete() -> None:
    manager = _manager()
    bars = {
        'NFO:NIFTYFUT': 10,
        'NFO:NIFTYATMCE': 120,
        'NFO:NIFTYATMPE': 5,
        'NSE:NIFTY': 120,
    }

    readiness = manager._readiness_state(
        bars, min_bars=50, requirements=manager._readiness_requirements
    )

    assert readiness['hard_ready'] is False
    assert 'futures' in readiness['missing_hard']
    assert 'atm_pe' in readiness['missing_hard']


def test_app_startup_uses_hard_ready_for_trading_indicators_gate() -> None:
    source = Path('src/nifty_scalper_bot/core/app.py').read_text(encoding='utf-8')

    assert 'indicators_ready_for_trading = hard_ready' in source
    assert 'ctx.market_regime_manager.indicators_ready = (' in source
    assert 'ctx.data_hub.indicators_ready = (' in source
