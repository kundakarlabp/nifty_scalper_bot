from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.core import app as app_module
from nifty_scalper_bot.data.pipeline import TickValidator
from nifty_scalper_bot.streaming.websocket_manager import ConnectionState, WebSocketManager


def test_history_lookback_days_defined() -> None:
    source = (app_module.Path(app_module.__file__)).read_text(encoding='utf-8')
    assert 'def _history_lookback_days(required_bars: int) -> int:' in source
    assert source.index('def _history_lookback_days') < source.index('_history_lookback_days(required_bars)')


def test_history_lookback_days_executes() -> None:
    assert app_module._history_lookback_days(100) >= 2


def test_botcontext_slots_include_evaluation_ready() -> None:
    assert 'evaluation_ready' in app_module.BotContext.__slots__
    assert 'runner_task' in app_module.BotContext.__slots__


def test_tick_validator_prefers_exchange_timestamp() -> None:
    validator = TickValidator()
    tick = validator.validate(
        {
            'symbol': 'NSE:NIFTY',
            'timestamp': '2026-01-01T12:00:10Z',
            'exchange_timestamp': '2026-01-01T12:00:00Z',
            'ltp': 100.0,
        }
    )
    assert tick is not None
    assert tick.timestamp.isoformat().startswith('2026-01-01T12:00:00')


def test_ws_tick_restores_connected_state() -> None:
    manager = WebSocketManager(api_key='k', access_token='t', tokens=[])
    manager._state = ConnectionState.DISCONNECTED
    manager._connected.clear()
    manager._on_ticks(None, [{'instrument_token': 123, 'last_price': 1.0}])
    assert manager._connected.is_set()
    assert manager._state == ConnectionState.CONNECTED
    assert manager._stream_health == 'healthy'


async def test_instrument_manager_missing_defers_basket_not_failure() -> None:
    ctx = SimpleNamespace(
        instrument_manager=None,
        market_data_manager=object(),
        broker_client=object(),
    )
    result = await app_module._build_and_hydrate_live_basket_from_spot(
        ctx,
        spot_ltp=25000.0,
        configured_mode='LIVE',
    )
    assert result.get('deferred') is True
    assert result.get('reason') == 'instrument_manager_not_ready'
