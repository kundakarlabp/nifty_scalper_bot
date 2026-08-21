"""Regression coverage for stale futures market-event recovery."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nifty_scalper_bot.core import polling_failover_runtime as runtime
from nifty_scalper_bot.streaming import polling_streamer as polling_module
from nifty_scalper_bot.streaming.polling_streamer import PollingStreamer
from nifty_scalper_bot.utils.market_hours import MarketState


class _Fallback:
    def __init__(self) -> None:
        self.running = False
        self.mode_calls: list[bool] = []
        self.starts = 0

    def is_running(self) -> bool:
        return self.running

    def set_websocket_mode(self, enabled: bool) -> None:
        self.mode_calls.append(bool(enabled))

    def start(self) -> None:
        self.starts += 1
        self.running = True


@pytest.mark.asyncio
async def test_futures_live_tick_stale_readiness_forces_existing_recovery_path() -> None:
    """Fresh packet arrivals must not suppress recovery of a stale market event."""
    ctx = SimpleNamespace(
        live_block_reason="execution_not_armed:futures_live_tick_stale",
        websocket_manager=SimpleNamespace(is_connected=lambda: True),
        market_data_manager=SimpleNamespace(
            trading_feed_health=lambda: {
                "lagging": False,
                "futures_fresh": True,
                "options_fresh": True,
            },
            data_age_ms=lambda: 100.0,
        ),
    )
    fallback = _Fallback()

    await runtime._polling_failover_supervisor_iteration(
        ctx,
        fallback,
        quote_stale_ms=120_000.0,
        degraded_since=0.0,
        recovered_since=None,
        activate_after=0.0,
        _app_module=SimpleNamespace(is_market_open_now=lambda: True),
    )

    assert fallback.mode_calls == [False]
    assert fallback.starts == 1


def _run_one_poll_cycle(
    monkeypatch: pytest.MonkeyPatch, *, websocket_mode_enabled: bool
) -> MagicMock:
    hub = SimpleNamespace(
        get_quote=lambda _symbol: {"arrival_time": 1.0},
        is_ws_fresh=lambda _symbol: True,
    )
    poller = PollingStreamer(
        broker_client=object(),
        on_tick=lambda _tick: None,
        instrument_resolver=object(),
        data_hub=hub,
        poll_interval_ms=500,
    )
    poller._tokens = {123}
    poller._websocket_mode_enabled = websocket_mode_enabled
    poller._resolve_instrument = lambda _token: "NFO:NIFTY26AUGFUT"
    fetch = MagicMock(
        return_value=[
            {
                "instrument_token": 123,
                "last_price": 25_000.0,
                "symbol": "NFO:NIFTY26AUGFUT",
            }
        ]
    )
    poller._fetch_ticks = fetch

    monkeypatch.setattr(polling_module, "get_market_state", lambda: MarketState.OPEN)
    monkeypatch.setattr(polling_module.time, "sleep", lambda _seconds: poller._stop.set())

    poller._run()
    return fetch


def test_active_polling_recovery_does_not_defer_to_fresh_ws_arrival(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once fallback is active it must perform the REST recovery it was started for."""
    fetch = _run_one_poll_cycle(monkeypatch, websocket_mode_enabled=False)

    fetch.assert_called_once()


def test_websocket_standby_still_skips_fresh_ws_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal healthy-WS standby behavior remains unchanged."""
    fetch = _run_one_poll_cycle(monkeypatch, websocket_mode_enabled=True)

    fetch.assert_not_called()
