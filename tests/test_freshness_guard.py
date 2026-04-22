from __future__ import annotations

import time

import pytest

from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.streaming.polling_streamer import PollingStreamer
from nifty_scalper_bot.quotes.freshness import FreshnessGuard
from nifty_scalper_bot.utils.errors import DataStaleError


def test_ensure_fresh_raises_on_stale_quote() -> None:
    guard = FreshnessGuard(stale_threshold_ms=100)
    stale_quote = {
        "symbol": "NIFTY",
        "ltp": 100.0,
        "ts_ms": int(time.time() * 1000) - 200,
    }
    with pytest.raises(DataStaleError):
        guard.ensure_fresh(stale_quote)


def test_market_data_age_ms_and_quote_age_use_canonical_symbol() -> None:
    mdm = MarketDataManager(broker=None, websocket=None, resolver=None)
    assert mdm.data_age_ms() >= 1_000_000_000
    now = time.time()
    with mdm._authoritative_tick_lock:  # noqa: SLF001
        mdm.last_tick_time = now
    assert mdm.data_age_ms() < 500
    with mdm._lock:  # noqa: SLF001
        mdm._last_quote_ts_ms["NSE:NIFTY"] = mdm._now_ms() - 150.0  # noqa: SLF001
        mdm._symbol_by_token[256265] = "NSE:NIFTY"  # noqa: SLF001
    assert 0 <= mdm.quote_age_ms("nse:nifty") < 5_000
    assert 0 <= mdm.quote_age_ms(256265) < 5_000


def test_polling_streamer_updates_authoritative_ticks(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyMDM:
        def __init__(self) -> None:
            self.calls = 0

        def update_authoritative_ticks(self, ticks):  # noqa: ANN001
            self.calls += len(ticks)

    class _DummyHub:
        def __init__(self, mdm) -> None:  # noqa: ANN001
            self._mdm = mdm

        def is_ws_fresh(self, _symbol: str) -> bool:
            return False

        def get_quote(self, _symbol: str):  # noqa: ANN001
            return None

    class _Resolver:
        def resolve_symbol(self, token: int) -> str:
            return f"NSE:T{token}"

        def get_symbol(self, token: int) -> str:
            return f"NSE:T{token}"

    class _Broker:
        pass

    updates: list[dict[str, float]] = []
    mdm = _DummyMDM()
    streamer = PollingStreamer(
        broker_client=_Broker(),
        on_tick=lambda tick: updates.append(dict(tick)),
        instrument_resolver=_Resolver(),
        data_hub=_DummyHub(mdm),
        poll_interval_ms=500,
        batch_size=10,
    )
    streamer.subscribe([101])
    monkeypatch.setattr(
        streamer,
        "_fetch_ticks",
        lambda _batch: [
            {
                "instrument_token": 101,
                "last_price": 100.0,
                "symbol": "NSE:T101",
            }
        ],
    )
    monkeypatch.setattr(streamer, "_should_escalate_starvation", lambda **_kwargs: False)

    def _stop_after_first_sleep(_seconds: float) -> None:
        streamer._stop.set()  # noqa: SLF001

    monkeypatch.setattr("nifty_scalper_bot.streaming.polling_streamer.time.sleep", _stop_after_first_sleep)
    streamer._run()  # noqa: SLF001
    assert mdm.calls == 1
    assert len(updates) == 1


def test_disable_rest_polling_stays_disabled_in_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    mdm = MarketDataManager(broker=None, websocket=None, resolver=None)
    mdm.disable_rest_polling(reason="test")
    assert mdm._rest_poll_enabled is False  # noqa: SLF001
    monkeypatch.setattr(mdm, "_is_ws_healthy", lambda: False)
    monkeypatch.setattr(
        "nifty_scalper_bot.data.market_data_manager.time.sleep",
        lambda _seconds: mdm._rest_poll_stop.set(),  # noqa: SLF001
    )
    mdm._rest_poll_loop()  # noqa: SLF001
    assert mdm._rest_poll_enabled is False  # noqa: SLF001
