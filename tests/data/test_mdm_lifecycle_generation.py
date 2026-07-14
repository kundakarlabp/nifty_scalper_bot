from __future__ import annotations

import time

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


class _FakeWebSocket:
    def __init__(self, *, connected: bool = True) -> None:
        self.calls = 0
        self.tokens: list[int] = []
        self.connected = connected

    def set_tokens(self, tokens):
        self.tokens = list(tokens)
        return True

    def is_connected(self) -> bool:
        return self.connected

    def force_reconnect(self) -> None:
        self.calls += 1


def _subscribe(mdm: MarketDataManager, symbol: str, token: int) -> None:
    assert mdm.request_token_subscription(token, symbol=symbol)
    mdm._confirmed_subscriptions.add(token)


def test_current_generation_first_tick_required_after_subscription() -> None:
    mdm = MarketDataManager(kite=None, websocket=_FakeWebSocket())
    symbol = "NFO:NIFTY26JUN24000CE"
    token = 24000

    _subscribe(mdm, symbol, token)

    pending = mdm.classify_live_tick_readiness(symbol, token, max_age_s=60.0)
    assert pending["ready"] is False
    assert pending["reason"] == "never_received_tick"
    assert pending["tick_age_s"] is None

    mdm._emit_tick(
        symbol,
        {
            "symbol": symbol,
            "instrument_token": token,
            "ltp": 10.0,
            "timestamp": time.time(),
        },
        source="ws",
    )

    ready = mdm.classify_live_tick_readiness(symbol, token, max_age_s=60.0)
    assert ready["ready"] is True
    assert ready["reason"] == "ready"
    assert ready["first_tick_received"] is True


def test_reconnect_generation_invalidates_old_tick_until_new_tick_arrives(monkeypatch) -> None:
    ws = _FakeWebSocket()
    mdm = MarketDataManager(kite=None, websocket=ws)
    symbol = "NFO:NIFTY26JUN24000CE"
    token = 24000
    _subscribe(mdm, symbol, token)
    mdm._emit_tick(
        symbol,
        {
            "symbol": symbol,
            "instrument_token": token,
            "ltp": 10.0,
            "timestamp": time.time(),
        },
        source="ws",
    )
    assert mdm.classify_live_tick_readiness(symbol, token, max_age_s=60.0)["ready"]
    monkeypatch.setattr(mdm, "_reconcile_ws_subscriptions", lambda: None)

    mdm._trigger_zombie_ws_restart()

    blocked = mdm.classify_live_tick_readiness(symbol, token, max_age_s=60.0)
    assert blocked["ready"] is False
    assert blocked["reason"] == "never_received_tick"

    mdm._emit_tick(
        symbol,
        {
            "symbol": symbol,
            "instrument_token": token,
            "ltp": 10.5,
            "timestamp": time.time(),
        },
        source="ws",
    )
    assert mdm.classify_live_tick_readiness(symbol, token, max_age_s=60.0)["ready"]


def test_zombie_restart_is_single_flight() -> None:
    ws = _FakeWebSocket()
    mdm = MarketDataManager(kite=None, websocket=ws)
    mdm._ws_restart_inflight = True
    mdm._ws_restart_generation = 7

    mdm._trigger_zombie_ws_restart()

    assert ws.calls == 0
    assert mdm._ws_restart_generation == 7


def test_restart_connection_without_expected_token_is_recovery_failure(monkeypatch) -> None:
    ws = _FakeWebSocket(connected=True)
    mdm = MarketDataManager(kite=None, websocket=ws)
    symbol = "NFO:NIFTY26JUN24000CE"
    token = 24000
    _subscribe(mdm, symbol, token)
    mdm._dispatched_subscriptions.clear()
    mdm._confirmed_subscriptions.clear()
    monkeypatch.setattr(mdm, "_reconcile_ws_subscriptions", lambda: None)

    mdm._trigger_zombie_ws_restart()

    assert ws.calls == 1
    recovery = mdm.verify_websocket_recovery()
    assert recovery["ok"] is False
    assert recovery["reason"] == "expected_tokens_not_restored"
    assert (
        mdm.classify_live_tick_readiness(symbol, token, max_age_s=60.0)["ready"]
        is False
    )


def test_volume_baseline_resets_on_subscription_generation() -> None:
    mdm = MarketDataManager(kite=None, websocket=_FakeWebSocket())
    symbol = "NFO:NIFTY26JUN24000CE"
    token = 24000
    _subscribe(mdm, symbol, token)

    first = mdm._normalise_tick_volume_delta(
        symbol, {"volume_traded_today": 6_000_000}
    )
    second = mdm._normalise_tick_volume_delta(
        symbol, {"volume_traded_today": 6_000_125}
    )
    assert first["volume_delta"] == 0
    assert second["volume_delta"] == 125

    mdm.request_token_subscription(token, symbol=symbol)
    after_generation_change = mdm._normalise_tick_volume_delta(
        symbol, {"volume_traded_today": 29_364_530}
    )
    assert after_generation_change["volume_delta"] == 0
    assert after_generation_change["volume_delta_untrusted"] is False


def test_stop_clears_restart_inflight_state() -> None:
    mdm = MarketDataManager(kite=None, websocket=_FakeWebSocket())
    mdm._started = True
    mdm._ws_restart_inflight = True
    mdm._ws_restart_started_at = time.monotonic()

    mdm.stop()

    assert mdm._ws_restart_inflight is False
    assert mdm._ws_restart_started_at is None
