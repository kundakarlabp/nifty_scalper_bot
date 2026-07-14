from __future__ import annotations

import threading
import time

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


class _FakeWebSocket:
    def __init__(self, *, connected: bool = True, raises: bool = False) -> None:
        self.calls = 0
        self.tokens: list[int] = []
        self.connected = connected
        self.raises = raises

    def set_tokens(self, tokens):
        self.tokens = list(tokens)
        return True

    def is_connected(self) -> bool:
        return self.connected

    def force_reconnect(self) -> None:
        self.calls += 1
        if self.raises:
            raise RuntimeError("boom")


class _NoReconnectWebSocket:
    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.connected = True

    def set_tokens(self, tokens):
        self.tokens = list(tokens)
        return True

    def is_connected(self) -> bool:
        return self.connected


def _subscribe(mdm: MarketDataManager, symbol: str, token: int) -> None:
    assert mdm.request_token_subscription(token, symbol=symbol)
    mdm._confirmed_subscriptions.add(token)


def _ws_tick(mdm: MarketDataManager, symbol: str, token: int, ltp: float = 10.0) -> None:
    mdm._emit_tick(
        symbol,
        {
            "symbol": symbol,
            "instrument_token": token,
            "ltp": ltp,
            "timestamp": time.time(),
        },
        source="ws",
    )


def test_current_generation_first_tick_required_after_subscription() -> None:
    mdm = MarketDataManager(kite=None, websocket=_FakeWebSocket())
    symbol = "NFO:NIFTY26JUN24000CE"
    token = 24000

    _subscribe(mdm, symbol, token)

    pending = mdm.classify_live_tick_readiness(symbol, token, max_age_s=60.0)
    assert pending["ready"] is False
    assert pending["reason"] == "never_received_tick"
    assert pending["tick_age_s"] is None

    _ws_tick(mdm, symbol, token)

    ready = mdm.classify_live_tick_readiness(symbol, token, max_age_s=60.0)
    assert ready["ready"] is True
    assert ready["reason"] == "ready"
    assert ready["current_generation_tick_received"] is True


def test_idempotent_subscription_does_not_reset_generation_tick_or_volume() -> None:
    mdm = MarketDataManager(kite=None, websocket=_FakeWebSocket())
    symbol = "NFO:NIFTY26JUN24000CE"
    token = 24000
    _subscribe(mdm, symbol, token)
    _ws_tick(mdm, symbol, token)
    generation = mdm._symbol_subscription_generation[symbol]
    first = mdm._normalise_tick_volume_delta(symbol, {"volume_traded_today": 1_000})
    second = mdm._normalise_tick_volume_delta(symbol, {"volume_traded_today": 1_025})
    assert first["volume_delta"] == 0
    assert second["volume_delta"] == 25

    assert mdm.request_token_subscription(token, symbol=symbol)

    assert mdm._symbol_subscription_generation[symbol] == generation
    assert mdm.classify_live_tick_readiness(symbol, token, max_age_s=60.0)["ready"]
    third = mdm._normalise_tick_volume_delta(symbol, {"volume_traded_today": 1_050})
    assert third["volume_delta"] == 25


def test_token_change_resets_generation_and_requires_new_matching_tick() -> None:
    mdm = MarketDataManager(kite=None, websocket=_FakeWebSocket())
    symbol = "NFO:NIFTY26JUN24000CE"
    old_token = 24000
    new_token = 24001
    _subscribe(mdm, symbol, old_token)
    _ws_tick(mdm, symbol, old_token)
    old_generation = mdm._symbol_subscription_generation[symbol]
    mdm._normalise_tick_volume_delta(symbol, {"volume_traded_today": 2_000})

    assert mdm.request_token_subscription(new_token, symbol=symbol)

    assert mdm._symbol_subscription_generation[symbol] > old_generation
    assert mdm.classify_live_tick_readiness(symbol, new_token, max_age_s=60.0)["ready"] is False
    baseline = mdm._normalise_tick_volume_delta(symbol, {"volume_traded_today": 29_364_530})
    assert baseline["volume_delta"] == 0
    assert baseline["volume_delta_untrusted"] is False

    _ws_tick(mdm, symbol, old_token, ltp=9.5)
    blocked = mdm.classify_live_tick_readiness(symbol, new_token, max_age_s=60.0)
    assert blocked["ready"] is False
    assert blocked["reason"] == "never_received_tick"

    _ws_tick(mdm, symbol, new_token, ltp=10.5)
    assert mdm.classify_live_tick_readiness(symbol, new_token, max_age_s=60.0)["ready"]


def test_reconnect_remains_inflight_until_current_generation_tick() -> None:
    ws = _FakeWebSocket(connected=True)
    mdm = MarketDataManager(kite=None, websocket=ws)
    symbol = "NFO:NIFTY26JUN24000CE"
    token = 24000
    _subscribe(mdm, symbol, token)
    _ws_tick(mdm, symbol, token)

    mdm._trigger_zombie_ws_restart()

    assert ws.calls == 1
    pending = mdm.verify_websocket_recovery()
    assert pending["ok"] is False
    assert pending["state"] == "waiting_for_first_ticks"
    assert pending["inflight"] is True
    assert mdm.classify_live_tick_readiness(symbol, token, max_age_s=60.0)["ready"] is False

    _ws_tick(mdm, symbol, token)
    recovered = mdm.verify_websocket_recovery()
    assert recovered["ok"] is True
    assert recovered["state"] == "recovered"
    assert recovered["inflight"] is False
    assert mdm.classify_live_tick_readiness(symbol, token, max_age_s=60.0)["ready"]


def test_concurrent_duplicate_restart_callers_start_one_reconnect() -> None:
    ws = _FakeWebSocket(connected=True)
    mdm = MarketDataManager(kite=None, websocket=ws)
    _subscribe(mdm, "NFO:NIFTY26JUN24000CE", 24000)
    mdm._zombie_last_restart_attempt_at = -10_000.0

    threads = [threading.Thread(target=mdm._trigger_zombie_ws_restart) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert ws.calls == 1
    assert mdm._ws_restart_inflight is True


def test_connected_dispatched_without_tick_is_not_recovered_at_deadline() -> None:
    ws = _FakeWebSocket(connected=True)
    mdm = MarketDataManager(kite=None, websocket=ws)
    symbol = "NFO:NIFTY26JUN24000CE"
    token = 24000
    _subscribe(mdm, symbol, token)
    mdm._ws_recovery_timeout_sec = 1.0

    mdm._trigger_zombie_ws_restart()
    mdm._dispatched_subscriptions.add(token)
    pending = mdm.verify_websocket_recovery(now_mono=(mdm._ws_restart_deadline_mono or 0) - 0.1)
    assert pending["ok"] is False
    assert pending["state"] == "waiting_for_first_ticks"

    failed = mdm.verify_websocket_recovery(now_mono=(mdm._ws_restart_deadline_mono or 0) + 0.1)
    assert failed["ok"] is False
    assert failed["state"] == "failed"
    assert mdm._ws_restart_inflight is False


def test_missing_force_reconnect_records_failure_and_clears_inflight() -> None:
    mdm = MarketDataManager(kite=None, websocket=_NoReconnectWebSocket())
    _subscribe(mdm, "NFO:NIFTY26JUN24000CE", 24000)

    mdm._trigger_zombie_ws_restart()

    assert mdm._ws_restart_inflight is False
    assert mdm._ws_restart_state == "failed"
    assert mdm._ws_restart_fail_reason == "force_reconnect_unavailable"


def test_force_reconnect_exception_records_failure_and_clears_inflight() -> None:
    mdm = MarketDataManager(kite=None, websocket=_FakeWebSocket(raises=True))
    _subscribe(mdm, "NFO:NIFTY26JUN24000CE", 24000)

    mdm._trigger_zombie_ws_restart()

    assert mdm._ws_restart_inflight is False
    assert mdm._ws_restart_state == "failed"
    assert mdm._ws_restart_fail_reason == "RuntimeError"


def test_shutdown_during_recovery_is_terminal_and_late_tick_cannot_recover() -> None:
    ws = _FakeWebSocket(connected=True)
    mdm = MarketDataManager(kite=None, websocket=ws)
    symbol = "NFO:NIFTY26JUN24000CE"
    token = 24000
    _subscribe(mdm, symbol, token)
    mdm._started = True
    mdm._trigger_zombie_ws_restart()

    mdm.stop()
    _ws_tick(mdm, symbol, token)

    state = mdm.verify_websocket_recovery()
    assert state["state"] == "failed"
    assert state["inflight"] is False
    assert mdm._ws_restart_fail_reason == "shutdown"


def test_runner_live_tick_classifier_uses_canonical_freshness_policy() -> None:
    runner_source = __import__("pathlib").Path(
        "src/nifty_scalper_bot/strategies/runner.py"
    ).read_text()
    classifier_call = runner_source[runner_source.index("live_tick_check = classifier") : runner_source.index("details[\"live_tick_generation_ready\"]")]
    assert "max_age_s=60.0" not in classifier_call
    assert "max_live_tick_age_s" in classifier_call
    assert "resolve_max_quote_age_seconds" in runner_source
