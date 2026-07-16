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


def _ws_tick(
    mdm: MarketDataManager, symbol: str, token: int, ltp: float = 10.0
) -> None:
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
    assert pending["reason"] == "current_generation_tick_pending"
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
    assert (
        mdm.classify_live_tick_readiness(symbol, new_token, max_age_s=60.0)["ready"]
        is False
    )
    baseline = mdm._normalise_tick_volume_delta(
        symbol, {"volume_traded_today": 29_364_530}
    )
    assert baseline["volume_delta"] == 0
    assert baseline["volume_delta_untrusted"] is False

    _ws_tick(mdm, symbol, old_token, ltp=9.5)
    blocked = mdm.classify_live_tick_readiness(symbol, new_token, max_age_s=60.0)
    assert blocked["ready"] is False
    assert blocked["reason"] in {
        "subscription_not_confirmed",
        "current_generation_tick_pending",
    }

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
    assert (
        mdm.classify_live_tick_readiness(symbol, token, max_age_s=60.0)["ready"]
        is False
    )

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

    threads = [
        threading.Thread(target=mdm._trigger_zombie_ws_restart) for _ in range(2)
    ]
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
    pending = mdm.verify_websocket_recovery(
        now_mono=(mdm._ws_restart_deadline_mono or 0) - 0.1
    )
    assert pending["ok"] is False
    assert pending["state"] == "waiting_for_first_ticks"

    failed = mdm.verify_websocket_recovery(
        now_mono=(mdm._ws_restart_deadline_mono or 0) + 0.1
    )
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


def test_failed_recovery_late_tick_remains_failed() -> None:
    ws = _FakeWebSocket(raises=True)
    mdm = MarketDataManager(kite=None, websocket=ws)
    symbol = "NFO:NIFTY26JUN24000CE"
    token = 24000
    _subscribe(mdm, symbol, token)

    mdm._trigger_zombie_ws_restart()
    _ws_tick(mdm, symbol, token)

    state = mdm.verify_websocket_recovery()
    assert state["ok"] is False
    assert state["state"] == "failed"
    assert state["reason"] == "RuntimeError"


def test_timeout_late_tick_remains_failed() -> None:
    mdm = MarketDataManager(kite=None, websocket=_FakeWebSocket(connected=True))
    symbol = "NFO:NIFTY26JUN24000CE"
    token = 24000
    _subscribe(mdm, symbol, token)
    mdm._ws_recovery_timeout_sec = 1.0

    mdm._trigger_zombie_ws_restart()
    failed = mdm.verify_websocket_recovery(
        now_mono=(mdm._ws_restart_deadline_mono or 0) + 0.1
    )
    _ws_tick(mdm, symbol, token)

    state = mdm.verify_websocket_recovery()
    assert failed["state"] == "failed"
    assert state["ok"] is False
    assert state["state"] == "failed"
    assert state["reason"] == "waiting_for_current_generation_ticks"


def test_unresolved_desired_token_mapping_cannot_recover() -> None:
    mdm = MarketDataManager(kite=None, websocket=_FakeWebSocket(connected=True))
    with mdm._lock:
        mdm._desired_tokens.add(999999)

    mdm._trigger_zombie_ws_restart()

    state = mdm.verify_websocket_recovery()
    assert state["ok"] is False
    assert state["reason"] == "affected_token_mapping_incomplete"
    assert state["unresolved_tokens"] == [999999]


def test_concurrent_tick_and_restart_state_remains_atomic() -> None:
    mdm = MarketDataManager(kite=None, websocket=_FakeWebSocket(connected=True))
    symbol = "NFO:NIFTY26JUN24000CE"
    token = 24000
    _subscribe(mdm, symbol, token)

    restart_thread = threading.Thread(target=mdm._trigger_zombie_ws_restart)
    tick_thread = threading.Thread(target=_ws_tick, args=(mdm, symbol, token))
    restart_thread.start()
    tick_thread.start()
    restart_thread.join()
    tick_thread.join()

    state = mdm.verify_websocket_recovery()
    assert state["state"] in {"recovered", "waiting_for_first_ticks"}
    assert mdm._symbol_subscription_generation[symbol] >= 1


def test_successful_new_generation_clears_previous_failure_reason() -> None:
    mdm = MarketDataManager(kite=None, websocket=_FakeWebSocket(raises=True))
    symbol = "NFO:NIFTY26JUN24000CE"
    token = 24000
    _subscribe(mdm, symbol, token)
    mdm._trigger_zombie_ws_restart()
    assert mdm._ws_restart_fail_reason == "RuntimeError"

    mdm._ws = _FakeWebSocket(connected=True)
    mdm._zombie_last_restart_attempt_at = -10_000.0
    mdm._trigger_zombie_ws_restart()
    assert mdm._ws_restart_fail_reason is None
    _ws_tick(mdm, symbol, token)

    assert mdm.verify_websocket_recovery()["ok"] is True
    assert mdm._ws_restart_fail_reason is None


def test_runner_live_tick_classifier_uses_canonical_freshness_policy(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from nifty_scalper_bot.strategies import runner as runner_module

    captured: dict[str, object] = {}

    class _FakeMDM:
        def classify_live_tick_readiness(self, symbol, token, *, max_age_s):
            captured["symbol"] = symbol
            captured["token"] = token
            captured["max_age_s"] = max_age_s
            return {
                "symbol": symbol,
                "token": token,
                "tracked": True,
                "subscription_requested": True,
                "subscription_confirmed": True,
                "token_matches": True,
                "expected_generation": 1,
                "tick_generation": None,
                "current_generation_tick_received": False,
                "tick_age_s": None,
                "fresh": False,
                "ready": False,
                "reason": "sentinel_block",
            }

    class _FakeIndicator:
        def get_history(self, symbol):
            return [object()] * 5

    runner = object.__new__(runner_module.StrategyRunner)
    runner._market_data = _FakeMDM()
    runner._runtime_indicators = {"NFO:NIFTY26JUN24000CE": {}}
    runner._resolve_execution_mode_snapshot = lambda: SimpleNamespace(
        execution_mode="LIVE",
        is_live_mode=True,
        env_live_enabled=True,
        paper_enabled=False,
        shadow_mode_enabled=False,
        order_manager_live=True,
    )
    runner._order_manager_kill_switch_status_for_entry = lambda: (False, None)
    runner._get_cached_quote_for_live_entry = lambda symbol: {
        "bid": 10.0,
        "ask": 10.1,
        "tradable_quote": True,
    }
    runner._risk_kill_switch_triggered = lambda: False
    runner._runtime_live_orders_armed = True
    runner._runtime_readiness_reason = "ready"
    runner._runtime_data_hard_ready = True
    runner._runtime_evaluation_ready = True
    runner._logger = SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None
    )
    runner._is_tradable_symbol = lambda symbol: True
    runner._contract_side_from_symbol = lambda symbol: "CE"
    runner._active_selected_ce = "NFO:NIFTY26JUN24000CE"
    runner._active_selected_pe = "NFO:NIFTY26JUN24000PE"
    runner._required_bars_for_symbol = lambda symbol: 1
    runner._indicator_engine = _FakeIndicator()
    runner._is_option_symbol_tick_fresh = lambda symbol, max_age_s: True
    runner._live_entry_candidate_eligibility = lambda symbol, direction_bias: (
        True,
        "eligible",
        {},
    )
    runner._active_basket_token_by_symbol = {"NFO:NIFTY26JUN24000CE": "24000"}
    runner._order_manager = SimpleNamespace(resolve_lot_size=lambda symbol: 75)
    runner._resolve_order_manager_health_for_entry = lambda: (
        True,
        "ok",
        {"broker_ready": True},
    )
    monkeypatch.setattr(
        runner_module, "resolve_max_quote_age_seconds", lambda *a, **k: 3.25
    )

    ready, reason, _details = runner._symbol_live_entry_ready("NFO:NIFTY26JUN24000CE")

    assert ready is False
    assert reason == "sentinel_block"
    assert captured == {
        "symbol": "NFO:NIFTY26JUN24000CE",
        "token": "24000",
        "max_age_s": 3.25,
    }
