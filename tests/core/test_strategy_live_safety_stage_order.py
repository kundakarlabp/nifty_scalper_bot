from __future__ import annotations

import time
from types import SimpleNamespace

from nifty_scalper_bot.core import strategy_live_safety as guard
from nifty_scalper_bot.strategies.signal_generator import Signal


class Strategy:
    name = "test"
    config = {}

    def __init__(self, symbol: str):
        self.calls = 0
        self.symbol = symbol

    def get_required_indicators(self):
        return []

    def generate_signal(self, symbol, indicators, current_price, position=None):
        self.calls += 1
        return Signal(
            "BUY",
            self.symbol,
            1,
            0.9,
            "ok",
            90.0,
            120.0,
            metadata={"timestamp": time.time()},
        )


class Engine:
    def get_history(self, _symbol):
        ts = (guard.time.time() // 60) * 60 - 60
        return [
            {"timestamp": ts, "open": 1, "high": 1, "low": 1, "close": 1}
            for _ in range(5)
        ]

    def get_indicators(self, *_):
        return {}


class MDM:
    def __init__(self, *, age=1.0):
        self.age = age
        self.recovery_requests = []

    def get_ohlc_bars(self, symbol, *, limit=None):
        ts = (guard.time.time() // 60) * 60 - 60
        rows = [
            {"timestamp": ts, "open": 1, "high": 1, "low": 1, "close": 1}
            for _ in range(5)
        ]
        return rows[-limit:] if limit else rows

    def get_latest_closed_bar(self, symbol):
        return self.get_ohlc_bars(symbol, limit=1)[-1]

    def time_since_last_live_ws_tick(self, symbol):
        return self.age

    def request_fallback_refresh(self, symbol, *, reason):
        self.recovery_requests.append((symbol, reason))

    def get_active_contract_basket(self):
        return {
            "selected_ce": "NFO:NIFTY2662324050CE",
            "selected_pe": "NFO:NIFTY2662324050PE",
        }


def live_manager(strategy):
    from nifty_scalper_bot.core.strategy_manager import StrategyManager

    mgr = StrategyManager(
        [strategy], Engine(), SimpleNamespace(get_position=lambda _s: None)
    )
    mgr._required_candles = 2
    mgr._bar_interval_seconds = 60
    mgr._market_data_manager = MDM()
    now = time.time()
    mgr._latest_context_snapshots = {
        "spot_context": {"timestamp": now - 1},
        "futures_context": {"timestamp": now - 1},
    }
    return mgr


def live_env(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")


def test_underlying_evaluation_does_not_call_option_checks(monkeypatch):
    live_env(monkeypatch)
    called = []
    monkeypatch.setattr(
        guard,
        "_selected_contract_block",
        lambda _m, s: called.append(("selected", s)) or None,
    )
    monkeypatch.setattr(
        guard,
        "_live_option_tick_block",
        lambda _m, s: called.append(("tick", s)) or None,
    )
    strategy = Strategy("NFO:NIFTY2662324050CE")
    mgr = live_manager(strategy)

    mgr.generate_signal("NSE:NIFTY", 24000.0)

    assert ("selected", "NSE:NIFTY") not in called
    assert ("tick", "NSE:NIFTY") not in called


def test_generated_non_selected_candidate_fails_closed():
    mgr = live_manager(Strategy("NFO:NIFTY2662324100CE"))

    block = guard._candidate_execution_block(mgr, "NFO:NIFTY2662324100CE")

    assert block is not None
    assert block["reason"] == "live_selected_contract_mismatch"


def test_candidate_readiness_applies_to_final_filtered_signal_symbol(monkeypatch):
    checked = []
    ce_a = "NFO:NIFTY2662324050CE"
    ce_b = "NFO:NIFTY2662324050PE"
    signal = Signal(
        "BUY",
        ce_a,
        1,
        0.9,
        "ok",
        90.0,
        120.0,
        metadata={"timestamp": time.time(), "is_approved": True},
    )
    mgr = SimpleNamespace(
        _filter_signal=lambda _signal: True,
        _orchestrator=SimpleNamespace(
            filter_signal=lambda _signal, _metadata, _pm: Signal(
                _signal.action,
                ce_b,
                _signal.quantity,
                _signal.confidence,
                _signal.reason,
                _signal.stop_loss,
                _signal.take_profit,
                metadata={
                    **_signal.metadata,
                    "timestamp": time.time(),
                    "is_approved": False,
                },
            )
        ),
        _position_manager=None,
        _last_no_signal_decision_by_symbol={},
    )
    monkeypatch.setattr(
        guard, "_candidate_execution_block", lambda _m, s: checked.append(s) or None
    )

    filtered = guard._final_filter(mgr, guard._add_identity(signal), "trace")
    filtered = guard._add_identity(filtered)
    block = guard._candidate_execution_block(mgr, filtered.symbol)

    assert block is None
    assert checked == [ce_b]
