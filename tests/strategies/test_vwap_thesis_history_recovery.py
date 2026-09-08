from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from nifty_scalper_bot.strategies.elite_strategies.config_models import VWAPProStrategyConfig
from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import VWAPProStrategy


class _HistoryEngine:
    def __init__(self, rows):
        self.rows = list(rows)

    def get_history(self, _symbol, field=None):
        assert field in (None, "bars")
        return list(self.rows)


def _bar(ts: str, *, open_: float, high: float, low: float, close: float):
    return {
        "timestamp": datetime.fromisoformat(ts).replace(tzinfo=ZoneInfo("Asia/Kolkata")),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1000.0,
        "is_complete": True,
    }


def _indicators(*, session_date: str, latest_bar_ts: str):
    return {
        "vwap": 100.0,
        "atr": 5.0,
        "close": 103.0,
        "open": 102.0,
        "high": 103.5,
        "low": 101.5,
        "volume": 1200.0,
        "avg_volume": 900.0,
        "spread_pct": 0.3,
        "quote_depth_valid": True,
        "tradable_quote": True,
        "direction_bias": "CE",
        "underlying_direction_bias": "CE",
        "underlying_direction_confidence": 0.95,
        "context_age_seconds": 0.0,
        "stale_data_used": False,
        "session_date": session_date,
        "latest_bar_ts": latest_bar_ts,
    }


def test_new_strategy_instance_recovers_same_contract_same_session_thesis(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    history = [
        _bar("2026-09-08T10:00:00", open_=101.0, high=101.2, low=99.4, close=99.8),
        _bar("2026-09-08T10:01:00", open_=100.2, high=102.0, low=100.1, close=101.8),
    ]
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _HistoryEngine(history))

    signal = strategy._evaluate_signal(
        "NFO:NIFTY2690823650CE",
        _indicators(session_date="2026-09-08", latest_bar_ts="2026-09-08T10:02:00+05:30"),
        103.0,
    )

    assert signal is not None
    assert signal.metadata["thesis_recovered_from_history"] is True
    assert signal.metadata["thesis_scope_session"] == "2026-09-08"
    assert "2026-09-08 10:00:00" in signal.metadata["setup_id"]


def test_recovery_never_uses_prior_session_history(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    history = [
        _bar("2026-09-07T15:20:00", open_=101.0, high=101.2, low=99.0, close=99.5),
    ]
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _HistoryEngine(history))

    signal = strategy._evaluate_signal(
        "NFO:NIFTY2690823650CE",
        _indicators(session_date="2026-09-08", latest_bar_ts="2026-09-08T10:02:00+05:30"),
        103.0,
    )

    assert signal is None
    assert strategy.last_no_vote_reason == "vwap_thesis_not_armed"


def test_recovery_never_uses_another_contract(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _HistoryEngine([]))
    strategy._thesis_anchor_by_scope[("NFO:NIFTY2690823700CE", "2026-09-08")] = "other"

    signal = strategy._evaluate_signal(
        "NFO:NIFTY2690823650CE",
        _indicators(session_date="2026-09-08", latest_bar_ts="2026-09-08T10:02:00+05:30"),
        103.0,
    )

    assert signal is None
    assert strategy.last_no_vote_reason == "vwap_thesis_not_armed"
