from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from nifty_scalper_bot.strategies.elite_strategies.config_models import VWAPProStrategyConfig
from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import VWAPProStrategy
from nifty_scalper_bot.strategies.signal_identity import deterministic_signal_id


class _HistoryEngine:
    def __init__(self, rows):
        self.rows = list(rows)

    def get_history(self, _symbol, field=None):
        assert field in (None, "bars")
        return list(self.rows)


def _bar(
    ts: str,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 1000.0,
):
    return {
        "timestamp": datetime.fromisoformat(ts).replace(tzinfo=ZoneInfo("Asia/Kolkata")),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
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
    assert "2026-09-08 04:31:00+00:00" in signal.metadata["setup_id"]


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


def test_recovery_does_not_compare_old_bars_with_latest_vwap(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("VWAP_THESIS_RECOVERY_LOOKBACK_BARS", "5")
    history = [
        _bar(
            f"2026-09-08T09:{minute:02d}:00",
            open_=90.0,
            high=90.0,
            low=90.0,
            close=90.0,
            volume=100.0,
        )
        for minute in range(15, 20)
    ]
    history.extend(
        _bar(
            f"2026-09-08T09:{minute:02d}:00",
            open_=110.0,
            high=111.0,
            low=109.8,
            close=110.5,
            volume=10.0,
        )
        for minute in range(20, 24)
    )
    history.append(
        _bar(
            "2026-09-08T09:24:00",
            open_=130.0,
            high=131.0,
            low=129.5,
            close=130.5,
            volume=2000.0,
        )
    )
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _HistoryEngine(history))
    indicators = _indicators(
        session_date="2026-09-08",
        latest_bar_ts="2026-09-08T09:25:00+05:30",
    )
    indicators.update(
        {
            "vwap": 120.0,
            "open": 130.0,
            "high": 132.0,
            "low": 129.0,
            "close": 131.0,
        }
    )

    signal = strategy._evaluate_signal(
        "NFO:NIFTY2690823650CE",
        indicators,
        131.0,
    )

    assert signal is None
    assert strategy.last_no_vote_reason == "vwap_thesis_not_armed"


@pytest.mark.parametrize(
    "live_anchor",
    [
        "2026-09-08T10:00:00+05:30",
        "2026-09-08T04:30:00Z",
        datetime.fromisoformat("2026-09-08T04:30:00+00:00").timestamp(),
        datetime.fromisoformat("2026-09-08T04:30:00+00:00").timestamp() * 1000,
        "1788841800000.0",
        datetime.fromisoformat("2026-09-08T04:30:00+00:00"),
        datetime.fromisoformat("2026-09-08T04:30:00"),
    ],
)
@pytest.mark.parametrize("reset_close", [99.5, 103.0])
def test_restart_preserves_identity_for_equivalent_anchor_formats(
    monkeypatch, live_anchor, reset_close
):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    symbol = "NFO:NIFTY2690823650CE"
    reset = _bar("2026-09-08T10:00:00", open_=101, high=104, low=99, close=reset_close)
    live = VWAPProStrategy(VWAPProStrategyConfig(), _HistoryEngine([]))
    reset_indicators = _indicators(session_date="2026-09-08", latest_bar_ts=live_anchor)
    reset_indicators.update({k: reset[k] for k in ("open", "high", "low", "close")})
    initial = live._evaluate_signal(symbol, reset_indicators, reset_close)
    if reset_close < 100.0:
        assert initial is None
    else:
        assert initial is not None
    current = _indicators(
        session_date="2026-09-08", latest_bar_ts="2026-09-08T10:02:00+05:30"
    )
    uninterrupted = live._evaluate_signal(symbol, current, 103.0)
    restarted = VWAPProStrategy(VWAPProStrategyConfig(), _HistoryEngine([reset]))
    recovered = restarted._evaluate_signal(symbol, current, 103.0)

    assert uninterrupted is not None and recovered is not None
    # Preserve the existing ID representation emitted by UTC indicator bars.
    assert uninterrupted.metadata["setup_id"] == (
        f"vwap:CE:2026-09-08 04:30:00+00:00:2026-09-08:{symbol}"
    )
    assert uninterrupted.metadata["setup_id"] == recovered.metadata["setup_id"]
    assert deterministic_signal_id(uninterrupted) == deterministic_signal_id(recovered)
