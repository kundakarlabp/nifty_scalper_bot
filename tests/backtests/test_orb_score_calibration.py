from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from scripts.analyze_orb_score_calibration import analyze
from nifty_scalper_bot.backtesting.strategy_research import (
    _research_orb_quality_context,
)


IST = ZoneInfo("Asia/Kolkata")


class _FakeEngine:
    def __init__(self) -> None:
        start = datetime(2026, 1, 5, 9, 15, tzinfo=IST)
        self.rows = {}
        for symbol, step in (("SPOT", 1.0), ("FUT", 1.2)):
            rows = []
            for idx in range(60):
                close = 100.0 + idx * step
                rows.append(
                    {
                        "timestamp": start + timedelta(minutes=idx),
                        "open": close - 0.4,
                        "high": close + 0.5,
                        "low": close - 0.5,
                        "close": close,
                        "volume": 1000.0 + idx * 10.0,
                    }
                )
            self.rows[symbol] = rows

    def get_history(self, symbol, count=None, *, field="close"):
        rows = list(self.rows[symbol])
        if count is not None:
            rows = rows[-count:]
        if field == "bars":
            return rows
        return [row["close"] for row in rows]

    def get_session_vwap(self, symbol):
        return 120.0

    def get_vwap(self, symbol):
        return 120.0

    def get_session_vwap_slope(self, symbol, *, lookback=3):
        del symbol, lookback
        return 0.25

    def get_ema(self, symbol, period=20):
        del period
        return self.rows[symbol][-1]["close"] - 1.0


def test_research_orb_quality_context_supplies_production_score_inputs():
    context = _research_orb_quality_context(
        _FakeEngine(),
        spot_symbol="SPOT",
        futures_symbol="FUT",
    )
    assert context["underlying_direction_bias"] == "CE"
    assert context["underlying_direction_confidence"] > 0
    assert context["context_fresh"] is True
    assert context["futures_vwap_slope"] == 0.25
    assert context["research_direction_resolution"] == "spot_futures_agree"


def test_score_calibration_reports_gross_and_net_expectancy_by_bucket():
    trades = [
        {
            "exit_time": "2026-01-05T10:00:00+05:30",
            "entry_time": "2026-01-05T09:45:00+05:30",
            "entry_price": 100.0,
            "exit_price": 102.0,
            "quantity": 1,
            "duration_minutes": 15.0,
            "gross_pnl": 2.0,
            "fees": 0.5,
            "net_pnl": 1.5,
            "raw_setup_score": 7.0,
            "entry_branch": "retest",
            "score_reasons": ["underlying_volume_confirmation"],
        },
        {
            "exit_time": "2026-01-05T11:00:00+05:30",
            "entry_time": "2026-01-05T10:45:00+05:30",
            "entry_price": 100.0,
            "exit_price": 99.0,
            "quantity": 1,
            "duration_minutes": 15.0,
            "gross_pnl": -1.0,
            "fees": 0.5,
            "net_pnl": -1.5,
            "raw_setup_score": 6.0,
            "entry_branch": "momentum",
            "score_reasons": [],
        },
    ]
    result = analyze(
        [
            {
                "slippage_bps_per_side": 10.0,
                "trades": trades,
            }
        ]
    )
    ten = result["10.0"]
    assert ten["overall"]["gross_pnl"] == 1.0
    assert ten["overall"]["net_pnl"] == 0.0
    assert ten["score_buckets"]["7.0-7.9"]["expectancy"] == 1.5
    assert ten["entry_branches"]["retest"]["gross_expectancy"] == 2.0
