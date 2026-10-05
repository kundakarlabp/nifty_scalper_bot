import datetime
import zoneinfo

from nifty_scalper_bot.backtesting import strategy_research


IST = zoneinfo.ZoneInfo("Asia/Kolkata")


class _FakeEngine:
    def __init__(self) -> None:
        start = datetime.datetime(2026, 1, 5, 9, 15, tzinfo=IST)
        self.rows = {}
        for symbol, step in (("SPOT", 1.0), ("FUT", 1.2)):
            rows = []
            for idx in range(60):
                close = 100.0 + idx * step
                rows.append(
                    {
                        "timestamp": start + datetime.timedelta(minutes=idx),
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
    context = strategy_research._research_orb_quality_context(  # noqa: SLF001
        _FakeEngine(),
        spot_symbol="SPOT",
        futures_symbol="FUT",
    )
    assert context["underlying_direction_bias"] == "CE"
    assert context["underlying_direction_confidence"] > 0
    assert context["context_fresh"] is True
    assert context["futures_vwap_slope"] == 0.25
    assert context["research_direction_resolution"] == "spot_futures_agree"

