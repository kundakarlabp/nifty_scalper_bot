from datetime import datetime

from nifty_scalper_bot.metrics.kpi import KPITracker, TradeSample, summarize_trades


def _trade(direction: str, entry: float, exit_price: float, hour: int) -> TradeSample:
    return TradeSample(
        symbol="OPT",
        direction=direction,
        quantity=50,
        entry_time=datetime(2024, 1, 1, hour, 0, 0),
        exit_time=datetime(2024, 1, 1, hour, 5, 0),
        entry_price=entry,
        exit_price=exit_price,
        mae=1.2,
        mfe=2.4,
        spread_cost=0.5,
        slippage=0.25,
        decay=0.1,
    )


def test_summarize_trades_computes_kpis() -> None:
    trades = [_trade("LONG", 100.0, 101.0, 9), _trade("SHORT", 120.0, 119.0, 10)]
    report = summarize_trades(trades)

    assert report.total_trades == 2
    assert report.hit_rate == 1.0
    assert report.total_spread_cost == 1.0
    assert 9 in report.per_hour_returns
    assert 10 in report.per_hour_returns


def test_kpi_tracker_formatting() -> None:
    tracker = KPITracker()
    tracker.record(_trade("LONG", 100.0, 101.0, 9))
    text = tracker.format_report()
    assert "Trades: 1" in text
    assert "Hit-rate" in text
