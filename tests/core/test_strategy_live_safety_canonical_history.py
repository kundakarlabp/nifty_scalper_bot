from __future__ import annotations

import time
from types import SimpleNamespace

from nifty_scalper_bot.core import strategy_live_safety as guard


class Engine:
    def __init__(self, bars):
        self.bars = bars

    def get_history(self, _):
        return [{} for _ in range(self.bars)]


class MDM:
    def __init__(self, rows):
        self.rows = rows

    def get_ohlc_bars(self, symbol, *, limit=None):
        return self.rows[-limit:] if limit else list(self.rows)

    def get_latest_closed_bar(self, symbol, interval="1min"):
        return self.rows[-1] if self.rows else None


def manager(mdm_rows, indicator_bars):
    return SimpleNamespace(
        _market_data_manager=MDM(mdm_rows),
        _indicator_engine=Engine(indicator_bars),
        _required_candles=2,
        _bar_interval_seconds=60,
        _latest_context_snapshots={
            "spot_context": {"timestamp": time.time() - 1},
            "futures_context": {"timestamp": time.time() - 1},
        },
        _is_live_mode=lambda: True,
    )


def bars(count, ts=None):
    ts = ts or ((time.time() // 60) * 60 - 60)
    return [
        {"timestamp": ts, "open": 1, "high": 1, "low": 1, "close": 1}
        for _ in range(count)
    ]


def test_mdm_history_is_authoritative_when_indicator_empty():
    assert guard._evaluation_readiness_block(manager(bars(3), 0), "NSE:NIFTY") is None


def test_indicator_history_does_not_override_empty_mdm():
    block = guard._evaluation_readiness_block(manager([], 5), "NSE:NIFTY")
    assert block["reason"] == "live_underlying_history_not_ready"
    assert block["mdm_bars"] == 0
    assert block["indicator_bars"] == 5


def test_current_forming_bar_is_not_closed():
    now = time.time()
    block = guard._evaluation_readiness_block(
        manager(bars(3, ts=(now // 60) * 60), 0), "NSE:NIFTY"
    )
    assert block["reason"] == "live_latest_closed_bar_open"
