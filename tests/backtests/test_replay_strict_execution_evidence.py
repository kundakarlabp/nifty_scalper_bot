"""Reject execution-grade claims from OHLCV-only replay at the ingress boundary."""
from datetime import datetime, timezone

from nifty_scalper_bot.backtesting.replay import ReplayHarness, ReplayResult


class QuoteHub:
    def __init__(self):
        self.latest = None

    def store_quote(self, symbol, payload, *, source):
        self.latest = (symbol, dict(payload), source)


class Paper:
    def __init__(self):
        self.hub = QuoteHub()
        self.events = []

    def process_quote(self, symbol):
        self.events.append(symbol)


class Runner:
    def __init__(self):
        self.ticks = []

    def on_tick_event(self, payload):
        self.ticks.append(payload)


def test_ohlcv_replay_strict_flag_marks_quotes_untradable():
    harness = object.__new__(ReplayHarness)
    harness._paper = Paper()
    harness._runner = Runner()
    harness._strict_execution_evidence = True
    stamp = datetime(2026, 10, 10, 9, 30, tzinfo=timezone.utc)
    harness._publish_and_dispatch(
        "NFO:NIFTY10CE",
        {"timestamp": stamp, "close": 100.0, "bid": 99.9, "ask": 100.1},
    )
    _, quote, _ = harness._paper.hub.latest
    assert quote["ltp"] == 100.0
    assert quote["tradable_quote"] is False
    assert quote["replay_reject_reason"] == "historical_ohlcv_not_executable"
    assert harness._paper.events == ["NFO:NIFTY10CE"]
    assert harness._runner.ticks[0]["tradable_quote"] is False


def test_legacy_bar_proxy_remains_labeled_not_execution_ready():
    harness = object.__new__(ReplayHarness)
    harness._paper = Paper()
    harness._runner = Runner()
    harness._strict_execution_evidence = False
    stamp = datetime(2026, 10, 10, 9, 30, tzinfo=timezone.utc)
    harness._publish_and_dispatch("NFO:NIFTY10CE", {"timestamp": stamp, "close": 100.0})
    _, quote, _ = harness._paper.hub.latest
    assert "replay_reject_reason" not in quote
    result = ReplayResult(1, stamp, stamp, [], [])
    assert result.execution_evidence_tier == "BAR_PROXY_ONLY"
    assert not result.execution_grade_ready
    assert not result.live_promotion_eligible
