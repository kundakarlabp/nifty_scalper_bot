"""The optional VWAP age check uses explicit LIVE decision time, not strategy clocks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core.strategy_live_safety import (
    _approved_vwap_continuation_freshness_block,
)

NOW = datetime(2026, 10, 29, 3, 58, 51, tzinfo=timezone.utc)
EPOCH = NOW.timestamp()


def _signal(anchor, *, strategy="VWAPPro", setup="vwap_continuation", action="BUY"):
    return SimpleNamespace(
        action=action,
        metadata={
            "strategy": strategy,
            "setup_type": setup,
            "setup_candle_timestamp": anchor,
            "latest_bar_ts": NOW,
            "signal_timestamp": NOW,
        },
    )


@pytest.mark.parametrize(
    ("age", "blocked"),
    [(0, False), (179.9, False), (180, False), (180.1, True), (231, True)],
)
def test_original_vwap_thesis_age_controls_entry(monkeypatch, age, blocked):
    monkeypatch.setenv("VWAP_STRICT_CONTINUATION_ENABLED", "true")
    result = _approved_vwap_continuation_freshness_block(
        _signal(NOW - timedelta(seconds=age)), now_epoch=EPOCH
    )
    assert (result is not None) is blocked
    if blocked:
        assert result["reason"] == "live_vwap_thesis_expired"


@pytest.mark.parametrize("anchor", [None, "", "invalid", float("nan"), float("inf")])
def test_missing_vwap_anchor_fails_closed(monkeypatch, anchor):
    monkeypatch.setenv("VWAP_STRICT_CONTINUATION_ENABLED", "true")
    block = _approved_vwap_continuation_freshness_block(
        _signal(anchor), now_epoch=EPOCH
    )
    assert block["reason"] == "live_vwap_thesis_timestamp_missing"


def test_no_retroactive_live_change_when_disabled(monkeypatch):
    monkeypatch.delenv("VWAP_STRICT_CONTINUATION_ENABLED", raising=False)
    assert (
        _approved_vwap_continuation_freshness_block(
            _signal(NOW - timedelta(minutes=10)), now_epoch=EPOCH
        )
        is None
    )


def test_reclaim_and_close_excluded_from_continuation_gate(monkeypatch):
    monkeypatch.setenv("VWAP_STRICT_CONTINUATION_ENABLED", "true")
    stale = NOW - timedelta(minutes=10)
    for signal in (
        _signal(stale, setup="vwap_reclaim_momentum"),
        _signal(stale, strategy="SMC"),
        _signal(stale, action="CLOSE_LONG"),
        _signal(stale, action="SELL"),
    ):
        assert (
            _approved_vwap_continuation_freshness_block(signal, now_epoch=EPOCH) is None
        )


def test_future_and_nonfinite_decision_time_rejected(monkeypatch):
    monkeypatch.setenv("VWAP_STRICT_CONTINUATION_ENABLED", "true")
    future = _approved_vwap_continuation_freshness_block(
        _signal(NOW + timedelta(seconds=12)), now_epoch=EPOCH
    )
    assert future["reason"] == "live_vwap_thesis_future_timestamp"
    missing_clock = _approved_vwap_continuation_freshness_block(
        _signal(NOW), now_epoch=float("nan")
    )
    assert missing_clock["reason"] == "live_vwap_clock_unavailable"
