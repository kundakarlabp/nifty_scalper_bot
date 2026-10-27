"""Live SMC freshness: the sweep is older than a newly emitted quote/signal."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core.strategy_live_safety import (
    _approved_smc_freshness_block,
)

NOW = datetime(2026, 10, 26, 8, 39, 27, tzinfo=timezone.utc)
EPOCH = NOW.timestamp()


def _signal(*, strategy="SMC", action="BUY", sweep=None, **metadata):
    return SimpleNamespace(
        action=action,
        metadata={
            "strategy": strategy,
            "setup_id": "smcv2:NFO:NIFTY26OCTFUT:CE:2026-10-26T08:33:00+00:00",
            "sweep_timestamp": sweep,
            "latest_bar_ts": NOW,
            "signal_timestamp": NOW,
            **metadata,
        },
    )


@pytest.mark.parametrize(
    ("seconds_old", "blocked"),
    [
        (0, False),
        (60, False),
        (299.9, False),
        (300.0, False),
        (300.1, True),
        (389.0, True),  # October 26 RANGE loss: 14:03 sweep, 14:09 entry
    ],
)
def test_old_sweep_cannot_be_made_fresh_by_new_signal_timestamp(
    seconds_old: float, blocked: bool
) -> None:
    signal = _signal(sweep=NOW - timedelta(seconds=seconds_old))
    result = _approved_smc_freshness_block(signal, now_epoch=EPOCH)
    assert (result is not None) is blocked
    if blocked:
        assert result["reason"] == "live_smc_sweep_expired"


@pytest.mark.parametrize(
    "sweep",
    [
        None,
        "",
        "invalid",
        float("nan"),
        float("inf"),
    ],
)
def test_missing_or_invalid_sweep_proof_fails_closed(sweep) -> None:
    result = _approved_smc_freshness_block(_signal(sweep=sweep), now_epoch=EPOCH)
    assert result is not None
    assert result["reason"] == "live_smc_sweep_timestamp_missing"


def test_epoch_ms_and_iso_string_are_accepted() -> None:
    when = NOW - timedelta(seconds=25)
    for sweep in (when.timestamp(), when.timestamp() * 1000, when.isoformat()):
        assert (
            _approved_smc_freshness_block(_signal(sweep=sweep), now_epoch=EPOCH) is None
        )


def test_future_sweep_timestamp_fails_closed() -> None:
    result = _approved_smc_freshness_block(
        _signal(sweep=NOW + timedelta(seconds=15)), now_epoch=EPOCH
    )
    assert result is not None
    assert result["reason"] == "live_smc_sweep_future_timestamp"


def test_nonfinite_wall_clock_fails_closed() -> None:
    result = _approved_smc_freshness_block(_signal(sweep=NOW), now_epoch=float("nan"))
    assert result is not None
    assert result["reason"] == "live_smc_clock_unavailable"


def test_explicit_shorter_window_respected(monkeypatch) -> None:
    monkeypatch.setenv("SMC_CONFIRMATION_MAX_MINUTES", "2")
    result = _approved_smc_freshness_block(
        _signal(sweep=NOW - timedelta(seconds=121)), now_epoch=EPOCH
    )
    assert result is not None
    assert result["reason"] == "live_smc_sweep_expired"
    assert result["max_sweep_age_seconds"] == 120


def test_operator_override_cannot_extend_window_beyond_five_minutes(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SMC_CONFIRMATION_MAX_MINUTES", "120")
    result = _approved_smc_freshness_block(
        _signal(sweep=NOW - timedelta(minutes=6)), now_epoch=EPOCH
    )
    assert result is not None
    assert result["max_sweep_age_seconds"] == 300


@pytest.mark.parametrize("value", ["nan", "inf", "not-a-number"])
def test_invalid_window_does_not_bypass_protection(monkeypatch, value) -> None:
    monkeypatch.setenv("SMC_CONFIRMATION_MAX_MINUTES", value)
    assert (
        _approved_smc_freshness_block(
            _signal(sweep=NOW - timedelta(minutes=6)), now_epoch=EPOCH
        )["reason"]
        == "live_smc_sweep_expired"
    )


def test_close_signals_and_other_strategies_unchanged() -> None:
    stale = NOW - timedelta(hours=1)
    assert (
        _approved_smc_freshness_block(
            _signal(action="CLOSE_LONG", sweep=stale), now_epoch=EPOCH
        )
        is None
    )
    assert (
        _approved_smc_freshness_block(
            _signal(strategy="VWAPPro", sweep=stale), now_epoch=EPOCH
        )
        is None
    )
