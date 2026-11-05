"""Regression for trusted option volume from WebSocket to completed indicators."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core.option_volume_diagnostics import (
    describe_option_volume_evidence,
)
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.strategies.runner import StrategyRunner

SYMBOL = "NFO:NIFTY26N0323650CE"


class _BuilderReached(BaseException):
    """End the runner tick after observing its incoming candle volume."""


def _tick_runner() -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = logging.getLogger("test_runner_option_volume_parity")
    runner._position_manager = None
    runner._risk_manager = SimpleNamespace(available_balance=100_000.0)
    runner._history_ready_by_symbol = {}
    runner._last_cumulative_volume = {}
    runner._last_valid_price = {}
    runner._last_valid_price_ts = {}
    runner._quote_update_versions = {}
    runner._first_tick_logged_symbols = {SYMBOL}
    runner._tick_log_throttle_seconds = 60.0
    runner._should_log_throttled = lambda *_args, **_kwargs: False
    runner._tick_has_quote_update = lambda _tick: False
    runner._refresh_risk_halt_state = lambda _symbol: False
    runner._is_market_open = lambda _now: True
    runner._startup_timestamp = time.time() - 60
    runner._bar_builders = {}
    return runner


@pytest.mark.parametrize(
    ("volume_untrusted", "expected_volume"),
    [(False, 1_999_000), (True, 0)],
)
def test_runner_preserves_trusted_large_delta_and_quarantines_untrusted(
    monkeypatch, volume_untrusted: bool, expected_volume: int
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("OPTION_MAX_REASONABLE_TICK_VOLUME_DELTA", "1")
    mdm = MarketDataManager(kite=None)
    now = datetime.now(timezone.utc)
    base = {"instrument_token": 123, "timestamp": now - timedelta(seconds=1)}
    mdm._normalise_tick_volume_delta(SYMBOL, {**base, "volume_traded_today": 1_000})
    normalized = mdm._normalise_tick_volume_delta(
        SYMBOL,
        {"instrument_token": 123, "timestamp": now, "volume_traded_today": 2_000_000},
    )
    assert normalized["volume_transition"]["state"] == "accepted"
    assert normalized["volume_delta"] == 1_999_000

    observed: list[int] = []

    def capture_bar(_builder, _symbol, _price, volume, _timestamp):
        observed.append(volume)
        raise _BuilderReached()

    monkeypatch.setattr(
        "nifty_scalper_bot.strategies.runner._update_runner_live_candle",
        capture_bar,
    )
    runner = _tick_runner()
    with pytest.raises(_BuilderReached):
        runner._on_tick(
            SYMBOL,
            {
                "ltp": 111.0,
                "timestamp": now,
                "source": "ws",
                "_protection_already_handled": True,
                "volume_delta": normalized["volume_delta"],
                "volume_delta_untrusted": volume_untrusted,
            },
        )
    assert observed == [expected_volume]


@pytest.mark.parametrize("volume", [6_989_060, 7_854_730])
def test_reseed_same_completed_canonical_minute_retains_option_volume(
    volume: int,
) -> None:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = logging.getLogger("test_option_canonical_reseed")
    runner._normalize_symbol = lambda symbol: symbol
    runner._lock = threading.RLock()
    runner._symbol_history = {}
    runner._last_bar_ts = {}
    runner._data_phase = {}
    runner._indicator_engine = IndicatorEngine()
    runner._set_symbol_hydration_state = lambda *_args: None
    runner._seed_pipeline_store = lambda *_args: None

    # A projection reseed must not feed derived bars back to MDM's canonical
    # history (or advance its queue-watermark on every parity repair).
    def reject_reverse_history_import(*_args):
        raise AssertionError("canonical history must not be reimported")

    runner._seed_candle_engine_from_history = reject_reverse_history_import
    runner._maybe_promote_pending_active_basket = lambda **_kwargs: None

    current = datetime(2026, 11, 3, 7, 2, tzinfo=timezone.utc)
    previous = {
        "timestamp": current - timedelta(minutes=1),
        "open": 110.0,
        "high": 112.0,
        "low": 109.0,
        "close": 111.0,
        "volume": 200,
    }
    canonical_completed = {
        **previous,
        "timestamp": current,
        "volume": volume,
    }
    assert (
        runner.reseed_history_from_bars(
            SYMBOL, [previous, canonical_completed], source="mdm_reseed", min_bars=2
        )
        == 2
    )
    indicator_completed = runner._indicator_engine.get_history(
        SYMBOL, count=1, field="bars"
    )[0]
    assert runner._symbol_history[SYMBOL][-1].volume == volume
    assert indicator_completed["volume"] == volume
    evidence = describe_option_volume_evidence(
        {"volume": float(volume)},
        latest_bar=indicator_completed,
        indicator_completed_bar=indicator_completed,
        canonical_completed_bar=canonical_completed,
    )
    assert evidence["completed_bar_parity"] == "same_minute_volume_match"


def test_live_completed_projection_uses_canonical_volume_not_coalesced_tick() -> None:
    """Skipped entry-evaluation updates must not silently remove traded volume."""
    runner = StrategyRunner.__new__(StrategyRunner)
    initial = datetime(2026, 11, 5, 4, 0, tzinfo=timezone.utc)
    canonical = [
        {
            "timestamp": initial + timedelta(minutes=i),
            "open": 150.0 + i,
            "high": 151.0 + i,
            "low": 149.0 + i,
            "close": 150.5 + i,
            "volume": total,
        }
        for i, total in enumerate((1_480_000.0, 936_650.0))
    ]
    runner._candle_engines = {
        SYMBOL: SimpleNamespace(get_completed_tail=lambda _limit: canonical)
    }
    runner._last_bar_ts = {}
    runner._indicator_engine = SimpleNamespace(get_history=lambda *_a, **_k: [])
    accepted = []

    def ingest(_symbol, bar):
        accepted.append(bar)
        runner._last_bar_ts[_symbol] = bar.timestamp

    runner._ingest_bar = ingest

    # Only one coalesced worker invocation; MDM has both complete bars.
    recent = runner._ingest_live_canonical_completed_bars(SYMBOL)
    assert [bar.volume for bar in accepted] == [1_480_000, 936_650]
    assert [bar.timestamp for bar in accepted] == [
        initial,
        initial + timedelta(minutes=1),
    ]
    assert recent is accepted[-1]
    assert runner._ingest_live_canonical_completed_bars(SYMBOL) is None
    assert len(accepted) == 2


def test_live_completed_projection_skips_hydrated_duplicate_and_bad_bar() -> None:
    runner = StrategyRunner.__new__(StrategyRunner)
    minute = datetime(2026, 11, 5, 4, 0, tzinfo=timezone.utc)
    valid = {
        "timestamp": minute,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1200.0,
    }
    malformed = {**valid, "timestamp": minute + timedelta(minutes=1), "volume": -1}
    subsequent = {**valid, "timestamp": minute + timedelta(minutes=2), "volume": 925.0}
    runner._candle_engines = {
        SYMBOL: SimpleNamespace(
            get_completed_tail=lambda _limit: [valid, malformed, subsequent]
        )
    }
    runner._last_bar_ts = {}
    runner._indicator_engine = SimpleNamespace(
        get_history=lambda *_a, **_k: [{"timestamp": minute}]
    )
    accepted = []

    def ingest(_symbol, bar):
        accepted.append(bar)
        runner._last_bar_ts[_symbol] = bar.timestamp

    runner._ingest_bar = ingest
    recent = runner._ingest_live_canonical_completed_bars(SYMBOL)
    assert [bar.timestamp for bar in accepted] == [minute + timedelta(minutes=2)]
    assert [bar.volume for bar in accepted] == [925]
    assert recent is accepted[-1]


def test_live_completed_projection_does_not_fallback_to_sampled_tick() -> None:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._candle_engines = {}
    runner._last_bar_ts = {}
    runner._indicator_engine = SimpleNamespace(get_history=lambda *_a, **_k: [])
    assert runner._ingest_live_canonical_completed_bars(SYMBOL) is None
