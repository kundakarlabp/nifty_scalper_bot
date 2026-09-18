import importlib

from nifty_scalper_bot.core.strategy_manager import (
    StrategyManager,
    _enrich_option_quote_context,
)

STRATEGY_MANAGER_MODULE = importlib.import_module(
    "nifty_scalper_bot.core.strategy_manager"
)


def _direction_manager() -> StrategyManager:
    manager = object.__new__(StrategyManager)
    manager._latest_context_snapshots = {}
    return manager


def _direction_inputs(side: str | None) -> dict[str, float]:
    if side == "PE":
        return {
            "close": 99.0,
            "vwap": 100.0,
            "ema_fast": 99.0,
            "ema_slow": 100.0,
            "ema_50": 100.0,
            "vwap_slope": -0.1,
            "ema_slope": -0.1,
        }
    if side == "CE":
        return {
            "close": 101.0,
            "vwap": 100.0,
            "ema_fast": 101.0,
            "ema_slow": 100.0,
            "ema_50": 100.0,
            "vwap_slope": 0.1,
            "ema_slope": 0.1,
        }
    return {
        "close": 100.0,
        "vwap": 100.0,
        "ema_fast": 100.0,
        "ema_slow": 100.0,
        "ema_50": 100.0,
        "vwap_slope": 0.0,
        "ema_slope": 0.0,
    }


def test_strategy_manager_context_age_uses_canonical_quote_age_schema() -> None:
    assert StrategyManager._context_tick_age_seconds({"quote_age_s": 0.25}) == 0.25
    assert StrategyManager._context_tick_age_seconds({"tick_age_ms": 250}) == 0.25
    assert StrategyManager._context_tick_age_seconds({"tick_age_ms": "invalid"}) is None


def test_option_quote_enrichment_preserves_canonical_freshness_provenance() -> None:
    indicators = {
        "tick_age_ms": 9_999.0,
        "quote_age_s": 9.999,
        "quote_update_version": 1,
        "depth": {"buy": [{"price": 99.0}], "sell": [{"price": 101.0}]},
    }
    quote = {
        "timestamp_quality": "exchange",
        "last_tick_ts_ms": 1_800_000_000_000.0,
        "tick_age_ms": 125.0,
        "quote_age_ms": 125.0,
        "quote_age_s": 0.125,
        "quote_update_version": 42,
        "depth": {"buy": [{"price": 100.0}], "sell": [{"price": 100.2}]},
        "depth_available": True,
        "tradable_quote": True,
    }

    enriched = _enrich_option_quote_context(indicators, quote)

    assert enriched["tick_age_ms"] == 125.0
    assert enriched["quote_age_s"] == 0.125
    assert enriched["quote_update_version"] == 42
    assert enriched["timestamp_quality"] == "exchange"
    assert enriched["depth"] == quote["depth"]
    assert enriched["tradable_quote"] is True


def test_futures_context_neutral_values_do_not_create_direction() -> None:
    from nifty_scalper_bot.core.strategy_manager import StrategyManager

    manager = object.__new__(StrategyManager)
    manager._latest_context_snapshots = {}
    manager._update_context_snapshot(
        symbol="NFO:NIFTY26JULFUT",
        indicators={
            "close": 25000.0,
            "vwap": 25000.0,
            "ema_fast": 25000.0,
            "ema_slow": 25000.0,
            "ema_50": 25000.0,
            "vwap_slope": 0.0,
            "ema_slope": 0.0,
        },
        role="futures_context",
    )
    snapshot = manager._latest_context_snapshots["futures_context"]
    assert snapshot["vwap_slope"] == 0.0
    assert snapshot["direction_bias"] is None


def test_futures_context_uses_same_evaluation_slope_only() -> None:
    from nifty_scalper_bot.core.strategy_manager import StrategyManager

    manager = object.__new__(StrategyManager)
    manager._latest_context_snapshots = {"futures_context": {"vwap": 100.0}}
    manager._update_context_snapshot(
        symbol="NFO:NIFTY26JULFUT",
        indicators={"close": 101.0, "vwap": 101.0, "vwap_slope": 0.001},
        role="futures_context",
    )
    snapshot = manager._latest_context_snapshots["futures_context"]
    assert snapshot["vwap_slope"] == 0.001
    assert snapshot["direction_bias"] == "CE"
    manager._update_context_snapshot(
        symbol="NFO:NIFTY26JULFUT",
        indicators={"close": 102.0, "vwap": 102.0},
        role="futures_context",
    )
    assert manager._latest_context_snapshots["futures_context"]["vwap_slope"] is None


def test_futures_context_uses_hydrated_history_when_ema_aliases_are_absent() -> None:
    from datetime import datetime, timedelta, timezone

    from nifty_scalper_bot.strategies.indicators import IndicatorEngine

    symbol = "NFO:NIFTY26AUGFUT"
    engine = IndicatorEngine()
    started_at = datetime(2026, 8, 5, 3, 45, tzinfo=timezone.utc)
    for index in range(60):
        price = 25000.0 + float(index)
        engine.update_price(
            symbol,
            {"open": price, "high": price, "low": price, "close": price},
            volume=0,
            timestamp=started_at + timedelta(minutes=index),
        )

    manager = object.__new__(StrategyManager)
    manager._indicator_engine = engine
    manager._latest_context_snapshots = {}
    manager._update_context_snapshot(
        symbol=symbol,
        indicators={"ltp": 25059.0, "close": 25059.0},
        role="futures_context",
    )

    snapshot = manager._latest_context_snapshots["futures_context"]
    assert snapshot["vwap"] is None
    assert snapshot["ema_fast"] > snapshot["ema_slow"] > snapshot["ema_50"]
    assert snapshot["ema_fast_source"] == "indicator_engine_history"
    assert snapshot["ema_slow_source"] == "indicator_engine_history"
    assert snapshot["ema_50_source"] == "indicator_engine_history"
    assert snapshot["direction_bias"] == "CE"
    assert "ema_fast_above_slow" in snapshot["direction_context_reasons"]


def test_futures_context_derives_current_vwap_slope_from_hydrated_history() -> None:
    from datetime import datetime, timedelta, timezone

    from nifty_scalper_bot.strategies.indicators import IndicatorEngine

    symbol = "NFO:NIFTY26AUGFUT"
    engine = IndicatorEngine()
    started_at = datetime(2026, 8, 5, 3, 45, tzinfo=timezone.utc)
    for index in range(10):
        price = 25000.0 + float(index)
        engine.update_price(
            symbol,
            {"open": price, "high": price, "low": price, "close": price},
            volume=100,
            timestamp=started_at + timedelta(minutes=index),
        )

    manager = object.__new__(StrategyManager)
    manager._indicator_engine = engine
    manager._latest_context_snapshots = {}
    manager._update_context_snapshot(
        symbol=symbol,
        indicators={"ltp": 25009.0, "close": 25009.0},
        role="futures_context",
    )

    snapshot = manager._latest_context_snapshots["futures_context"]
    assert snapshot["vwap_slope"] > 0
    assert snapshot["vwap_slope_source"] == "indicator_engine_history"
    assert "vwap_slope_positive" in snapshot["direction_context_reasons"]


def test_transient_tick_reversal_cannot_flip_authoritative_source_direction(
    monkeypatch,
) -> None:
    now = [1_000.0]
    monkeypatch.setattr(STRATEGY_MANAGER_MODULE.time, "time", lambda: now[0])
    manager = _direction_manager()
    manager._update_context_snapshot(
        symbol="NFO:NIFTY26SEPFUT",
        indicators={**_direction_inputs("PE"), "recent_ltp_delta": -1.0},
        role="futures_context",
    )

    now[0] += 1.0
    manager._update_context_snapshot(
        symbol="NFO:NIFTY26SEPFUT",
        indicators={**_direction_inputs(None), "recent_ltp_delta": 1.0},
        role="futures_context",
    )

    snapshot = manager._latest_context_snapshots["futures_context"]
    assert snapshot["direction_bias"] == "PE"
    assert "direction_tie_hysteresis" in snapshot["direction_context_reasons"]


def test_tick_movement_cannot_manufacture_source_direction() -> None:
    manager = _direction_manager()
    manager._update_context_snapshot(
        symbol="NSE:NIFTY",
        indicators={**_direction_inputs(None), "recent_ltp_delta": 1.0},
        role="spot_context",
    )

    snapshot = manager._latest_context_snapshots["spot_context"]
    assert snapshot["direction_bias"] is None
    assert "tick_slope_positive" in snapshot["direction_context_reasons"]


def test_source_direction_switch_requires_persistent_contrary_structure(
    monkeypatch,
) -> None:
    now = [2_000.0]
    monkeypatch.setenv("STRATEGY_CONTEXT_REVERSAL_CONFIRM_SECONDS", "5")
    monkeypatch.setenv("STRATEGY_CONTEXT_REVERSAL_MIN_OBSERVATIONS", "3")
    monkeypatch.setattr(STRATEGY_MANAGER_MODULE.time, "time", lambda: now[0])
    manager = _direction_manager()
    manager._update_context_snapshot(
        symbol="NFO:NIFTY26SEPFUT",
        indicators=_direction_inputs("PE"),
        role="futures_context",
    )

    for timestamp in (2_001.0, 2_003.0):
        now[0] = timestamp
        manager._update_context_snapshot(
            symbol="NFO:NIFTY26SEPFUT",
            indicators=_direction_inputs("CE"),
            role="futures_context",
        )
        assert (
            manager._latest_context_snapshots["futures_context"]["direction_bias"]
            == "PE"
        )

    now[0] = 2_006.0
    manager._update_context_snapshot(
        symbol="NFO:NIFTY26SEPFUT",
        indicators=_direction_inputs("CE"),
        role="futures_context",
    )
    snapshot = manager._latest_context_snapshots["futures_context"]
    assert snapshot["direction_bias"] == "CE"
    assert "direction_reversal_confirmed" in snapshot["direction_context_reasons"]


def test_temporary_direction_tie_expires_fail_closed(monkeypatch) -> None:
    now = [3_000.0]
    monkeypatch.setenv("STRATEGY_CONTEXT_TIE_GRACE_SECONDS", "5")
    monkeypatch.setattr(STRATEGY_MANAGER_MODULE.time, "time", lambda: now[0])
    manager = _direction_manager()
    manager._update_context_snapshot(
        symbol="NSE:NIFTY",
        indicators=_direction_inputs("PE"),
        role="spot_context",
    )

    now[0] = 3_002.0
    manager._update_context_snapshot(
        symbol="NSE:NIFTY",
        indicators=_direction_inputs(None),
        role="spot_context",
    )
    assert manager._latest_context_snapshots["spot_context"]["direction_bias"] == "PE"

    now[0] = 3_006.0
    manager._update_context_snapshot(
        symbol="NSE:NIFTY",
        indicators=_direction_inputs(None),
        role="spot_context",
    )
    assert manager._latest_context_snapshots["spot_context"]["direction_bias"] is None


def test_entry_acceptance_forwards_exact_setup_identity_to_strategy() -> None:
    observed: list[tuple[str, str | None]] = []

    class _Strategy:
        name = "SMC"

        def notify_entry_accepted(
            self, side: str, *, setup_id: str | None = None
        ) -> None:
            observed.append((side, setup_id))

    manager = object.__new__(StrategyManager)
    manager._strategies = [_Strategy()]

    manager.notify_entry_accepted("SMC", "PE", setup_id="smcv2:nifty:PE:1")

    assert observed == [("PE", "smcv2:nifty:PE:1")]
