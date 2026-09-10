"""Regime vocabulary contract.

Producers and consumers of the regime label previously used four different
vocabularies, so gates and weights keyed on words nothing emitted resolved to
their neutral default. These tests pin the single canonical vocabulary and the
normalisation seam that every consumer must go through.
"""

from __future__ import annotations

import pytest

from nifty_scalper_bot.config.regime_ontology import (
    MarketRegime,
    normalize_regime,
    regime_label,
)
from nifty_scalper_bot.core.market_regime import MarketRegimeDetector
from nifty_scalper_bot.core.strategy_manager import REGIME_STRATEGY_WEIGHTS
from nifty_scalper_bot.risk.regime_sizing import RegimeType
from nifty_scalper_bot.strategies.market_regime_engine import MarketRegimeEngine


CANONICAL = {member.value for member in MarketRegime}


def test_regime_weight_table_is_keyed_by_canonical_names_only() -> None:
    """A weight row keyed on a word no producer emits can never be applied."""
    unknown_keys = set(REGIME_STRATEGY_WEIGHTS) - CANONICAL
    assert unknown_keys == set(), f"non-canonical weight rows: {sorted(unknown_keys)}"


def test_runner_regime_engine_only_emits_canonical_regimes() -> None:
    engine = MarketRegimeEngine()
    indicator_sets = [
        {"adx": 30.0, "atr": 12.0, "atr_average": 10.0, "vwap_slope": 0.5, "volume_expansion": 1.2},
        {"adx": 20.0, "atr": 30.0, "atr_average": 10.0, "vwap_slope": 0.0, "volume_expansion": 1.0},
        {"adx": 10.0, "atr": 9.0, "atr_average": 10.0, "vwap_slope": 0.0, "volume_expansion": 1.0},
        {"adx": 20.0, "atr": 9.0, "atr_average": 10.0, "vwap_slope": 0.0, "volume_expansion": 0.2},
        {"adx": None, "atr": 0.0, "atr_average": 0.0, "vwap_slope": 0.0, "volume_expansion": 0.0},
    ]
    for indicators in indicator_sets:
        snapshot = engine.classify(indicators)
        assert snapshot.regime.value in CANONICAL


def test_core_detector_regimes_normalise_to_canonical_names() -> None:
    """The core detector emits lowercase words; they must resolve, not fall through."""
    for emitted in ("trend", "range", "volatile", "event"):
        assert normalize_regime(emitted) is not MarketRegime.UNKNOWN
        assert normalize_regime(emitted).value in CANONICAL


def test_detector_classify_outputs_are_canonical() -> None:
    detector = MarketRegimeDetector()
    features = {
        "adx": 32.0,
        "atr": 14.0,
        "price_momentum": 0.02,
        "ema_fast": 102.0,
        "ema_slow": 100.0,
        "close": 100.0,
        "price": 100.0,
        "volume_ratio": 1.1,
    }
    snapshot = detector.evaluate("NSE:NIFTY", features)
    assert snapshot is not None
    assert normalize_regime(snapshot.regime) is not MarketRegime.UNKNOWN


def test_risk_regime_sizing_labels_are_canonical() -> None:
    for member in RegimeType:
        assert normalize_regime(member.value).value in CANONICAL


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("TREND_UP", MarketRegime.TREND),
        ("TREND_DOWN", MarketRegime.TREND),
        ("trending", MarketRegime.TREND),
        ("CHOPPY", MarketRegime.RANGE),
        ("NORMAL", MarketRegime.RANGE),
        ("ranging", MarketRegime.RANGE),
        ("HIGH_VOLATILITY", MarketRegime.VOLATILE),
        ("HIGHVOL", MarketRegime.VOLATILE),
        ("LOW_VOLATILITY", MarketRegime.LOW_ACTIVITY),
        ("event", MarketRegime.EVENT),
    ],
)
def test_legacy_labels_normalise_to_canonical(raw: str, expected: MarketRegime) -> None:
    assert normalize_regime(raw) is expected


@pytest.mark.parametrize("raw", [None, "", "   ", "gibberish", 17, object()])
def test_unresolvable_labels_fail_closed_as_unknown(raw: object) -> None:
    """An unrecognised label must not score as a tradable regime."""
    assert normalize_regime(raw) is MarketRegime.UNKNOWN


def test_enum_and_snapshot_inputs_are_accepted() -> None:
    class _Snapshot:
        regime = "volatile"

    assert normalize_regime(MarketRegime.EVENT) is MarketRegime.EVENT
    assert normalize_regime(_Snapshot()) is MarketRegime.VOLATILE
    assert regime_label("trend_up") == "TREND"


def test_trend_regime_now_receives_its_configured_weight() -> None:
    """The defect: detector said TREND, the table said TREND_UP, SMC got 1.0."""
    trend_row = REGIME_STRATEGY_WEIGHTS[MarketRegime.TREND.value]
    assert trend_row["SMC"] > 1.0
    assert REGIME_STRATEGY_WEIGHTS[normalize_regime("trend").value]["SMC"] == trend_row["SMC"]


def test_defensive_regimes_damp_directional_triggers() -> None:
    for regime in (MarketRegime.VOLATILE, MarketRegime.EVENT, MarketRegime.LOW_ACTIVITY):
        row = REGIME_STRATEGY_WEIGHTS[regime.value]
        assert row["SMC"] < 1.0, f"{regime.value} must not score SMC at full weight"
