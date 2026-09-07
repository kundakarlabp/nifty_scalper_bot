from __future__ import annotations

from nifty_scalper_bot.core.underlying_direction import (
    UnderlyingDirectionObservation,
    arbitrate_underlying_direction,
)


def _obs(bias: str, *, source: str, age: float, confidence: float = 0.8):
    return UnderlyingDirectionObservation(
        bias=bias,
        confidence=confidence,
        age_seconds=age,
        source=source,
    )


def test_agreement_uses_futures_as_price_discovery_authority() -> None:
    spot = _obs("PE", source="spot_context", age=2.4, confidence=0.84)
    futures = _obs("PE", source="futures_context", age=0.2, confidence=0.91)

    resolved = arbitrate_underlying_direction(spot, futures)

    assert resolved.conflict is False
    assert resolved.observation is futures
    assert resolved.observation.age_seconds == 0.2
    assert resolved.observation.confidence == 0.91
    assert resolved.confirming_source == "spot_context"


def test_comparable_disagreement_fails_closed_even_when_futures_is_primary() -> None:
    resolved = arbitrate_underlying_direction(
        _obs("CE", source="spot_context", age=0.4, confidence=0.82),
        _obs("PE", source="futures_context", age=0.1, confidence=0.74),
    )

    assert resolved.conflict is True
    assert resolved.observation is None


def test_materially_stronger_spot_can_override_weak_futures() -> None:
    spot = _obs("PE", source="spot_context", age=0.2, confidence=0.90)
    resolved = arbitrate_underlying_direction(
        spot,
        _obs("CE", source="futures_context", age=0.1, confidence=0.58),
    )
    assert resolved.conflict is False
    assert resolved.observation is spot
    assert resolved.confirming_source == "futures_context:weak_disagreement"


def test_materially_stronger_futures_can_override_weak_spot() -> None:
    futures = _obs("CE", source="futures_context", age=0.1, confidence=0.91)
    resolved = arbitrate_underlying_direction(
        _obs("PE", source="spot_context", age=0.2, confidence=0.60),
        futures,
    )
    assert resolved.conflict is False
    assert resolved.observation is futures
    assert resolved.confirming_source == "spot_context:weak_disagreement"


def test_low_conviction_disagreement_still_fails_closed() -> None:
    resolved = arbitrate_underlying_direction(
        _obs("PE", source="spot_context", age=0.2, confidence=0.68),
        _obs("CE", source="futures_context", age=0.1, confidence=0.45),
    )
    assert resolved.conflict is True
    assert resolved.observation is None


def test_single_resolved_source_is_accepted_without_cross_source_age() -> None:
    futures = _obs("PE", source="futures_context", age=1.7, confidence=0.76)

    resolved = arbitrate_underlying_direction(None, futures)

    assert resolved.conflict is False
    assert resolved.observation is futures
    assert resolved.observation.age_seconds == 1.7


def test_direction_observation_rejects_non_directional_values() -> None:
    try:
        _obs("NEUTRAL", source="spot_context", age=0.1)
    except ValueError as exc:
        assert "invalid underlying direction" in str(exc)
    else:
        raise AssertionError("non-directional observation must fail closed")


def test_strategy_manager_documents_and_uses_underlying_only_authority() -> None:
    from pathlib import Path

    source = Path("src/nifty_scalper_bot/core/strategy_manager.py").read_text()
    assert "OPTION PREMIUM DATA MUST NEVER AUTHORIZE UNDERLYING DIRECTION" in source
    assert "arbitrate_underlying_direction" in source
    assert 'direction_bias = (indicators.get("direction_bias")' not in source
    assert 'DIRECTION_CONTEXT_CONFLICT_FAIL_CLOSED' in source
