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


def test_spot_and_futures_agreement_preserves_spot_atomic_provenance() -> None:
    spot = _obs("PE", source="spot_context", age=2.4, confidence=0.84)
    futures = _obs("PE", source="futures_context", age=0.2, confidence=0.91)

    resolved = arbitrate_underlying_direction(spot, futures)

    assert resolved.conflict is False
    assert resolved.observation is spot
    assert resolved.observation.age_seconds == 2.4
    assert resolved.observation.confidence == 0.84
    assert resolved.confirming_source == "futures_context"


def test_fresh_spot_futures_direction_disagreement_fails_closed() -> None:
    resolved = arbitrate_underlying_direction(
        _obs("CE", source="spot_context", age=0.4),
        _obs("PE", source="futures_context", age=0.1),
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
    # Regression guard: option-local direction must not lead the underlying
    # resolver as it did before this fix.
    assert 'direction_bias = (indicators.get("direction_bias")' not in source
    assert 'DIRECTION_CONTEXT_CONFLICT_FAIL_CLOSED' in source
