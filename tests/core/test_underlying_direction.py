from __future__ import annotations

from nifty_scalper_bot.core.underlying_direction import (
    UnderlyingDirectionObservation,
    UnderlyingDirectionState,
    arbitrate_underlying_direction,
)


def _obs(bias: str, *, source: str, age: float, confidence: float = 0.8):
    return UnderlyingDirectionObservation(bias=bias, confidence=confidence, age_seconds=age, source=source)


def test_agreement_uses_futures_as_price_discovery_authority() -> None:
    spot = _obs("PE", source="spot_context", age=2.4, confidence=0.84)
    futures = _obs("PE", source="futures_context", age=0.2, confidence=0.91)
    resolved = arbitrate_underlying_direction(spot, futures)
    assert resolved.conflict is False
    assert resolved.observation is futures
    assert resolved.state is UnderlyingDirectionState.CONFIRMED_BEAR
    assert resolved.confirming_source == "spot_context"


def test_any_fresh_spot_futures_disagreement_is_non_executable_transition() -> None:
    for spot_conf, fut_conf in ((0.95, 0.58), (0.58, 0.95), (0.82, 0.74), (0.68, 0.45)):
        resolved = arbitrate_underlying_direction(
            _obs("CE", source="spot_context", age=0.2, confidence=spot_conf),
            _obs("PE", source="futures_context", age=0.1, confidence=fut_conf),
        )
        assert resolved.conflict is True
        assert resolved.observation is None
        assert resolved.executable_bias is None
        assert resolved.state is UnderlyingDirectionState.TRANSITION
        assert resolved.reason == "fresh_spot_futures_disagreement"


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
    assert "DIRECTION_CONTEXT_CONFLICT_FAIL_CLOSED" in source
