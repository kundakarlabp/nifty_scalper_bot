from __future__ import annotations

from nifty_scalper_bot.core.direction_stability import DirectionStabilityGate


def test_first_direction_is_immediately_usable() -> None:
    gate = DirectionStabilityGate(confirm_seconds=5.0, confirm_updates=3, min_confidence=0.60)

    decision = gate.observe("spot_context", "CE", 0.72, now=100.0)

    assert decision.direction == "CE"
    assert decision.pending is False


def test_opposite_direction_fails_closed_until_persistent() -> None:
    gate = DirectionStabilityGate(confirm_seconds=5.0, confirm_updates=3, min_confidence=0.60)
    gate.observe("spot_context", "CE", 0.80, now=100.0)

    first_flip = gate.observe("spot_context", "PE", 0.82, now=101.0)
    second_flip = gate.observe("spot_context", "PE", 0.83, now=103.0)

    assert first_flip.direction is None
    assert first_flip.pending is True
    assert second_flip.direction is None
    assert second_flip.pending is True

    confirmed = gate.observe("spot_context", "PE", 0.84, now=106.1)

    assert confirmed.direction == "PE"
    assert confirmed.pending is False


def test_flip_candidate_resets_when_original_direction_returns() -> None:
    gate = DirectionStabilityGate(confirm_seconds=5.0, confirm_updates=3, min_confidence=0.60)
    gate.observe("futures_context", "CE", 0.80, now=100.0)
    gate.observe("futures_context", "PE", 0.80, now=101.0)

    recovered = gate.observe("futures_context", "CE", 0.75, now=102.0)

    assert recovered.direction == "CE"
    assert recovered.pending is False
    assert recovered.candidate_direction is None


def test_low_confidence_flip_never_accumulates_confirmation() -> None:
    gate = DirectionStabilityGate(confirm_seconds=2.0, confirm_updates=2, min_confidence=0.70)
    gate.observe("spot_context", "CE", 0.85, now=100.0)

    weak = gate.observe("spot_context", "PE", 0.55, now=110.0)
    weak_again = gate.observe("spot_context", "PE", 0.60, now=120.0)

    assert weak.direction is None
    assert weak.pending is True
    assert weak.candidate_updates == 0
    assert weak_again.direction is None
    assert weak_again.candidate_updates == 0
