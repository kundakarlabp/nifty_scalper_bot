from __future__ import annotations

import pytest

from nifty_scalper_bot.config.score_scale import (
    canonical_score,
    legacy_confidence_score,
    resolve_score_setting,
)


def test_canonical_score_never_infers_fraction_or_percent_units() -> None:
    assert canonical_score(0.75, field="x") == 0.75
    assert canonical_score(7.5, field="x") == 7.5
    with pytest.raises(ValueError, match="within 0..10"):
        canonical_score(75, field="x")


def test_legacy_confidence_converter_is_the_only_multi_unit_adapter() -> None:
    assert legacy_confidence_score("0.75", field="legacy") == 7.5
    assert legacy_confidence_score("7.5", field="legacy") == 7.5
    assert legacy_confidence_score("75", field="legacy") == 7.5
    assert legacy_confidence_score("75%", field="legacy") == 7.5


def test_canonical_setting_takes_precedence_over_legacy_alias() -> None:
    resolved = resolve_score_setting(
        {
            "GLOBAL_MIN_SIGNAL_SCORE": "6.9",
            "GLOBAL_MIN_SIGNAL_CONFIDENCE": "0.5",
        },
        canonical_key="GLOBAL_MIN_SIGNAL_SCORE",
        legacy_keys=("GLOBAL_MIN_SIGNAL_CONFIDENCE",),
        default=6.8,
    )

    assert resolved == 6.9
