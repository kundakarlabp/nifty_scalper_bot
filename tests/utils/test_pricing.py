from __future__ import annotations

import pytest

from nifty_scalper_bot.utils.pricing import canonical_price_source


@pytest.mark.parametrize(
    "source, expected",
    [
        ("live", "live"),
        ("mdm", "mdm"),
        ("historical", "historical"),
        ("est", "est"),
    ],
)
def test_canonical_price_source_exact_matches(source: str, expected: str) -> None:
    """Canonical price source should return exact matches unchanged.

    Args:
        source: Input label to normalise.
        expected: Expected canonical label for assertion.

    Returns:
        None.

    Raises:
        AssertionError: If the canonical label does not match ``expected``.
    """

    assert canonical_price_source(source) == expected


@pytest.mark.parametrize(
    "source, expected",
    [
        ("Live", "live"),
        ("BROKER", "live"),
        ("mdm-feed", "mdm"),
        ("hist_cache", "historical"),
        ("snapshot", "historical"),
        ("approx", "est"),
        (None, "mdm"),
        ("", "mdm"),
        ("unknown", "mdm"),
    ],
)
def test_canonical_price_source_variants(source: object, expected: str) -> None:
    """Canonical price source should map variants to canonical labels.

    Args:
        source: Input label to normalise.
        expected: Expected canonical label for assertion.

    Returns:
        None.

    Raises:
        AssertionError: If the canonical label does not match ``expected``.
    """

    assert canonical_price_source(source) == expected
