"""ATM-distance derivation instead of the blanket 999 sentinel.

Production logs showed `distance_from_atm: 999.0` on every candidate,
including exact-ATM selections, because the snapshot rarely carries an
explicit distance and the code fell straight through to the sentinel.

MEASURED SEVERITY (before changing anything): rank_score is NOT the selection
mechanism -- `selected_symbol` arrives as a parameter, already chosen
upstream. rank_score feeds exactly one decision, the signal-attempt debounce,
which compares CURRENT against PREVIOUS rank_score. A uniform sentinel applies
the same `min(999/100, 10) * 0.5 ~= 5.0` penalty to both sides and cancels in
the difference. So this was a diagnostics/latent-trap defect, not corrupted
candidate selection.
"""

from __future__ import annotations

from nifty_scalper_bot.strategies.runner import _derive_strike_distance_from_atm


def test_exact_atm_strike_is_zero_distance() -> None:
    assert _derive_strike_distance_from_atm(
        "NFO:NIFTY26JUL24000CE", {"atm_strike": 24000}, {}
    ) == 0.0


def test_off_atm_strike_distance_from_symbol() -> None:
    assert _derive_strike_distance_from_atm(
        "NFO:NIFTY26JUL23950CE", {"atm_strike": 24000}, {}
    ) == 50.0
    assert _derive_strike_distance_from_atm(
        "NFO:NIFTY26JUL24050PE", {"atm_strike": 24000}, {}
    ) == 50.0


def test_explicit_strike_on_snapshot_wins_over_symbol_parse() -> None:
    assert _derive_strike_distance_from_atm(
        "NFO:NIFTY26JUL23950CE", {"atm_strike": 24000, "strike": 23900}, {}
    ) == 100.0


def test_atm_strike_may_come_from_metadata() -> None:
    assert _derive_strike_distance_from_atm(
        "NFO:NIFTY26JUL23900CE", {}, {"atm_strike": 24000}
    ) == 100.0


def test_returns_none_when_atm_unresolvable() -> None:
    """Caller must keep its explicit unknown marker, not invent a distance."""
    assert _derive_strike_distance_from_atm(
        "NFO:NIFTY26JUL23950CE", {}, {}
    ) is None


def test_returns_none_when_strike_unparseable() -> None:
    assert _derive_strike_distance_from_atm(
        "NOT-AN-OPTION", {"atm_strike": 24000}, {}
    ) is None


def test_malformed_values_do_not_raise() -> None:
    assert _derive_strike_distance_from_atm(
        "NFO:NIFTY26JUL23950CE", {"atm_strike": "abc"}, {}
    ) is None
    assert _derive_strike_distance_from_atm(
        "NFO:NIFTY26JUL23950CE", {"atm_strike": 0}, {}
    ) is None
    assert _derive_strike_distance_from_atm("NFO:NIFTY26JUL23950CE", None, None) is None
