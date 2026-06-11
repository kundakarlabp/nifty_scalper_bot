"""Regression tests for the single global option-universe cap."""

from __future__ import annotations

from nifty_scalper_bot.instruments.active_contracts import cap_option_universe


def _full_universe(atm: int = 23300, step: int = 50, around: int = 3) -> list[tuple[str, int, float, str]]:
    items: list[tuple[str, int, float, str]] = []
    token = 1000
    for i in range(-around, around + 1):
        strike = atm + i * step
        for side in ("CE", "PE"):
            items.append((f"NFO:NIFTY26616{strike}{side}", token, float(strike), side))
            token += 1
    return items  # 14 options for around=3


def test_cap_never_exceeds_max_and_keeps_selected_pair() -> None:
    universe = _full_universe()
    assert len(universe) == 14
    capped = cap_option_universe(
        universe,
        selected_ce="NFO:NIFTY2661623300CE",
        selected_pe="NFO:NIFTY2661623300PE",
        atm_strike=23300.0,
        max_options=8,
    )
    assert len(capped) == 8
    symbols = {item[0] for item in capped}
    assert "NFO:NIFTY2661623300CE" in symbols and "NFO:NIFTY2661623300PE" in symbols
    # the basket consumers (DataHub flush, MDM, WS tokens, runner, eval) all
    # derive from this list, so 8 here means <=8 everywhere downstream
    assert len({item[1] for item in capped}) == 8  # unique tokens


def test_cap_fills_nearest_strikes_alternating_sides() -> None:
    capped = cap_option_universe(
        _full_universe(),
        selected_ce="NFO:NIFTY2661623300CE",
        selected_pe="NFO:NIFTY2661623300PE",
        atm_strike=23300.0,
        max_options=6,
    )
    distances = [abs(item[2] - 23300.0) for item in capped]
    assert max(distances) <= 50.0  # nearest strikes only
    sides = [item[3] for item in capped]
    assert sides.count("CE") == 3 and sides.count("PE") == 3  # balanced


def test_cap_noop_when_under_limit_or_disabled() -> None:
    universe = _full_universe(around=1)  # 6 options
    assert cap_option_universe(universe, selected_ce=universe[0][0], selected_pe=universe[1][0], atm_strike=23300.0, max_options=8) == universe
    assert cap_option_universe(universe, selected_ce=universe[0][0], selected_pe=universe[1][0], atm_strike=23300.0, max_options=0) == universe
