from datetime import date
from types import SimpleNamespace

from src.strategies.runner import StrategyRunner


def test_sync_atm_state_updates_tokens_and_metadata() -> None:
    ds = SimpleNamespace()

    StrategyRunner._sync_atm_state(
        ds,
        option_type="CE",
        token="12345",
        strike=17050,
        expiry=date(2024, 1, 4),
    )

    assert ds.current_atm_strike == 17050
    assert ds.current_atm_expiry == date(2024, 1, 4)
    assert ds.atm_tokens == (12345, None)


def test_sync_atm_state_preserves_other_side_and_existing_expiry() -> None:
    ds = SimpleNamespace(
        atm_tokens=(555, None),
        current_atm_strike=16900,
        current_atm_expiry="keep",
    )

    StrategyRunner._sync_atm_state(
        ds,
        option_type="PE",
        token=98765,
        strike=17100,
        expiry=None,
    )

    assert ds.atm_tokens == (555, 98765)
    assert ds.current_atm_strike == 17100
    assert ds.current_atm_expiry == "keep"


def test_sync_atm_state_handles_unknown_type_gracefully() -> None:
    ds = SimpleNamespace(atm_tokens=(1, 2))

    StrategyRunner._sync_atm_state(
        ds,
        option_type="UNKNOWN",
        token=333,
        strike=17500,
        expiry=date(2024, 1, 11),
    )

    # Should not change tokens when option type is unknown
    assert ds.atm_tokens == (1, 2)
    # Strike/expiry metadata is still updated when provided
    assert ds.current_atm_strike == 17500
    assert ds.current_atm_expiry == date(2024, 1, 11)
