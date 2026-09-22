from datetime import date

from nifty_scalper_bot.instruments.active_contracts import parse_nifty_future_expiry
from nifty_scalper_bot.utils.smart_symbol import get_nifty_monthly_expiry_date
from scripts.sync_nifty_options import classify


def test_nifty_monthly_expiry_preserves_pre_transition_thursday_rule() -> None:
    assert get_nifty_monthly_expiry_date(2025, 8) == date(2025, 8, 28)


def test_nifty_monthly_expiry_uses_tuesday_from_september_2025() -> None:
    assert get_nifty_monthly_expiry_date(2025, 9) == date(2025, 9, 30)


def test_nifty_monthly_expiry_moves_back_over_nse_holiday() -> None:
    # 31 Mar 2026 is an NSE F&O holiday (Mahavir Jayanti).
    assert get_nifty_monthly_expiry_date(2026, 3) == date(2026, 3, 30)


def test_future_symbol_fallback_uses_canonical_nifty_monthly_expiry() -> None:
    assert parse_nifty_future_expiry("NFO:NIFTY26MARFUT") == date(2026, 3, 30)


def test_bhavcopy_classification_uses_same_date_aware_monthly_rule() -> None:
    assert classify(date(2025, 8, 28)) == "MONTHLY"
    assert classify(date(2025, 8, 21)) == "WEEKLY"
    assert classify(date(2026, 3, 30)) == "MONTHLY"
    assert classify(date(2026, 3, 24)) == "WEEKLY"
