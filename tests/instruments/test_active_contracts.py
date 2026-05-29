from datetime import date

from nifty_scalper_bot.instruments.active_contracts import (
    derive_nifty_future_from_selected_options,
    parse_nifty_future_expiry,
    resolve_active_nifty_future_from_instruments,
)


def test_parse_nifty_future_expiry_last_thursday_fallback_calendar_only() -> None:
    assert parse_nifty_future_expiry("NFO:NIFTY26MAYFUT") == date(2026, 5, 28)


def test_resolve_active_nifty_future_from_instruments_earliest_non_expired() -> None:
    rows = [
        {"exchange": "NFO", "instrument_type": "FUT", "name": "NIFTY", "tradingsymbol": "NIFTY26MAYFUT", "expiry": "2026-05-26"},
        {"exchange": "NFO", "instrument_type": "FUT", "name": "NIFTY", "tradingsymbol": "NIFTY26JUNFUT", "expiry": "2026-06-25"},
    ]
    out = resolve_active_nifty_future_from_instruments(rows, now=date(2026, 5, 27))
    assert out.symbol == "NFO:NIFTY26JUNFUT"


def test_derive_nifty_future_from_selected_options() -> None:
    out = derive_nifty_future_from_selected_options(["NFO:NIFTY26JUN23900CE", "NFO:NIFTY26JUN23900PE"])
    assert out.symbol == "NFO:NIFTY26JUNFUT"


def test_active_contracts_no_selected_option_fallback_by_default() -> None:
    out = resolve_active_nifty_future_from_instruments([], now=date(2026, 5, 27))
    assert out.symbol is None
    from nifty_scalper_bot.instruments.active_contracts import resolve_active_nifty_future
    selected = ["NFO:NIFTY26MAY23900CE"]
    assert resolve_active_nifty_future(selected_option_symbols=selected, now=date(2026, 5, 27)).symbol is None
    assert resolve_active_nifty_future(selected_option_symbols=selected, now=date(2026, 5, 27), allow_selected_option_fallback=True).symbol == "NFO:NIFTY26MAYFUT"
