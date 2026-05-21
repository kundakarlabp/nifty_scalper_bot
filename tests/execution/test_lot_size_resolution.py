import os

from nifty_scalper_bot.utils.lot_size import resolve_lot_size


def test_resolve_lot_size_prefixed_nifty_option_env_fallback(monkeypatch):
    monkeypatch.setenv("NIFTY_LOT_SIZE", "75")
    lot, source = resolve_lot_size("NFO:NIFTY26MAY23650PE", lookup=lambda _s: None)
    assert lot == 75
    assert source == "env_fallback"


def test_resolve_lot_size_unprefixed_nifty_option_env_fallback(monkeypatch):
    monkeypatch.setenv("INSTRUMENTS__NIFTY_LOT_SIZE", "65")
    monkeypatch.delenv("NIFTY_LOT_SIZE", raising=False)
    lot, source = resolve_lot_size("NIFTY26MAY23650PE", lookup=lambda _s: None)
    assert lot == 65
    assert source in {"env_fallback", "fallback_default"}


def test_unknown_symbol_does_not_use_nifty_fallback(monkeypatch):
    monkeypatch.delenv("NIFTY_LOT_SIZE", raising=False)
    monkeypatch.delenv("INSTRUMENTS__NIFTY_LOT_SIZE", raising=False)
    lot, source = resolve_lot_size("NFO:BANKNIFTY26MAY52000PE", lookup=lambda _s: 30)
    assert lot == 30
    assert source == "instrument_dump"
