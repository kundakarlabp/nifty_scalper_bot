from __future__ import annotations

from nifty_scalper_bot.config.settings import get_settings


def test_nifty_lot_size_default_fallback_65(monkeypatch) -> None:
    monkeypatch.delenv('NIFTY_LOT_SIZE', raising=False)
    settings = get_settings()
    assert settings.risk.contract_lot_size == 65
