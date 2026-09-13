from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.risk.risk_manager import RiskManager


def _risk(lookup=None) -> RiskManager:
    owner = object.__new__(RiskManager)
    owner._lot_size_lookup = lookup
    owner._lot_size_symbol = "NIFTY"
    owner._logger = SimpleNamespace(info=lambda *_a, **_k: None)
    return owner


def test_lot_size_resolution_is_owned_by_risk_manager_module() -> None:
    assert RiskManager._resolve_lot_size.__module__ == (
        "nifty_scalper_bot.risk.risk_manager"
    )


def test_lot_size_requires_the_injected_instrument_provider() -> None:
    with pytest.raises(RuntimeError, match="lot size provider not configured"):
        _risk()._resolve_lot_size("NIFTY26SEP25000CE")


def test_lot_size_uses_provider_without_secondary_fallback() -> None:
    risk = _risk(lambda symbol: 65 if symbol == "NIFTY26SEP25000CE" else 0)

    assert risk._resolve_lot_size("NIFTY26SEP25000CE") == 65
