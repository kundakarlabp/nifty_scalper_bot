from __future__ import annotations

from nifty_scalper_bot.execution.readiness import compute_live_readiness


def test_readiness_allows_ws_when_quote_unavailable() -> None:
    armed, reasons = compute_live_readiness(
        live_mode=True,
        hard_ready=True,
        quote_available=False,
        ws_quote_proof=True,
        market_open=True,
    )
    assert armed is True
    assert reasons == []


def test_readiness_blocks_when_no_market_data_proof() -> None:
    armed, reasons = compute_live_readiness(
        live_mode=True,
        hard_ready=True,
        quote_available=False,
        ws_quote_proof=False,
        market_open=True,
    )
    assert armed is False
    assert 'market_data_proof_unavailable' in reasons
