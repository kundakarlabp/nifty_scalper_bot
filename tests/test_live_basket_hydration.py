from __future__ import annotations

import pytest
from types import SimpleNamespace

from nifty_scalper_bot.core import app


@pytest.mark.asyncio
async def test_build_live_basket_does_not_double_hydrate_when_hydrate_false(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {'fetch_history': 0}

    async def _fetch_history(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        called['fetch_history'] += 1
        return []

    mdm = SimpleNamespace(
        ensure_tracking=lambda *_a, **_k: None,
        register_symbol=lambda *_a, **_k: None,
        request_token_subscription=lambda *_a, **_k: True,
        fetch_history=_fetch_history,
        ingest_historical_bar=lambda *_a, **_k: None,
        update_hydration_status=lambda *_a, **_k: None,
        get_ohlc_bars=lambda *_a, **_k: [],
        set_readiness_requirements=lambda **_k: None,
    )
    im = SimpleNamespace(is_loaded=lambda: True, get_token=lambda _s: 111)
    bc = SimpleNamespace(get_instrument_token=lambda _s: 111)
    settings = SimpleNamespace(option_universe=SimpleNamespace(strike_step=50))
    ctx = SimpleNamespace(instrument_manager=im, market_data_manager=mdm, broker_client=bc, settings=settings, strategy_runner=None)

    monkeypatch.setattr(app, '_build_canonical_active_basket', lambda **_k: {'symbols': ['NSE:NIFTY', 'NFO:ABC'], 'spot_symbol': 'NSE:NIFTY', 'futures_symbol': 'NFO:FUT', 'ce_symbols': ['NFO:CE'], 'pe_symbols': ['NFO:PE'], 'option_symbols': ['NFO:CE', 'NFO:PE'], 'atm_strike': 25000})

    await app._build_and_hydrate_live_basket_from_spot(ctx, spot_ltp=25000.0, configured_mode='LIVE', hydrate=False)
    assert called['fetch_history'] == 0
