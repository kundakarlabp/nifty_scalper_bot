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


def test_commit_active_dynamic_basket_persists_context_fields() -> None:
    ctx = SimpleNamespace(active_trading_universe={}, selected_ce=None, selected_pe=None, strategy_runner=None)
    basket = {
        "futures_symbol": "NFO:NIFTY26MAYFUT",
        "futures_token": 222,
        "option_symbols": ["NFO:NIFTY26MAY25000CE", "NFO:NIFTY26MAY25000PE"],
        "ce_symbols": ["NFO:NIFTY26MAY25000CE"],
        "pe_symbols": ["NFO:NIFTY26MAY25000PE"],
        "atm_strike": 25000,
    }
    app._commit_active_dynamic_basket(
        ctx,
        basket=basket,
        option_symbols=basket["option_symbols"],
        symbols=basket["option_symbols"],
        atm_strike=25000,
    )
    committed = ctx.active_trading_universe
    assert committed["spot_symbol"] == "NSE:NIFTY"
    assert "spot_symbol" in committed and "futures_symbol" in committed
    assert "NSE:NIFTY" in committed["symbols"]
    assert "NFO:NIFTY26MAYFUT" in committed["symbols"]


def test_commit_active_dynamic_basket_repairs_stale_selected_pair_after_atm_shift() -> None:
    old_ce = "NFO:NIFTY26AUG24250CE"
    old_pe = "NFO:NIFTY26AUG24250PE"
    new_ce = "NFO:NIFTY26AUG24300CE"
    new_pe = "NFO:NIFTY26AUG24300PE"
    option_symbols = [old_ce, old_pe, new_ce, new_pe]
    ctx = SimpleNamespace(
        active_trading_universe={
            "selected_ce": old_ce,
            "selected_pe": old_pe,
            "atm_strike": 24250,
        },
        selected_ce=old_ce,
        selected_pe=old_pe,
        strategy_runner=None,
    )

    committed_ce, committed_pe = app._commit_active_dynamic_basket(
        ctx,
        basket=ctx.active_trading_universe,
        option_symbols=option_symbols,
        symbols=option_symbols,
        atm_strike=24300,
    )

    assert (committed_ce, committed_pe) == (new_ce, new_pe)
    assert ctx.active_trading_universe["atm_strike"] == 24300
