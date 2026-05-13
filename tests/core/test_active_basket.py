from types import SimpleNamespace

from nifty_scalper_bot.core import app


def test_tradable_quote_uses_symbol_list(monkeypatch) -> None:
    captured = {}

    class Mdm:
        def get_symbol_snapshot(self, _sym):
            return None

        def has_ws_tradable_quote(self, symbols):
            captured['arg'] = symbols
            return True

    ctx = SimpleNamespace(
        active_trading_universe={'option_symbols': ['NFO:ABC24000CE', 'NFO:ABC24000PE'], 'symbols': ['NFO:ABC24000CE', 'NFO:ABC24000PE']},
        market_data_manager=Mdm(),
        selected_ce=None,
        selected_pe=None,
        atm_ce_symbol=None,
        atm_pe_symbol=None,
        strategy_runner=None,
    )

    import asyncio
    asyncio.run(app._recompute_and_push_runtime_readiness(ctx, reason='unit_test'))
    assert captured['arg'] == ['NFO:ABC24000CE'] or captured['arg'] == ['NFO:ABC24000PE']


def test_build_canonical_active_basket_returns_selected_symbols() -> None:
    class IM:
        def is_loaded(self):
            return True
        def select_tokens_for_universe(self, **_k):
            return [1, 2]
        def get_symbol(self, token):
            return {1: 'NIFTY26MAY24000CE', 2: 'NIFTY26MAY24000PE'}[token]
        def get_token(self, _symbol):
            return 99

    basket = app._build_canonical_active_basket(
        instrument_manager=IM(),
        spot_token_resolver=lambda _s: 256265,
        spot_ltp=24010.0,
        futures_symbol='NFO:NIFTYFUT',
    )
    assert basket['selected_ce'] == basket['atm_ce']
    assert basket['selected_pe'] == basket['atm_pe']
