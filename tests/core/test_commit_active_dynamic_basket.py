from types import SimpleNamespace

from nifty_scalper_bot.core import app


def test_commit_active_dynamic_basket_selects_ce_pe_without_atm() -> None:
    ctx = SimpleNamespace(
        selected_ce=None,
        selected_pe=None,
        atm_ce_symbol=None,
        atm_pe_symbol=None,
        active_trading_universe={},
        strategy_runner=None,
    )
    options = [
        'NFO:NIFTY2460019900CE',
        'NFO:NIFTY2460020000CE',
        'NFO:NIFTY2460019900PE',
        'NFO:NIFTY2460020000PE',
    ]
    selected_ce, selected_pe = app._commit_active_dynamic_basket(
        ctx,
        basket={},
        option_symbols=options,
        symbols=['NSE:NIFTY', *options],
        atm_strike=None,
    )
    assert selected_ce is not None and selected_ce.endswith('CE')
    assert selected_pe is not None and selected_pe.endswith('PE')


def test_commit_active_dynamic_basket_preserves_old_selected_when_valid() -> None:
    old_ce = 'NFO:NIFTY2460020000CE'
    old_pe = 'NFO:NIFTY2460020000PE'
    ctx = SimpleNamespace(
        selected_ce=old_ce,
        selected_pe=old_pe,
        atm_ce_symbol=old_ce,
        atm_pe_symbol=old_pe,
        active_trading_universe={},
        strategy_runner=None,
    )
    selected_ce, selected_pe = app._commit_active_dynamic_basket(
        ctx,
        basket={},
        option_symbols=[old_ce, old_pe],
        symbols=['NSE:NIFTY', old_ce, old_pe],
        atm_strike=None,
    )
    assert selected_ce == old_ce
    assert selected_pe == old_pe
