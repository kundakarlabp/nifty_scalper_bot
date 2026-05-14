import logging
from types import SimpleNamespace
from nifty_scalper_bot.core.app import _emit_option_symbol_pipeline_status


def test_selected_status_uses_ctx_and_basket(caplog):
    caplog.set_level(logging.INFO)
    ctx = SimpleNamespace(
        data_hub=None,
        market_data_manager=None,
        strategy_runner=None,
        selected_ce=None,
        selected_pe=None,
        active_trading_universe={'selected_ce': 'NFO:NIFTY26MAY23700CE', 'selected_pe': 'NFO:XPE'},
    )
    _emit_option_symbol_pipeline_status(ctx, symbol='NFO:NIFTY26MAY23700CE', token=1, selected=False, hydrated_bars=0, runner_added=False, source='t', reason='t')
    rec = next(r for r in caplog.records if r.message.startswith('OPTION_SYMBOL_PIPELINE_STATUS'))
    assert rec.selected is True
