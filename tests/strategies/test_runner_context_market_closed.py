from nifty_scalper_bot.strategies.runner import StrategyRunner
import inspect

def test_runner_phase9_market_closed_context_skip():
    r = StrategyRunner.__new__(StrategyRunner)
    assert r._symbol_role_for_runner('NSE:NIFTY') == 'spot_context'
    assert r._symbol_role_for_runner('NFO:NIFTY26MAYFUT') == 'futures_context'


def test_runner_market_closed_option_path_returns_before_strategy_manager_call() -> None:
    source = inspect.getsource(StrategyRunner._on_tick)
    compact_log_idx = source.index("MARKET_SESSION_CLOSED orders_disabled=True reason=normal_close")
    market_closed_emit_idx = source.index('reason="market_closed"')
    generate_signal_idx = source.index("self._strategy_manager.generate_signal(")
    assert compact_log_idx < market_closed_emit_idx < generate_signal_idx
