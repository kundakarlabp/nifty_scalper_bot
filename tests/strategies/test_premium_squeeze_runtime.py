from types import SimpleNamespace

from nifty_scalper_bot.strategies.runner import StrategyRunner


class _IndicatorEngine:
    def get_indicators(self, symbol: str):
        return {'rsi': 70.0, 'vwap': 100.0, 'ema': 99.0}

    def get_history(self, symbol: str):
        return [1] * 30


def test_premium_squeeze_has_local_scores():
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._premium_squeeze_last_signal_ts = {}
    runner._underlying_signal_cooldown_seconds = 0.0
    runner._indicator_engine = _IndicatorEngine()
    runner._extract_underlying = lambda _s: 'NIFTY'
    runner._vwap_sl_pct = 1.5
    runner._vwap_tp_pct = 2.0
    runner._warmup_bars_required = 20
    runner._active_selected_ce = 'NFO:NIFTY26MAY24100CE'
    runner._active_selected_pe = 'NFO:NIFTY26MAY24100PE'
    runner._active_atm_strike = 24100
    runner._active_option_symbols = {'NFO:NIFTY26MAY24100CE'}
    runner._extract_strike_from_symbol = lambda _s: 24100
    runner._should_log_throttled = lambda *_a, **_k: False
    runner._logger = SimpleNamespace(info=lambda *a, **k: None)
    signal = runner._maybe_generate_premium_squeeze_signal('NFO:NIFTY26MAY24100CE', 101.0)
    assert signal is not None
    assert signal.metadata['strategy_name'] == 'premium_momentum_squeeze'
    for key in ('direction_score', 'strategy_score', 'option_score', 'data_score', 'rr_score'):
        assert key in signal.metadata
