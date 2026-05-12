from __future__ import annotations

from nifty_scalper_bot.core import app


class _RunnerEngine:
    def __init__(self) -> None:
        self._h = {'NFO:OPTCE': [1]}

    def get_history(self, symbol: str):
        return self._h.get(symbol, [])


class _Runner:
    def __init__(self) -> None:
        self._indicator_engine = _RunnerEngine()

    def _hydrate_from_mdm_cache(self, symbol: str) -> None:
        self._indicator_engine._h[symbol] = [1] * 100

    def add_symbol(self, symbol: str) -> None:
        _ = symbol


class _MDM:
    def get_ohlc_bars(self, symbol: str):
        return [1] * 100

    def get_symbol_snapshot(self, symbol: str):
        return {'ltp': 10}


class _Ctx:
    strategy_runner = _Runner()
    market_data_manager = _MDM()
    data_hub = None


def test_selected_option_runner_sync_backfills_from_mdm(monkeypatch):
    monkeypatch.setenv('READINESS_OPTION_EVAL_MIN_BARS', '20')
    pending: set[str] = set()
    added = app._gate_runner_symbol_add(_Ctx(), 'NFO:OPTCE', pending, token=1)
    assert added is True
