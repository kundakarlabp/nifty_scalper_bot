from unittest.mock import MagicMock

from nifty_scalper_bot.strategies.runner import StrategyRunner, StrategyRunnerConfig


class DummyIndicatorEngine:
    def get_history(self, _symbol):
        return [1] * 100

    def get_indicators(self, _symbol):
        return {}


def _runner() -> StrategyRunner:
    mdm = MagicMock()
    mdm.get_active_contract_basket.return_value = None
    return StrategyRunner(
        market_data_manager=mdm,
        indicator_engine=DummyIndicatorEngine(),
        strategy_manager=MagicMock(),
        risk_manager=MagicMock(),
        order_manager=MagicMock(),
        position_manager=MagicMock(),
        config=StrategyRunnerConfig(),
        data_hub=MagicMock(),
    )


def test_active_contract_selection_corrects_stale_legacy_fields(caplog) -> None:
    runner = _runner()
    runner._active_selected_ce = "NFO:NIFTY2660923150CE"
    runner._active_selected_pe = "NFO:NIFTY2660923150PE"
    runner.set_active_trading_universe(
        {
            "selected_ce": "NFO:NIFTY2660923250CE",
            "selected_pe": "NFO:NIFTY2660923250PE",
            "futures_symbol": "NFO:NIFTY26JUNFUT",
            "atm_strike": 23250,
            "option_symbols": ["NFO:NIFTY2660923250CE", "NFO:NIFTY2660923250PE"],
            "token_by_symbol": {
                "NFO:NIFTY2660923250CE": 111,
                "NFO:NIFTY2660923250PE": 222,
                "NFO:NIFTY26JUNFUT": 333,
            },
            "committed_at": "2026-06-09T10:00:00+00:00",
        }
    )

    selection = runner.get_active_contract_selection()

    assert selection["selected_ce"] == "NFO:NIFTY2660923250CE"
    assert selection["selected_pe"] == "NFO:NIFTY2660923250PE"
    assert runner._active_selected_ce == "NFO:NIFTY2660923250CE"
    assert runner._active_selected_pe == "NFO:NIFTY2660923250PE"
    assert selection["selected_ce_token"] == 111
    assert any(record.getMessage().startswith("ACTIVE_SELECTION_DRIFT_CORRECTED") for record in caplog.records)


def test_active_contract_selection_updates_atomically() -> None:
    runner = _runner()
    runner.set_active_trading_universe(
        {
            "selected_ce": "NFO:NIFTY2660923250CE",
            "selected_pe": "NFO:NIFTY2660923250PE",
            "option_symbols": ["NFO:NIFTY2660923250CE", "NFO:NIFTY2660923250PE"],
        }
    )
    runner.set_active_trading_universe(
        {
            "selected_ce": "NFO:NIFTY2660923300CE",
            "selected_pe": "NFO:NIFTY2660923300PE",
            "option_symbols": ["NFO:NIFTY2660923300CE", "NFO:NIFTY2660923300PE"],
        }
    )

    selection = runner.get_active_contract_selection()

    assert selection["selected_ce"] == "NFO:NIFTY2660923300CE"
    assert selection["selected_pe"] == "NFO:NIFTY2660923300PE"
    assert runner._active_option_symbols == {"NFO:NIFTY2660923300CE", "NFO:NIFTY2660923300PE"}


def test_post_market_candle_gap_warning_suppressed(monkeypatch, caplog) -> None:
    import nifty_scalper_bot.strategies.runner as runner_module

    runner = _runner()
    symbol = "NFO:NIFTY2660923250CE"
    runner._symbol_states[symbol] = runner_module.SymbolState.HYDRATING
    runner._required_candles = 1
    runner._has_session_candle_gaps = lambda _symbol: True  # type: ignore[assignment]
    runner._session_gap_count[symbol] = 2
    runner._last_tick_time_by_symbol[symbol] = 0.0
    monkeypatch.setattr(runner_module, "get_runtime_market_mode", lambda: "POST_MARKET")
    monkeypatch.setattr(runner_module, "post_market_suppress_candle_gap_warnings", lambda: True)

    state = runner.update_symbol_hydration(symbol, [1.0, 2.0], {symbol: {"vwap": 100.0, "cum_volume": 10}})

    assert state == runner_module.SymbolState.READY
    assert not any("repeated_missing_candles" in record.getMessage() for record in caplog.records)
