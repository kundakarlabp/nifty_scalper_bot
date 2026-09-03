from __future__ import annotations

from nifty_scalper_bot.core.strategy_context_builder import build_strategy_history_context

SPOT = "NSE:NIFTY"
FUTURE = "NFO:NIFTY26SEPFUT"
OPTION = "NFO:NIFTY2690823950CE"


class _Hub:
    def get_active_contract_basket(self):
        return {
            "spot_symbol": SPOT,
            "futures_symbol": FUTURE,
            "selected_ce": OPTION,
        }

    def get_ohlc_bars(self, symbol: str):
        if symbol == OPTION:
            return [{"timestamp": index, "close": 100.0 + index} for index in range(35)]
        return []


class _Indicator:
    def get_history(self, _symbol: str):
        return []


def test_option_context_carries_active_underlying_symbol_identities() -> None:
    context = build_strategy_history_context(
        symbol=OPTION,
        indicator_engine=_Indicator(),
        data_hub=_Hub(),
        runner_context={},
    )

    assert context["spot_symbol"] == SPOT
    assert context["futures_symbol"] == FUTURE
    assert context["history_domain_used"] == "options"


def test_runner_context_underlying_identity_overrides_active_basket() -> None:
    next_future = "NFO:NIFTY26OCTFUT"
    context = build_strategy_history_context(
        symbol=OPTION,
        indicator_engine=_Indicator(),
        data_hub=_Hub(),
        runner_context={"spot_symbol": SPOT, "futures_symbol": next_future},
    )

    assert context["spot_symbol"] == SPOT
    assert context["futures_symbol"] == next_future


def test_non_option_history_context_does_not_manufacture_option_linkage() -> None:
    context = build_strategy_history_context(
        symbol=SPOT,
        indicator_engine=_Indicator(),
        data_hub=_Hub(),
        runner_context={},
    )

    assert "spot_symbol" not in context
    assert "futures_symbol" not in context
