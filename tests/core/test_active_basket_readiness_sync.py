from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.core.history_readiness import compute_selected_option_history_readiness


OLD_CE = "NFO:NIFTY2670724350CE"
OLD_PE = "NFO:NIFTY2670724350PE"
NEW_CE = "NFO:NIFTY2670724400CE"
NEW_PE = "NFO:NIFTY2670724400PE"


class _FakeMarketDataManager:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def get_ohlc_bars(self, symbol: str, limit: int | None = None) -> list[object]:
        del limit
        return [object()] * self._counts.get(symbol, 0)


class _FakeRunner:
    _option_required_bars = 30

    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def runner_history_count(self, symbol: str) -> int:
        return self._counts.get(symbol, 0)

    def indicator_history_count(self, symbol: str) -> int:
        return self._counts.get(symbol, 0)


def test_selected_option_readiness_prefers_active_contract_basket_over_stale_legacy_symbols() -> None:
    counts = {
        OLD_CE: 100,
        OLD_PE: 100,
        NEW_CE: 0,
        NEW_PE: 0,
    }
    ctx = SimpleNamespace(
        active_contract_basket={"selected_ce": NEW_CE, "selected_pe": NEW_PE},
        selected_ce=OLD_CE,
        selected_pe=OLD_PE,
        market_data_manager=_FakeMarketDataManager(counts),
        strategy_runner=_FakeRunner(counts),
    )

    result = compute_selected_option_history_readiness(ctx, OLD_CE, OLD_PE)

    assert result.selected_ce == NEW_CE
    assert result.selected_pe == NEW_PE
    assert result.both_ready is False
    assert result.blocker == "selected_option_history_cold"


def test_selected_option_readiness_arms_only_when_active_basket_pair_is_ready() -> None:
    counts = {
        NEW_CE: 35,
        NEW_PE: 35,
    }
    ctx = SimpleNamespace(
        active_contract_basket={"selected_ce": NEW_CE, "selected_pe": NEW_PE},
        selected_ce=OLD_CE,
        selected_pe=OLD_PE,
        market_data_manager=_FakeMarketDataManager(counts),
        strategy_runner=_FakeRunner(counts),
    )

    result = compute_selected_option_history_readiness(ctx, OLD_CE, OLD_PE)

    assert result.selected_ce == NEW_CE
    assert result.selected_pe == NEW_PE
    assert result.both_ready is True
    assert result.blocker is None
