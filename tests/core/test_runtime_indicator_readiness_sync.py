"""The post-hydration indicator safety flag must follow current live readiness."""

from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.core.app import _sync_runtime_indicator_readiness


SELECTED_CE = "NFO:NIFTY26NOV24900CE"


def _ctx(bars: list[int]) -> SimpleNamespace:
    class IndicatorEngine:
        def has_min_bars(self, symbol: str, minimum: int) -> bool:
            return symbol == SELECTED_CE and bars[0] >= minimum

        def get_history(self, symbol: str) -> list:
            return [object()] * bars[0] if symbol == SELECTED_CE else []

    return SimpleNamespace(
        strategy_runner=SimpleNamespace(
            _required_candles=30,
            _active_symbols={SELECTED_CE},
        ),
        indicator_engine=IndicatorEngine(),
        data_hub=SimpleNamespace(indicators_ready=False),
        market_regime_manager=SimpleNamespace(indicators_ready=False),
        evaluation_ready=False,
        live_orders_armed=False,
    )


def test_runtime_indicator_flag_rearms_when_live_history_becomes_ready() -> None:
    bars = [25]
    ctx = _ctx(bars)
    assert _sync_runtime_indicator_readiness(ctx, data_hard_ready=True) is False
    assert ctx.data_hub.indicators_ready is False

    # After five genuine closed candles, the authoritative per-symbol
    # indicator count crosses the existing 30-bar threshold.
    bars[0] = 30
    assert _sync_runtime_indicator_readiness(ctx, data_hard_ready=True) is True
    assert ctx.data_hub.indicators_ready is True
    assert ctx.market_regime_manager.indicators_ready is True
    # Updating the indicator flag never itself arms execution.
    assert ctx.live_orders_armed is False


def test_runtime_indicator_flag_fails_closed_on_regression() -> None:
    bars = [30]
    ctx = _ctx(bars)
    assert _sync_runtime_indicator_readiness(ctx, data_hard_ready=True) is True

    # Good local indicators cannot override a failed canonical data-hard gate.
    assert _sync_runtime_indicator_readiness(ctx, data_hard_ready=False) is False
    assert ctx.data_hub.indicators_ready is False
    assert ctx.market_regime_manager.indicators_ready is False

    # And loss of actual indicator history must revoke readiness.
    bars[0] = 29
    assert _sync_runtime_indicator_readiness(ctx, data_hard_ready=True) is False


def test_runtime_indicator_flag_requires_runner_and_indicator_engine() -> None:
    ctx = SimpleNamespace(
        strategy_runner=None,
        indicator_engine=None,
        data_hub=SimpleNamespace(indicators_ready=True),
    )
    assert _sync_runtime_indicator_readiness(ctx, data_hard_ready=True) is False
    assert ctx.data_hub.indicators_ready is False
