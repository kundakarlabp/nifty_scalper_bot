from __future__ import annotations

import inspect
from types import SimpleNamespace

import nifty_scalper_bot.core as core
import nifty_scalper_bot.core.strategy_context_fast_path as context_fast_path_module
import nifty_scalper_bot.core.strategy_manager as strategy_manager_module
from nifty_scalper_bot.core.strategy_context_fast_path import (
    CONTEXT_REQUIRED_INDICATORS,
    _generate_context_only,
)


class _Indicators:
    def __init__(self) -> None:
        self.calls: list[tuple[str, set[str]]] = []

    def get_indicators(self, symbol: str, required: set[str]):
        self.calls.append((symbol, set(required)))
        return {
            "ltp": 24450.0,
            "close": 24450.0,
            "vwap": 24440.0,
            "ema_fast": 24448.0,
            "ema_slow": 24442.0,
            "volume": 1000.0,
            "avg_volume": 800.0,
        }


def _manager():
    engine = _Indicators()
    updates: list[dict] = []
    augmented: list[dict] = []
    manager = SimpleNamespace(
        _indicator_engine=engine,
        _last_no_signal_decision_by_symbol={"NSE:NIFTY": object()},
        _augment_futures_metrics=lambda indicators: augmented.append(dict(indicators)),
        _update_context_snapshot=lambda **kwargs: updates.append(kwargs),
    )
    return manager, engine, updates, augmented


def test_context_fast_path_uses_compact_indicator_contract() -> None:
    manager, engine, updates, augmented = _manager()

    assert _generate_context_only(manager, "NSE:NIFTY", 24451.0, "spot_context") is None

    assert len(engine.calls) == 1
    symbol, required = engine.calls[0]
    assert symbol == "NSE:NIFTY"
    assert required == set(CONTEXT_REQUIRED_INDICATORS)
    assert "bos_confirmed" not in required
    assert "liquidity_sweep_confirmed" not in required
    assert "selected_ce" not in required
    assert augmented
    assert updates and updates[0]["role"] == "spot_context"
    indicators = updates[0]["indicators"]
    assert indicators["ltp"] == 24450.0
    assert indicators["price"] == 24451.0
    assert indicators["symbol_role"] == "spot_context"
    assert "NSE:NIFTY" not in manager._last_no_signal_decision_by_symbol


def test_context_fast_path_accepts_mapping_like_indicator_result() -> None:
    manager, engine, updates, _ = _manager()

    class _MappingLike:
        def items(self):
            return {"ltp": 25000.0, "vwap": 24990.0}.items()

    engine.get_indicators = lambda _symbol, _required: _MappingLike()

    _generate_context_only(manager, "NFO:NIFTY26AUGFUT", 25001.0, "futures_context")

    assert updates[0]["indicators"]["ltp"] == 25000.0
    assert updates[0]["indicators"]["price"] == 25001.0
    assert updates[0]["role"] == "futures_context"


def test_context_fast_path_fails_closed_to_empty_indicator_mapping() -> None:
    manager, engine, updates, _ = _manager()
    engine.get_indicators = lambda _symbol, _required: None

    _generate_context_only(manager, "NSE:NIFTY", 24451.0, "spot_context")

    indicators = updates[0]["indicators"]
    assert indicators["ltp"] == 24451.0
    assert indicators["close"] == 24451.0
    assert indicators["price"] == 24451.0



def test_strategy_context_fast_path_is_native_not_runtime_replacement() -> None:
    helper_source = inspect.getsource(context_fast_path_module)
    manager_source = inspect.getsource(strategy_manager_module)
    core_source = inspect.getsource(core)

    assert "def apply_patches(" not in helper_source
    assert "StrategyManager.generate_signal =" not in helper_source
    assert "_strategy_context_fast_path_adapter" not in core_source
    assert "_context_only_fast_path_installed" not in core_source
    assert "return _generate_context_only(" in manager_source
    assert "_context_only_fast_path_native = True" in manager_source


def test_native_strategy_manager_short_circuits_context_before_heavy_state(
    monkeypatch,
) -> None:
    calls: list[tuple[str, float, str]] = []

    def _fake_context(
        _manager: object,
        symbol: str,
        current_price: float,
        role: str,
    ) -> None:
        calls.append((symbol, current_price, role))
        return None

    monkeypatch.setattr(
        strategy_manager_module,
        "_generate_context_only",
        _fake_context,
        raising=False,
    )
    native = getattr(
        strategy_manager_module.StrategyManager,
        "_strategy_live_safety_original_generate_signal",
        strategy_manager_module.StrategyManager.generate_signal,
    )

    assert native(SimpleNamespace(), "NSE:NIFTY", 24451.0, trace_id="ctx") is None
    assert calls == [("NSE:NIFTY", 24451.0, "spot_context")]
