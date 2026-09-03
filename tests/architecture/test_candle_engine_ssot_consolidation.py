from __future__ import annotations

import ast
import inspect
from pathlib import Path

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "nifty_scalper_bot"


async def test_mdm_capacity_is_candle_engine_capacity_not_tick_cache() -> None:
    source = inspect.getsource(MarketDataManager.history_capacity_for)
    assert "get_candle_engine" in source
    assert "max_bars" in source
    assert "_cache_len" not in source


async def test_mdm_canonical_read_is_candle_engine_backed() -> None:
    source = inspect.getsource(MarketDataManager.get_ohlc_bars)
    assert "get_candle_engine" in source
    assert "get_completed_bars" in source


async def test_projection_capacity_runtime_adapter_is_removed() -> None:
    assert not (SRC / "data" / "ohlc_capacity_contract.py").exists()
    runtime = (SRC / "core" / "runtime_history_event_loop_hardening.py").read_text(
        encoding="utf-8"
    )
    assert "install_mdm_ohlc_capacity_contract" not in runtime
    assert "MDM_OHLC_CACHE_LEN" not in "\n".join(
        path.read_text(encoding="utf-8")
        for path in SRC.rglob("*.py")
    )


async def test_context_history_continuity_is_native_not_monkeypatched() -> None:
    assert not (SRC / "core" / "context_history_continuity.py").exists()
    dynamic = (SRC / "core" / "strategy_runner_dynamic_universe_safety.py").read_text(
        encoding="utf-8"
    )
    assert "context_history_continuity" not in dynamic
    assert "_context_history_read_limit" not in dynamic

    runner = (SRC / "strategies" / "runner.py").read_text(encoding="utf-8")
    assert "def _sync_context_history_if_cold" in runner
    assert "context_tick_bar_sync" in runner
    assert "_CONTEXT_SESSION_HISTORY_BARS" in runner


async def test_strategy_history_context_has_one_implementation() -> None:
    canonical_path = SRC / "core" / "strategy_context_builder.py"
    legacy_path = SRC / "strategies" / "signal_generator.py"

    canonical_tree = ast.parse(canonical_path.read_text(encoding="utf-8"))
    legacy_tree = ast.parse(legacy_path.read_text(encoding="utf-8"))
    canonical_defs = [
        node
        for node in ast.walk(canonical_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "build_strategy_history_context"
    ]
    legacy_defs = [
        node
        for node in ast.walk(legacy_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "build_strategy_history_context"
    ]
    assert len(canonical_defs) == 1
    assert legacy_defs == []

    strategy_manager = (SRC / "core" / "strategy_manager.py").read_text(encoding="utf-8")
    assert "from nifty_scalper_bot.core.strategy_context_builder import" in strategy_manager


async def test_order_and_datahub_layers_still_do_not_own_history() -> None:
    datahub = (SRC / "data" / "data_hub.py").read_text(encoding="utf-8")
    order_manager = (SRC / "execution" / "order_manager.py").read_text(encoding="utf-8")
    assert "_history_cache" not in datahub
    for forbidden in ("CandleEngine(", "ensure_history(", "historical_data("):
        assert forbidden not in order_manager
