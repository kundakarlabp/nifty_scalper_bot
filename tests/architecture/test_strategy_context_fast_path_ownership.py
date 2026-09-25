from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[2]
CORE = ROOT / "src/nifty_scalper_bot/core"


def test_strategy_context_fast_path_is_native_not_runtime_replacement() -> None:
    helper_source = (CORE / "strategy_context_fast_path.py").read_text(encoding="utf-8")
    manager_source = (CORE / "strategy_manager.py").read_text(encoding="utf-8")
    core_source = (CORE / "__init__.py").read_text(encoding="utf-8")

    assert "def apply_patches(" not in helper_source
    assert "StrategyManager.generate_signal =" not in helper_source
    assert "_strategy_context_fast_path_adapter" not in core_source
    assert "_context_only_fast_path_installed" not in core_source
    assert "return _generate_context_only(" in manager_source
    assert "_context_only_fast_path_native = True" in manager_source
