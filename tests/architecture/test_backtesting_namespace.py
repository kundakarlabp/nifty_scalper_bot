"""Architecture guards for the canonical backtesting namespace."""

from __future__ import annotations

import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "src" / "nifty_scalper_bot"
LEGACY = PKG / "backtest"
CANONICAL = PKG / "backtesting"

_IMPLEMENTATION_MODULES = ("replay", "parity", "premium_decay_backtest")


def test_backtesting_owns_all_backtest_implementations() -> None:
    missing = [
        name for name in _IMPLEMENTATION_MODULES
        if not (CANONICAL / f"{name}.py").exists()
    ]
    assert missing == [], f"canonical backtesting modules missing: {missing}"


def test_legacy_backtest_modules_are_thin_compatibility_shims() -> None:
    for name in _IMPLEMENTATION_MODULES:
        source = (LEGACY / f"{name}.py").read_text(encoding="utf-8")
        assert f"nifty_scalper_bot.backtesting.{name}" in source, name
        assert source.count("\ndef ") == 0, name
        assert source.count("\nclass ") == 0, name


def test_legacy_and_canonical_exports_have_stable_identity() -> None:
    pairs = {
        "replay": ("ReplayHarness", "ReplayResult", "HistoricalContractCatalog"),
        "parity": ("SinglePipelineParity", "PipelineParityResult"),
        "premium_decay_backtest": ("BacktestBar", "run_premium_decay_backtest"),
    }
    for module_name, names in pairs.items():
        legacy = importlib.import_module(f"nifty_scalper_bot.backtest.{module_name}")
        canonical = importlib.import_module(
            f"nifty_scalper_bot.backtesting.{module_name}"
        )
        for name in names:
            assert getattr(legacy, name) is getattr(canonical, name), (
                module_name,
                name,
            )


def test_backtesting_engine_does_not_depend_on_legacy_namespace() -> None:
    source = (CANONICAL / "backtest_engine.py").read_text(encoding="utf-8")
    assert "nifty_scalper_bot.backtest." not in source
