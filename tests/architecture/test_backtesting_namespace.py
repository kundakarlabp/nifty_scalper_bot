"""Architecture guards for the canonical backtesting namespace."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "src" / "nifty_scalper_bot"
LEGACY = PKG / "backtest"
CANONICAL = PKG / "backtesting"

_IMPLEMENTATION_MODULES = ("replay", "parity", "premium_decay_backtest")


def test_backtesting_owns_all_backtest_implementations() -> None:
    missing = [
        name
        for name in _IMPLEMENTATION_MODULES
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


def test_internal_code_does_not_import_legacy_backtest_namespace() -> None:
    offenders: list[tuple[str, str]] = []
    roots = (PKG, ROOT / "tests")
    for root in roots:
        for path in root.rglob("*.py"):
            if LEGACY in path.parents:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                module = None
                if isinstance(node, ast.ImportFrom):
                    module = node.module
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.name
                        if name == "nifty_scalper_bot.backtest" or name.startswith(
                            "nifty_scalper_bot.backtest."
                        ):
                            offenders.append((str(path.relative_to(ROOT)), name))
                    continue
                if module and (
                    module == "nifty_scalper_bot.backtest"
                    or module.startswith("nifty_scalper_bot.backtest.")
                    or module == "src.nifty_scalper_bot.backtest"
                    or module.startswith("src.nifty_scalper_bot.backtest.")
                ):
                    offenders.append((str(path.relative_to(ROOT)), module))
    assert offenders == []


def test_runtime_plotly_and_prometheus_do_not_resolve_to_local_stubs() -> None:
    import plotly
    import prometheus_client

    for module in (plotly, prometheus_client):
        location = Path(module.__file__).resolve()
        assert not location.is_relative_to(ROOT / "src"), location


def test_real_plotly_charts_and_prometheus_exposition_work() -> None:
    from plotly import graph_objects as go
    from plotly.offline import plot
    from prometheus_client import CollectorRegistry, Counter, generate_latest

    html = plot(
        go.Figure(data=[go.Scatter(x=[0, 1], y=[1, 2])]),
        include_plotlyjs="cdn",
        output_type="div",
    )
    assert "Plotly.newPlot" in html

    registry = CollectorRegistry()
    metric = Counter(
        "nifty_dependency_probe_total",
        "Verifies real Prometheus output",
        registry=registry,
    )
    metric.inc()
    assert b"nifty_dependency_probe_total" in generate_latest(registry)
