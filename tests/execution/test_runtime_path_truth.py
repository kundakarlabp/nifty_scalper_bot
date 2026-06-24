from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src" / "nifty_scalper_bot"


async def test_runner_dispatches_trade_plan_not_legacy_market_ladder() -> None:
    runner = (SRC_ROOT / "strategies" / "runner.py").read_text(encoding="utf-8")

    assert 'getattr(self._order_manager, "submit_trade_plan_result", None)' in runner
    assert "self._order_manager.execute_market_order(" not in runner


async def test_prototype_lifecycle_modules_are_not_imported_by_production() -> None:
    forbidden = (
        "nifty_scalper_bot.execution.dynamic_tp",
        "nifty_scalper_bot.execution.lifecycle_manager",
    )
    prototypes = {
        SRC_ROOT / "execution" / "dynamic_tp.py",
        SRC_ROOT / "execution" / "lifecycle_manager.py",
    }
    offenders: list[str] = []

    for path in SRC_ROOT.rglob("*.py"):
        if path in prototypes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(module in text for module in forbidden):
            offenders.append(str(path.relative_to(ROOT)))

    assert not offenders, f"Prototype lifecycle modules imported at runtime: {offenders}"
