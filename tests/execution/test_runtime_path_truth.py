from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src" / "nifty_scalper_bot"


async def test_runner_dispatches_trade_plan_not_legacy_market_ladder() -> None:
    runner = (SRC_ROOT / "strategies" / "runner.py").read_text(encoding="utf-8")

    assert 'getattr(self._order_manager, "submit_trade_plan_result", None)' in runner
    assert "self._order_manager.execute_market_order(" not in runner
