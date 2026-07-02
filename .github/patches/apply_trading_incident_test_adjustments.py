from __future__ import annotations

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one anchor, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# Preserve the geometry-validation test while keeping the protected-price move
# inside the new production repricing tolerance.
replace_once(
    "tests/execution/test_order_manager_trade_plan.py",
    "    m._protected_limit_price = lambda p: 111.0\n",
    "    m._protected_limit_price = lambda p: 107.0\n",
)

# Reanchor remains supported for ordinary quote drift. Abnormal 20%+ repricing is
# now covered by the incident regression and must fail before broker placement.
replace_once(
    "tests/test_bracket_reanchor.py",
    '''async def test_submit_reanchors_instead_of_rejecting(monkeypatch: Any) -> None:\n    """End-to-end: a stale BUY plan is accepted (re-anchored), not rejected."""\n''',
    '''async def test_submit_reanchors_instead_of_rejecting(monkeypatch: Any) -> None:\n    """End-to-end: ordinary quote drift is re-anchored, not rejected."""\n''',
)
replace_once(
    "tests/test_bracket_reanchor.py",
    '    monkeypatch.setattr(mgr, "_protected_limit_price", lambda plan: 138.45)\n',
    '    monkeypatch.setattr(mgr, "_protected_limit_price", lambda plan: 119.00)\n',
)
replace_once(
    "tests/test_bracket_reanchor.py",
    '    assert captured["entry_price"] == 138.45\n    assert captured["stop_loss"] < 138.45 < captured["take_profit"]\n',
    '    assert captured["entry_price"] == 119.00\n    assert captured["stop_loss"] < 119.00 < captured["take_profit"]\n',
)

# The CI suite disables automatic bracket restoration globally to isolate tests.
# This specific persistence test must opt in before constructing the restorer.
replace_once(
    "tests/execution/test_execution_safety_audit_fixes.py",
    '''    manager.save_state()\n\n    restored = BracketManager(order_manager=SimpleNamespace())\n''',
    '''    manager.save_state()\n\n    monkeypatch.setenv("BRACKET_AUTO_RESTORE", "true")\n    restored = BracketManager(order_manager=SimpleNamespace())\n''',
)

# The closed-market branch intentionally short-circuits before querying feed
# health, so its throttle key uses the canonical spot symbol rather than a local
# variable that does not exist on that path.
replace_once(
    "tests/core/test_app_polling_fallback_market_aware.py",
    '    assert "polling_fallback_skipped:{spot_symbol}:market_closed" in s\n',
    '    assert "polling_fallback_skipped:NSE:NIFTY:market_closed" in s\n',
)

# The runner now submits the canonical TradePlan contract. Keep the quantity test
# double on that production path instead of the retired direct place_order route.
replace_once(
    "tests/strategies/test_runner_live_path_guards.py",
    '''    def place_order(self, **kwargs) -> str:\n        self.calls += 1\n        self.last_quantity = int(kwargs.get('quantity') or 0)\n        return f'order-{self.calls}'\n\n    def is_kill_switch_active(self) -> bool:\n''',
    '''    def place_order(self, **kwargs) -> str:\n        self.calls += 1\n        self.last_quantity = int(kwargs.get('quantity') or 0)\n        return f'order-{self.calls}'\n\n    def submit_trade_plan(self, plan) -> str:\n        self.calls += 1\n        self.last_quantity = int(plan.quantity or 0)\n        return f'order-{self.calls}'\n\n    def is_kill_switch_active(self) -> bool:\n''',
)

runpy.run_path(
    str(ROOT / ".github/patches/apply_runner_test_fixture_adjustment.py"),
    run_name="__main__",
)
print("Adjusted tests for bounded reanchoring, explicit restore and current polling contract")
