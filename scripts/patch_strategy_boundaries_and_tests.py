#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, value: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(value, encoding="utf-8")


def replace_once(path: str, old: str, new: str, *, sentinel: str | None = None) -> None:
    value = read(path)
    if sentinel and sentinel in value:
        return
    count = value.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}: {old[:100]!r}")
    write(path, value.replace(old, new, 1))


def insert_before(path: str, marker: str, block: str, *, sentinel: str) -> None:
    value = read(path)
    if sentinel in value:
        return
    if value.count(marker) != 1:
        raise RuntimeError(f"{path}: insertion marker drifted")
    write(path, value.replace(marker, block + marker, 1))


def append_once(path: str, block: str, *, sentinel: str) -> None:
    value = read(path)
    if sentinel in value:
        return
    write(path, value.rstrip() + "\n\n" + block.strip() + "\n")


def patch_premium_decay() -> None:
    path = "src/nifty_scalper_bot/strategies/premium_decay.py"
    replace_once(path, "from dataclasses import dataclass\n", "from dataclasses import dataclass\nimport os\n")
    replace_once(
        path,
        "class PremiumDecayStrategy:\n    \"\"\"Implement a short strangle with theta capture risk management.\"\"\"\n",
        "class PremiumDecayStrategy:\n    \"\"\"Backtest/paper short-strangle strategy; never a live execution authority.\"\"\"\n\n    LIVE_CAPABLE = False\n",
        sentinel="LIVE_CAPABLE = False",
    )
    insert_before(
        path,
        "        try:\n            if self._active is not None:\n",
        '''        execution_mode = str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper()
        live_enabled = execution_mode == "LIVE" or str(
            os.getenv("ENABLE_LIVE", os.getenv("ENABLE_LIVE_TRADING", "false"))
        ).strip().lower() in {"1", "true", "yes", "on"}
        if live_enabled:
            self._logger.error(
                "PREMIUM_DECAY_LIVE_DISABLED reason=noncanonical_multileg_execution",
                extra={
                    "event": "PREMIUM_DECAY_LIVE_DISABLED",
                    "underlying": underlying,
                    "reason": "noncanonical_multileg_execution",
                },
            )
            return False
''',
        sentinel="PREMIUM_DECAY_LIVE_DISABLED",
    )


def patch_architecture_test() -> None:
    append_once(
        "tests/architecture/test_canonical_bo_ownership.py",
        '''def test_live_capable_strategies_do_not_bypass_trade_plan_execution() -> None:
    offenders: list[str] = []
    strategies_root = SRC / "strategies"
    for path in strategies_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if ".place_order(" not in source:
            continue
        if path.name == "premium_decay.py" and "LIVE_CAPABLE = False" in source:
            continue
        offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"Live-capable strategy bypasses canonical TradePlan path: {offenders}"
''',
        sentinel="test_live_capable_strategies_do_not_bypass_trade_plan_execution",
    )


def patch_premium_test() -> None:
    append_once(
        "tests/strategies/test_premium_decay_strategy.py",
        '''def test_live_mode_disables_noncanonical_premium_decay_entry(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    order_manager = _StubOrderManager()
    strategy = PremiumDecayStrategy(
        order_manager=order_manager,
        risk_manager=_StubRiskManager(),
        orchestrator=_StubOrchestrator(),
        position_manager=_StubPositionManager(),
    )
    option_chain = [
        _make_contract("NFO:NIFTY26JUN24000CE", "CE", 24000, 0.25),
        _make_contract("NFO:NIFTY26JUN24000PE", "PE", 24000, -0.25),
    ]
    assert strategy.evaluate_entry(
        underlying="NIFTY",
        indicators={"atr": 10.0, "adx": 10.0},
        option_chain=option_chain,
        iv=0.20,
    ) is False
    assert order_manager.placed == []
''',
        sentinel="test_live_mode_disables_noncanonical_premium_decay_entry",
    )


def patch_conflict_tests() -> None:
    path = "tests/strategies/test_order_flow_conflict_override.py"
    value = read(path)
    if '"quote_update_version": 1,' not in value:
        value = value.replace(
            '"is_selected_option": True, "strike_distance_from_atm": 0,\n',
            '"is_selected_option": True, "strike_distance_from_atm": 0,\n        "quote_update_version": 1,\n',
            1,
        )
    old = '''# B. Stale PE bias + CE candidate WITH confirming microstructure -> allowed
def test_stale_pe_strong_micro_ce_allowed(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    sig = _eval(strat, "NFO:NIFTY26MAY24000CE", _ind("PE", "UP", buy=400, sell=80))
    assert sig.metadata["trigger_conditions_met"] is True
    assert sig.metadata["bias_invalidated_by_microstructure"] is True


# C. Reversal down: stale CE bias + PE candidate with confirming PE demand -> allowed
def test_stale_ce_strong_micro_pe_allowed(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    sig = _eval(strat, "NFO:NIFTY26MAY24000PE", _ind("CE", "UP", buy=400, sell=80))
    assert sig.metadata["trigger_conditions_met"] is True
    assert sig.metadata["bias_invalidated_by_microstructure"] is True
'''
    new = '''# B. One strong snapshot cannot invalidate directional context
def test_stale_pe_single_strong_micro_ce_blocked(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ORDERFLOW_REVERSAL_MIN_UPDATES", "3")
    monkeypatch.setenv("ORDERFLOW_REVERSAL_MIN_PERSISTENCE_MS", "0")
    sig = _eval(strat, "NFO:NIFTY26MAY24000CE", _ind("PE", "UP", buy=400, sell=80, quote_update_version=1))
    assert sig.metadata["trigger_conditions_met"] is False
    assert sig.metadata["bias_invalidated_by_microstructure"] is False


# C. Reversal becomes eligible only after distinct persistent updates
def test_stale_ce_strong_micro_pe_requires_persistence(monkeypatch, strat):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ORDERFLOW_REVERSAL_MIN_UPDATES", "3")
    monkeypatch.setenv("ORDERFLOW_REVERSAL_MIN_PERSISTENCE_MS", "0")
    results = [
        _eval(strat, "NFO:NIFTY26MAY24000PE", _ind("CE", "UP", buy=400, sell=80, quote_update_version=version))
        for version in (1, 2, 3)
    ]
    assert results[0].metadata["trigger_conditions_met"] is False
    assert results[1].metadata["trigger_conditions_met"] is False
    assert results[2].metadata["trigger_conditions_met"] is True
    assert results[2].metadata["bias_invalidated_by_microstructure"] is True
    assert results[2].metadata["reversal_persistence_confirmed"] is True
'''
    if old not in value and "test_stale_pe_single_strong_micro_ce_blocked" not in value:
        raise RuntimeError(f"{path}: conflict test anchor drifted")
    if old in value:
        value = value.replace(old, new, 1)
    write(path, value)


def create_quote_tests() -> None:
    write(
        "tests/execution/test_quote_readiness.py",
        '''from __future__ import annotations

from nifty_scalper_bot.execution.quote_readiness import (
    evaluate_execution_quote,
    resolve_real_tick_count,
    resolve_tick_age_ms,
)


def test_missing_age_never_becomes_fresh_in_live_mode():
    result = evaluate_execution_quote(
        "NFO:NIFTY26JUN24000CE",
        {"bid": 99.9, "ask": 100.1, "depth_available": True},
        live_mode=True,
        max_tick_age_ms=2500,
        max_spread_pct=0.75,
        require_depth=True,
    )
    assert result.allowed is False
    assert result.reason == "tick_age_missing"


def test_tick_age_schema_is_canonical():
    assert resolve_tick_age_ms({"tick_age_ms": 125}) == 125
    assert resolve_tick_age_ms({"tick_age_s": 0.25}) == 250


def test_fresh_ms_quote_can_prove_one_recent_update():
    ticks, derived = resolve_real_tick_count(
        {"tick_age_ms": 100},
        tick_age_ms=100,
        max_age_ms=2500,
        has_bid_ask=True,
    )
    assert ticks == 1
    assert derived is True


def test_seconds_only_age_does_not_invent_tick_count():
    ticks, derived = resolve_real_tick_count(
        {"tick_age_s": 0.1},
        tick_age_ms=100,
        max_age_ms=2500,
        has_bid_ask=True,
    )
    assert ticks == 0
    assert derived is False
''',
    )


if __name__ == "__main__":
    patch_premium_decay()
    patch_architecture_test()
    patch_premium_test()
    patch_conflict_tests()
    create_quote_tests()
