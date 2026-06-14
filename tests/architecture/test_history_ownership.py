from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "nifty_scalper_bot"


def _calls(path: Path, names: set[str]) -> list[str]:
    tree = ast.parse(path.read_text())
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr in names:
                found.append(fn.attr)
            elif isinstance(fn, ast.Name) and fn.id in names:
                found.append(fn.id)
    return found


def test_only_mdm_calls_broker_historical_data() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.endswith("data/market_data_manager.py"):
            continue
        if "historical_data" in _calls(path, {"historical_data"}):
            offenders.append(rel)
    assert offenders == []


def test_app_uses_canonical_runtime_history_not_mdm_wrappers() -> None:
    text = (SRC / "core" / "app.py").read_text()
    assert "ctx.market_data_manager.hydrate_symbol_history" not in text
    assert "ctx.market_data_manager.fetch_history" not in text
    assert "_indicator_engine.replace_history" not in text


def test_datahub_has_no_authoritative_history_cache() -> None:
    text = (SRC / "data" / "data_hub.py").read_text()
    assert "_history_cache" not in text
    assert "def _normalize_history_rows" not in text


def test_runner_does_not_call_broker_history() -> None:
    text = (SRC / "strategies" / "runner.py").read_text()
    assert ".historical_data(" not in text
    assert ".fetch_history(" not in text


def test_execution_and_order_manager_do_not_hydrate() -> None:
    offenders = []
    for path in (SRC / "execution").rglob("*.py"):
        calls = _calls(path, {"hydrate_symbol_history", "ensure_history", "sync_history_from_mdm", "reseed_history_from_bars"})
        if calls:
            offenders.append((path.relative_to(ROOT).as_posix(), calls))
    assert offenders == []


def test_runner_runtime_history_inflight_is_target_aware() -> None:
    text = (SRC / "strategies" / "runner.py").read_text()
    assert "_runtime_history_ensure_inflight: dict[str, int]" in text
    assert "existing_target" in text and "requested_target" in text
    assert ">= requested_target" in text
    assert "CANONICAL_HISTORY_ENSURE_UPGRADED" in text
