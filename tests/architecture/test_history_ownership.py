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
    assert "history_inflight" not in text
    assert "def _normalize_history_rows" not in text
    assert ".historical_data(" not in text


def test_runner_does_not_call_broker_or_owner_hydration_directly() -> None:
    text = (SRC / "strategies" / "runner.py").read_text()
    assert ".historical_data(" not in text
    assert ".fetch_history(" not in text
    assert ".ensure_history(" not in text
    assert ".request_hydration(" not in text


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
    assert "_runtime_history_ensure_inflight: set" not in text
    assert "isinstance(inflight, set)" not in text
    assert "existing_target" in text and "requested_target" in text
    assert ">= requested_target" in text
    assert "CANONICAL_HISTORY_ENSURE_UPGRADED" in text


def test_no_old_set_based_runtime_history_fixtures() -> None:
    offenders = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if any(part in rel for part in (".git/", ".pytest_cache/", "__pycache__/")):
            continue
        text = path.read_text()
        if "_runtime_history_ensure_inflight = set()" in text:
            offenders.append(rel)
    assert offenders == []


def test_single_production_runtime_history_ensurer_injection() -> None:
    text = (SRC / "core" / "app.py").read_text()
    assert text.count("set_runtime_history_ensurer(runtime_history_ensurer)") == 1
    assert text.count("async def runtime_history_ensurer") == 1


def test_one_canonical_role_resolver_and_readiness_helper() -> None:
    app_text = (SRC / "core" / "app.py").read_text()
    assert app_text.count("def resolve_symbol_history_role") == 1
    assert app_text.count("def compute_history_readiness") == 1
    assert "class HistoryPolicyDecision" not in app_text
    assert "class HistoryPolicy" in app_text


def test_datahub_compatibility_has_no_target_policy_or_coordinator() -> None:
    text = (SRC / "data" / "data_hub.py").read_text()
    assert "* 375" not in text
    assert "ensure_history(" not in text
    assert "_history_inflight" not in text


def test_mdm_historical_ohlc_ingest_uses_only_ohlc_store() -> None:
    text = (SRC / "data" / "market_data_manager.py").read_text()
    start = text.index("    def ingest_historical_ohlc")
    end = text.index("    def normalize_history_row", start)
    body = text[start:end]
    assert "_ohlc" in body
    assert "_history" not in body


def test_runner_compatibility_wrappers_delegate_to_canonical_scheduler_or_sync() -> None:
    text = (SRC / "strategies" / "runner.py").read_text()
    req_start = text.index("    def _request_mdm_hydration")
    req_end = text.index("    def _hydrate_from_mdm_cache", req_start)
    req_body = text[req_start:req_end]
    assert "_schedule_runtime_history_ensure" in req_body
    assert "request_hydration" not in req_body
    hyd_start = req_end
    hyd_end = text.index("    def _emit_history_hydration_trace", hyd_start)
    hyd_body = text[hyd_start:hyd_end]
    assert "sync_history_from_mdm" in hyd_body
    assert "ingest_historical_bar" not in hyd_body
