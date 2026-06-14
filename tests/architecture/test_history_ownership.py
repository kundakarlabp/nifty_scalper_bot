"""Architecture guards for canonical history-hydration ownership (spec §13).

These tests are written as ``async def`` deliberately. The repository conftest's
``pytest_pyfunc_call`` hook executes coroutine test bodies but currently no-ops
plain sync test bodies; writing the guards as coroutines guarantees their
assertions actually run instead of silently passing.

Broker historical-data access is allow-listed to the real architecture: the
canonical owner (MarketDataManager) plus the broker adapter that defines the
raw client method, and the known legacy ``data/source.py`` fetch path (tracked
for removal). The allowlist is intentionally explicit so a NEW unexpected caller
fails the guard.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "nifty_scalper_bot"

BROKER_HISTORY_ALLOWLIST = {
    "src/nifty_scalper_bot/data/market_data_manager.py",
    "src/nifty_scalper_bot/data/rest/zerodha_client.py",
    "src/nifty_scalper_bot/data/source.py",
}

HYDRATION_FORBIDDEN_DIRS = ("execution", "notifications")
HYDRATION_OWNERSHIP_NAMES = {
    "hydrate_symbol_history",
    "ensure_history",
    "reseed_history_from_bars",
    "ingest_historical_bar",
    "replace_history",
}


def _call_names(path: Path, names: set[str]) -> list[str]:
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


def _call_names_in_node(node: ast.AST, names: set[str]) -> list[str]:
    found: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            if isinstance(fn, ast.Attribute) and fn.attr in names:
                found.append(fn.attr)
            elif isinstance(fn, ast.Name) and fn.id in names:
                found.append(fn.id)
    return found


async def test_only_allowlisted_files_call_broker_historical_data() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel in BROKER_HISTORY_ALLOWLIST:
            continue
        if _call_names(path, {"historical_data", "get_historical_data"}):
            offenders.append(rel)
    assert offenders == [], f"Unexpected broker-history callers: {offenders}"


async def test_app_uses_canonical_runtime_history_not_mdm_wrappers() -> None:
    text = (SRC / "core" / "app.py").read_text()
    assert "ctx.market_data_manager.hydrate_symbol_history" not in text
    assert "ctx.market_data_manager.fetch_history" not in text
    assert "_indicator_engine.replace_history" not in text


async def test_datahub_has_no_authoritative_history_cache() -> None:
    text = (SRC / "data" / "data_hub.py").read_text()
    assert "_history_cache" not in text
    assert "def _normalize_history_rows" not in text


async def test_runner_does_not_call_broker_history() -> None:
    text = (SRC / "strategies" / "runner.py").read_text()
    assert ".historical_data(" not in text
    assert ".fetch_history(" not in text


async def test_execution_and_notifications_do_not_hydrate() -> None:
    offenders = []
    for sub in HYDRATION_FORBIDDEN_DIRS:
        base = SRC / sub
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            calls = _call_names(path, HYDRATION_OWNERSHIP_NAMES)
            if calls:
                offenders.append((path.relative_to(ROOT).as_posix(), calls))
    assert offenders == [], f"Hydration ownership leaked into {offenders}"


async def test_canonical_readiness_function_exists_and_is_pure() -> None:
    text = (SRC / "core" / "app.py").read_text()
    tree = ast.parse(text)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "compute_selected_option_history_readiness":
            target = node
            break
    assert target is not None, "compute_selected_option_history_readiness missing"
    body_calls = _call_names_in_node(target, {"ensure_history", "historical_data", "reseed_history_from_bars", "replace_history", "ingest_historical_bar"})
    assert body_calls == [], f"Canonical readiness must be pure, found: {body_calls}"


def _func_source(path: Path, func_name: str) -> str:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return ast.get_source_segment(path.read_text(), node) or ""
    return ""


async def test_runner_prewarm_has_no_datahub_hydrate_or_reseed() -> None:
    # Spec §3/§13: production prewarm must not call DataHub hydrate, reseed, or
    # ingest historical bars.
    src = _func_source(SRC / "strategies" / "runner.py", "_request_selected_option_history_prewarm")
    assert src, "prewarm function not found"
    assert "hydrate_symbol_history" not in src, "prewarm must not call DataHub hydrate"
    assert "reseed_history_from_bars" not in src, "prewarm must not reseed raw rows"
    assert "ingest_historical_bar" not in src, "prewarm must not ingest rows directly"


async def test_hydrate_from_mdm_cache_is_thin_delegate() -> None:
    # Spec §2/§13: _hydrate_from_mdm_cache delegates to sync_history_from_mdm and
    # performs no manual ingestion loop.
    src = _func_source(SRC / "strategies" / "runner.py", "_hydrate_from_mdm_cache")
    assert src, "function not found"
    assert "sync_history_from_mdm" in src, "must delegate to canonical sync"
    assert "ingest_historical_bar" not in src, "must not ingest rows manually"


async def test_app_injects_runtime_history_ensurer() -> None:
    # Spec §13: app wires the canonical callback into the runner.
    text = (SRC / "core" / "app.py").read_text()
    assert "set_runtime_history_ensurer" in text
    assert "ensure_symbol_runtime_history" in text


async def test_runner_declares_ensurer_field() -> None:
    text = (SRC / "strategies" / "runner.py").read_text()
    assert "self._runtime_history_ensurer" in text
    assert "def set_runtime_history_ensurer" in text


async def test_runner_makes_no_direct_request_hydration_call() -> None:
    # Spec §9: Runner must not call request_hydration on DataHub or MDM. We scan
    # for an ast.Call whose attr is 'request_hydration'.
    src = (SRC / "strategies" / "runner.py").read_text()
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == "request_hydration":
                bad.append(node.lineno)
    assert bad == [], f"Runner calls request_hydration directly at lines {bad}"


async def test_sync_history_schedules_only_via_canonical_ensurer() -> None:
    # Spec §9: sync_history_from_mdm body must schedule through
    # _schedule_runtime_history_ensure, not request_hydration.
    src = _func_source(SRC / "strategies" / "runner.py", "sync_history_from_mdm")
    assert src, "sync_history_from_mdm not found"
    assert "request_hydration" not in src
    assert "_schedule_runtime_history_ensure" in src


async def test_request_mdm_hydration_is_thin_canonical_delegate() -> None:
    # Spec §2: if _request_mdm_hydration remains, it must delegate to the
    # canonical scheduler and not call request_hydration.
    src = _func_source(SRC / "strategies" / "runner.py", "_request_mdm_hydration")
    if src:
        assert "_schedule_runtime_history_ensure" in src
        assert "request_hydration" not in src.replace("request_hydration; routes", "")
