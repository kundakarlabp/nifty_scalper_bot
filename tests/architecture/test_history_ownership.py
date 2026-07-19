"""Architecture guards for canonical history-hydration ownership (spec §13).

These tests are written as ``async def`` deliberately. The repository conftest's
``pytest_pyfunc_call`` hook executes coroutine test bodies but currently no-ops
plain sync test bodies; writing the guards as coroutines guarantees their
assertions actually run instead of silently passing.

Broker historical-data access is allow-listed to the final architecture: the
canonical owner (MarketDataManager) plus the broker adapter that defines the
raw client method. The allowlist is intentionally explicit so a NEW unexpected
caller fails the guard.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "nifty_scalper_bot"

BROKER_HISTORY_ALLOWLIST = {
    "src/nifty_scalper_bot/data/market_data_manager.py",
    "src/nifty_scalper_bot/data/rest/zerodha_client.py",
}

HYDRATION_FORBIDDEN_DIRS = ("execution", "notifications")
HYDRATION_OWNERSHIP_NAMES = {
    "hydrate_symbol_history",
    "ensure_history",
    "reseed_history_from_bars",
    "ingest_historical_bar",
    "replace_history",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _call_names(path: Path, names: set[str]) -> list[str]:
    tree = ast.parse(_read_text(path))
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
    text = _read_text(SRC / "core" / "app.py")
    assert "ctx.market_data_manager.hydrate_symbol_history" not in text
    assert "ctx.market_data_manager.fetch_history" not in text
    assert "_indicator_engine.replace_history" not in text


async def test_datahub_has_no_authoritative_history_cache() -> None:
    text = _read_text(SRC / "data" / "data_hub.py")
    assert "_history_cache" not in text
    assert "def _normalize_history_rows" not in text


async def test_runner_does_not_call_broker_history() -> None:
    text = _read_text(SRC / "strategies" / "runner.py")
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
    # Readiness computation was extracted from app.py to core/history_readiness.py.
    text = _read_text(SRC / "core" / "history_readiness.py")
    tree = ast.parse(text)
    target = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "compute_selected_option_history_readiness"
        ):
            target = node
            break
    assert target is not None, "compute_selected_option_history_readiness missing"
    body_calls = _call_names_in_node(
        target,
        {
            "ensure_history",
            "historical_data",
            "reseed_history_from_bars",
            "replace_history",
            "ingest_historical_bar",
        },
    )
    assert body_calls == [], f"Canonical readiness must be pure, found: {body_calls}"


def _func_source(path: Path, func_name: str) -> str:
    text = _read_text(path)
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == func_name
        ):
            return ast.get_source_segment(text, node) or ""
    return ""


def _function_call_attrs(path: Path) -> dict[str, set[str]]:
    text = _read_text(path)
    tree = ast.parse(text)
    result: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            calls: set[str] = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    if isinstance(fn, ast.Attribute):
                        calls.add(fn.attr)
                    elif isinstance(fn, ast.Name):
                        calls.add(fn.id)
            result[node.name] = calls
    return result


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(_read_text(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.rsplit(".", 1)[-1])
    return names


async def test_runner_prewarm_has_no_datahub_hydrate_or_reseed() -> None:
    # Spec §3/§13: production prewarm must not call DataHub hydrate, reseed, or
    # ingest historical bars.
    src = _func_source(
        SRC / "strategies" / "runner.py", "_request_selected_option_history_prewarm"
    )
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
    text = _read_text(SRC / "core" / "app.py")
    assert "set_runtime_history_ensurer" in text
    assert "ensure_symbol_runtime_history" in text


async def test_runner_declares_ensurer_field() -> None:
    text = _read_text(SRC / "strategies" / "runner.py")
    assert "self._runtime_history_ensurer" in text
    assert "def set_runtime_history_ensurer" in text


async def test_runner_makes_no_direct_request_hydration_call() -> None:
    # Spec §9: Runner must not call request_hydration on DataHub or MDM. We scan
    # for an ast.Call whose attr is 'request_hydration'.
    src = _read_text(SRC / "strategies" / "runner.py")
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


async def test_runtime_history_ensurer_callback_contract_is_explicit() -> None:
    text = _read_text(SRC / "core" / "app.py")
    tree = ast.parse(text)
    funcs = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_runtime_history_ensurer"
    ]
    assert len(funcs) == 1
    fn = funcs[0]
    assert fn.args.vararg is None
    assert (
        fn.args.kwarg is None
    ), "production runtime history ensurer must not accept **kwargs"
    assert [a.arg for a in fn.args.args] == ["symbol"]
    assert [a.arg for a in fn.args.kwonlyargs] == [
        "role",
        "phase",
        "reason",
        "required_bars",
        "target_bars",
    ]


async def test_only_one_production_history_ensurer_injection_point() -> None:
    text = _read_text(SRC / "core" / "app.py")
    assert text.count("set_runtime_history_ensurer(") == 1
    assert "CANONICAL_HISTORY_ENSURER_INJECTION_FAILED" in text
    assert (
        "LOGGER.error"
        in text[text.index("CANONICAL_HISTORY_ENSURER_INJECTION_FAILED") - 400 :]
    )


async def test_mdm_readiness_concepts_are_explicit_and_separate() -> None:
    text = _read_text(SRC / "data" / "market_data_manager.py")
    assert "def is_tick_ready" in text
    assert "def is_ohlc_ready" in text
    assert "def is_market_data_ready" in text
    wait_src = _func_source(SRC / "data" / "market_data_manager.py", "wait_until_ready")
    assert "len(self._ohlc" not in wait_src
    assert "len(self._raw_tick_history" in wait_src


async def test_historical_bar_ingestion_does_not_write_raw_tick_history() -> None:
    src = _func_source(SRC / "data" / "market_data_manager.py", "ingest_historical_bar")
    assert "_ohlc" in src
    assert "_raw_tick_history[symbol].append" not in src


async def test_data_source_no_longer_calls_broker_historical_data() -> None:
    text = _read_text(SRC / "data" / "source.py")
    assert ".historical_data(" not in text
    assert "MarketDataManager.ensure_history" in text


async def test_market_data_source_get_ohlc_has_no_production_callers() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel == "src/nifty_scalper_bot/data/source.py":
            continue
        tree = ast.parse(_read_text(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get_ohlc"
            ):
                offenders.append((rel, node.lineno))
    assert (
        offenders == []
    ), f"Production get_ohlc callers must use MDM.get_ohlc_bars/cache helpers: {offenders}"


async def test_no_cache_miss_then_direct_broker_history_fetch_pattern() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(_read_text(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                body_calls = _call_names_in_node(
                    node, {"historical_data", "get_historical_data"}
                )
                if body_calls:
                    offenders.append((rel, node.lineno, body_calls))
    assert (
        offenders == []
    ), f"Cache-miss fallbacks must not fetch broker history directly: {offenders}"


async def test_datahub_has_no_runtime_history_ownership_or_readiness_decisions() -> (
    None
):
    text = _read_text(SRC / "data" / "data_hub.py")
    assert "_history_cache" not in text
    assert "def _normalize_history_rows" not in text
    assert "historical_data" not in text
    assert "get_historical_data" not in text
    assert "compute_history_readiness" not in text
    assert "is_symbol_ready(" not in text.replace("def is_symbol_ready(", "")
    for func in ("get_ohlc", "fetch_history", "hydrate_symbol_history"):
        src = _func_source(SRC / "data" / "data_hub.py", func)
        if src:
            assert "historical_data" not in src
            assert "_history_cache" not in src
            assert "normalize_history" not in src


async def test_datahub_get_ohlc_accepts_positional_limit() -> None:
    from nifty_scalper_bot.data.data_hub import DataHub

    class _Mdm:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int | None]] = []

        def get_ohlc_bars(self, symbol: str, *, limit: int | None = None):
            self.calls.append((symbol, limit))
            return (
                [{"close": 1}, {"close": 2}][-limit if limit else 0 :]
                if limit
                else [{"close": 1}, {"close": 2}]
            )

    mdm = _Mdm()
    hub = DataHub(market_data_manager=mdm)

    rows = hub.get_ohlc("NSE:NIFTY", 1)

    assert rows == [{"close": 2}]
    assert mdm.calls == [("NSE:NIFTY", 1)]


async def test_runner_has_no_direct_mdm_or_datahub_hydration_ownership_calls() -> None:
    text = _read_text(SRC / "strategies" / "runner.py")
    assert ".ensure_history(" not in text
    assert ".fetch_history(" not in text
    assert ".hydrate_symbol_history(" not in text
    assert ".historical_data(" not in text
    assert ".request_hydration(" not in text


async def test_app_injection_failure_marks_health_and_live_state() -> None:
    text = _read_text(SRC / "core" / "app.py")
    failure_idx = text.index("CANONICAL_HISTORY_ENSURER_INJECTION_FAILED")
    failure_block = text[failure_idx - 2000 : failure_idx + 1000]
    assert "canonical_history_ensurer_injection_failed" in failure_block
    assert "if safe_order_manager is not None" in failure_block
    assert "safe_order_manager.set_live_enabled(False)" in failure_block
    assert "live_block_reason" in failure_block
    checker_src = _func_source(
        SRC / "core" / "app.py", "_check_canonical_history_ensurer"
    )
    assert checker_src
    assert "canonical_history_ensurer_injection_failed" in checker_src
    assert "return False" in checker_src


async def test_mdm_storage_names_remain_separate() -> None:
    text = _read_text(SRC / "data" / "market_data_manager.py")
    assert "self._ohlc" in text
    assert "self._raw_tick_history" in text
    ingest_src = _func_source(
        SRC / "data" / "market_data_manager.py", "ingest_historical_bar"
    )
    assert "_ohlc" in ingest_src
    assert (
        "_raw_tick_history[" not in ingest_src
        and "_raw_tick_history.setdefault" not in ingest_src
    )


async def test_production_code_uses_explicit_readiness_apis() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(_read_text(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "is_symbol_ready"
            ):
                offenders.append((rel, node.lineno))
    assert (
        offenders == []
    ), f"Use is_tick_ready/is_ohlc_ready/is_market_data_ready instead: {offenders}"
    mdm_src = _func_source(SRC / "data" / "market_data_manager.py", "is_symbol_ready")
    if mdm_src:
        assert "is_tick_ready" in mdm_src
        assert "is_ohlc_ready" not in mdm_src


async def test_mdm_contract_ssot_regression_guards() -> None:
    path = SRC / "data" / "market_data_manager.py"
    text = _read_text(path)
    imported = _imported_names(path)
    assert "NIFTY_SPOT_TOKEN" not in text
    assert "256265" not in text
    assert "260105" not in text
    assert "resolve_active_nifty_future" not in imported
    assert "resolve_active_nifty_future_from_instruments" not in imported
    assert '.instruments("NFO")' not in text
    assert "broker_instruments" not in text
    assert "CRITICAL FALLBACK" not in text


async def test_mdm_broker_history_calls_stay_inside_canonical_fetch_helper() -> None:
    calls_by_func = _function_call_attrs(SRC / "data" / "market_data_manager.py")
    offenders = {
        name: sorted(calls & {"historical_data", "get_historical_data"})
        for name, calls in calls_by_func.items()
        if name != "fetch_history"
        and calls & {"historical_data", "get_historical_data"}
    }
    assert offenders == {}


async def test_mdm_warmup_history_delegates_to_ensure_history() -> None:
    src = _func_source(SRC / "data" / "market_data_manager.py", "warmup_history")
    assert "ensure_history" in src
    assert "historical_data" not in src
    assert "get_historical_data" not in src
    assert "ingest_historical_bar" not in src


async def test_mdm_basket_replaces_desired_tokens_and_uses_mapping_helper() -> None:
    text = _read_text(SRC / "data" / "market_data_manager.py")
    set_basket = _func_source(
        SRC / "data" / "market_data_manager.py", "set_active_contract_basket"
    )
    assert "_desired_tokens.update(tokens)" not in set_basket
    assert "_replace_desired_tokens_from_basket" in set_basket
    assert "_set_symbol_token_mapping" in text
    assert "seed_symbol_tokens" in text


async def test_mdm_readiness_requires_tradable_quote_and_canonical_bars() -> None:
    src = _func_source(
        SRC / "data" / "market_data_manager.py", "hydrate_active_contract_basket"
    )
    assert "tradable_quote_ready" in src
    assert "tradable_quote_not_ready" in src
    assert "_selected_option_history_required_bars" in src
    assert '"ohlc_ready": False' in src


async def test_mdm_poll_quote_does_not_fabricate_completed_candles() -> None:
    text = _read_text(SRC / "data" / "market_data_manager.py")
    src = _func_source(SRC / "data" / "market_data_manager.py", "ingest_rest_quote")
    assert "_process_poll_quote" not in text
    assert "_emit_poll_candle" not in text
    assert "_last_cumulative_volume_by_symbol" not in src
    assert "ingest_historical_ohlc" not in src


async def test_runtime_execution_path_has_no_forbidden_router_imports() -> None:
    forbidden = {
        "execution_router",
        "order_execution_hub",
        "preflight_validator",
        "ExecutionRouter",
        "OrderExecutionHub",
        "PreflightValidator",
    }
    offenders = []
    for rel in [
        "core/app.py",
        "strategies/runner.py",
        "execution/order_manager.py",
        "execution/bracket_manager.py",
    ]:
        text = _read_text(SRC / rel)
        hits = sorted(item for item in forbidden if item in text)
        if hits:
            offenders.append((rel, hits))
    assert offenders == []
