from __future__ import annotations

import ast
from pathlib import Path
import re
import textwrap

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "nifty_scalper_bot"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _node_start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", ()) or ()
    return min([getattr(node, "lineno", 1), *[item.lineno for item in decorators]])


def _replace_class_method(path: Path, class_name: str, method_name: str, replacement: str) -> None:
    text = _read(path)
    tree = ast.parse(text)
    target = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            target = next(
                (
                    item
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == method_name
                ),
                None,
            )
            break
    if target is None or target.end_lineno is None:
        raise RuntimeError(f"Missing {class_name}.{method_name} in {path}")
    lines = text.splitlines(keepends=True)
    lines[_node_start(target) - 1 : target.end_lineno] = [textwrap.dedent(replacement).rstrip() + "\n"]
    _write(path, "".join(lines))


def _replace_top_level_function(path: Path, name: str, replacement: str) -> None:
    text = _read(path)
    tree = ast.parse(text)
    target = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        ),
        None,
    )
    if target is None or target.end_lineno is None:
        raise RuntimeError(f"Missing function {name} in {path}")
    lines = text.splitlines(keepends=True)
    lines[_node_start(target) - 1 : target.end_lineno] = [textwrap.dedent(replacement).rstrip() + "\n"]
    _write(path, "".join(lines))


def _remove_functions(path: Path, names: set[str]) -> None:
    text = _read(path)
    tree = ast.parse(text)
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    if not nodes:
        raise RuntimeError(f"No functions {sorted(names)} found in {path}")
    lines = text.splitlines(keepends=True)
    ranges = sorted(
        [(_node_start(node) - 1, int(node.end_lineno or node.lineno)) for node in nodes],
        reverse=True,
    )
    for start, end in ranges:
        del lines[start:end]
    _write(path, "".join(lines))


def _insert_method_start(path: Path, class_name: str, method_name: str, insertion: str) -> None:
    text = _read(path)
    tree = ast.parse(text)
    cls = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name),
        None,
    )
    if cls is None:
        raise RuntimeError(f"Missing class {class_name}")
    target = next(
        (
            item
            for item in cls.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name
        ),
        None,
    )
    if target is None or not target.body:
        raise RuntimeError(f"Missing method {class_name}.{method_name}")
    body_index = 0
    if (
        isinstance(target.body[0], ast.Expr)
        and isinstance(target.body[0].value, ast.Constant)
        and isinstance(target.body[0].value.value, str)
    ):
        body_index = 1
    if body_index < len(target.body):
        insert_line = target.body[body_index].lineno - 1
    else:
        insert_line = int(target.end_lineno or target.lineno)
    lines = text.splitlines(keepends=True)
    lines[insert_line:insert_line] = [textwrap.dedent(insertion).rstrip() + "\n"]
    _write(path, "".join(lines))


def _replace_required(text: str, old: str, new: str, *, label: str, count: int | None = None) -> str:
    hits = text.count(old)
    if hits == 0:
        raise RuntimeError(f"Missing replacement target: {label}")
    if count is not None and hits != count:
        raise RuntimeError(f"Unexpected replacement count for {label}: {hits} != {count}")
    return text.replace(old, new)


def patch_mdm() -> None:
    path = SRC / "data" / "market_data_manager.py"
    _replace_class_method(
        path,
        "MarketDataManager",
        "history_capacity_for",
        '''
        def history_capacity_for(self, symbol: str, interval: str = "minute") -> int:
            """Return the authoritative CandleEngine retention for ``symbol``.

            Raw tick retention is intentionally unrelated. Hydration targets are
            bounded by CandleEngine because it is the sole finalized-OHLC SSOT.
            """
            del interval
            try:
                normalized = self._canonical_symbol(symbol)
                engine = self.get_candle_engine(normalized)
                return max(1, int(getattr(engine, "max_bars", 0) or 0))
            except Exception:
                return 0
        ''',
    )
    _replace_class_method(
        path,
        "MarketDataManager",
        "get_ohlc_bars",
        '''
        def get_ohlc_bars(
            self, symbol: str, *, limit: int | None = None
        ) -> list[dict[str, Any]]:
            """Return canonical finalized OHLC directly from CandleEngine.

            ``_ohlc`` is a compatibility/diagnostic projection only. Production
            readiness, hydration and strategy consumers must remain correct even
            when that projection is stale or absent.
            """
            if limit is not None and limit < 0:
                raise ValueError("limit must be non-negative or None")
            if limit == 0:
                return []
            normalized = self._canonical_symbol(symbol)
            try:
                engine = self.get_candle_engine(normalized)
                completed = list(engine.get_completed_bars() or [])
            except Exception:
                completed = []
            if completed:
                bars: list[dict[str, Any]] = []
                for raw in completed:
                    if not isinstance(raw, Mapping):
                        continue
                    row = dict(raw)
                    if bool(row.get("provisional") or row.get("is_provisional")):
                        continue
                    if row.get("is_complete") is False:
                        continue
                    row.setdefault("symbol", normalized)
                    row.setdefault("source", "candle_engine")
                    bars.append(row)
                return bars[-limit:] if limit is not None else bars

            # Compatibility only for deliberately partial legacy/test objects
            # that never passed through the production constructor. A normally
            # initialized runtime never falls back to the projection.
            fully_initialized = hasattr(self, "_raw_tick_history") and hasattr(
                self, "_settings"
            )
            if fully_initialized:
                return []
            projection = getattr(self, "_ohlc", {}) or {}
            key = self._bar_symbol_key(normalized)
            rows = [dict(row) for row in projection.get(key, ()) or ()]
            return rows[-limit:] if limit is not None else rows
        ''',
    )
    _replace_class_method(
        path,
        "MarketDataManager",
        "is_ohlc_ready",
        '''
        def is_ohlc_ready(self, symbol: str, required_bars: int | None = None) -> bool:
            """Return readiness from authoritative finalized CandleEngine bars."""
            normalized = normalize_symbol(str(symbol or ""))
            needed = max(
                1,
                int(
                    required_bars
                    if required_bars is not None
                    else self._min_required_bars
                ),
            )
            try:
                engine = self.get_candle_engine(normalized)
                return len(engine.get_completed_bars() or []) >= needed
            except Exception:
                # Partial compatibility objects used by diagnostics/tests may not
                # have a usable engine registry. They are not production owners.
                if hasattr(self, "_raw_tick_history") and hasattr(self, "_settings"):
                    return False
                projection = getattr(self, "_ohlc", {}) or {}
                return len(projection.get(self._bar_symbol_key(normalized), ())) >= needed
        ''',
    )
    text = _read(path)
    old = "            after = self._refresh_candle_projection(normalized_symbol)\n            self.update_hydration_status(normalized_symbol, after)"
    new = "            self._refresh_candle_projection(normalized_symbol)\n            after = self.get_ohlc_bars(normalized_symbol)\n            self.update_hydration_status(normalized_symbol, after)"
    text = _replace_required(text, old, new, label="canonical post-import history", count=1)
    old_snapshot = "            bars = list(self._ohlc.get(self._bar_symbol_key(canonical), ()))"
    if old_snapshot in text:
        text = text.replace(old_snapshot, "            bars = self.get_ohlc_bars(canonical, limit=1)")
    text = text.replace(
        "- Performs broker history fetch/backfill (ensure_history) and is the sole holder\n  of the OHLC bar cache used for readiness and indicators.",
        "- Performs broker history fetch/backfill (ensure_history) into the per-symbol\n  CandleEngine registry used as the finalized-OHLC SSOT.",
    )
    _write(path, text)


def patch_history_policy() -> None:
    path = SRC / "core" / "history_readiness.py"
    _replace_top_level_function(
        path,
        "resolve_history_policy",
        '''
        def resolve_history_policy(
            ctx: "BotContext",
            symbol: str,
            *,
            role: str,
            phase: str,
            reason: str,
        ) -> HistoryPolicyDecision:
            """Resolve role-aware history policy without changing execution minima."""
            runner = getattr(ctx, "strategy_runner", None)
            role = str(role or "option_context")
            phase = str(phase or "dynamic_update")
            option_min = int(
                os.getenv(
                    "READINESS_OPTION_EXEC_MIN_BARS",
                    os.getenv("OPTION_EXECUTION_MIN_BARS", str(_DEFAULT_OPT_MIN_BARS)),
                )
                or _DEFAULT_OPT_MIN_BARS
            )
            context_env = int(
                os.getenv(
                    "READINESS_CONTEXT_MIN_BARS",
                    os.getenv("CONTEXT_EXECUTION_MIN_BARS", "20"),
                )
                or 20
            )
            context_min = max(
                context_env, int(getattr(runner, "_context_required_bars", 0) or 0)
            )
            from nifty_scalper_bot.core.app import _symbol_history_requirement

            generic_required = _symbol_history_requirement(ctx)
            if role == "selected_option":
                required = max(
                    option_min, int(getattr(runner, "_option_required_bars", 0) or 0)
                )
                target = max(required, generic_required)
                priority = 10
            elif role in {"spot_context", "futures_context"}:
                required = max(context_min, 1)
                target = max(required, generic_required)
                priority = 5 if role == "spot_context" else 4
            elif role == "recovery_or_open_position":
                required = max(option_min, generic_required)
                target = required
                priority = 9
            else:
                required = max(option_min, 1)
                target = max(required, generic_required)
                priority = 1

            market_closed_context = (
                role == "option_context"
                and get_runtime_market_mode()
                in {"PRE_MARKET", "POST_MARKET", "HOLIDAY"}
                and phase != "recovery"
            )
            role_caps = {
                "selected_option": int(os.getenv("HYDRATION_CAP_SELECTED_OPTION", "75") or 75),
                "option_context": int(os.getenv("HYDRATION_CAP_OPTION_CONTEXT", "50") or 50),
                "spot_context": int(os.getenv("HYDRATION_CAP_SPOT_CONTEXT", "100") or 100),
                "futures_context": int(os.getenv("HYDRATION_CAP_FUTURES_CONTEXT", "100") or 100),
                "recovery_or_open_position": int(os.getenv("HYDRATION_CAP_RECOVERY", "100") or 100),
            }
            role_cap = role_caps.get(
                role, int(os.getenv("HYDRATION_CAP_DEFAULT", "75") or 75)
            )
            deep_caps = {
                "selected_option": int(os.getenv("HYDRATION_DEEP_SELECTED_OPTION", "300") or 300),
                "option_context": int(os.getenv("HYDRATION_DEEP_OPTION_CONTEXT", str(role_cap)) or role_cap),
                "spot_context": int(os.getenv("HYDRATION_DEEP_SPOT_CONTEXT", "300") or 300),
                "futures_context": int(os.getenv("HYDRATION_DEEP_FUTURES_CONTEXT", "300") or 300),
                "recovery_or_open_position": int(os.getenv("HYDRATION_DEEP_RECOVERY", "300") or 300),
            }
            deep_cap = max(role_cap, deep_caps.get(role, role_cap))

            orb_enabled = str(os.getenv("ORB_ENABLED", "true") or "true").strip().lower() in {
                "1", "true", "yes", "y", "on", "enable", "enabled"
            }
            if role in {"spot_context", "futures_context"} and orb_enabled:
                structural_target = 400
                role_cap = max(role_cap, structural_target)
                deep_cap = max(deep_cap, structural_target)
                if phase == "startup" and reason == "startup_hydration":
                    target = max(target, structural_target)

            safety_max = int(os.getenv("HYDRATION_MAX_BARS", "0") or 0)
            if safety_max > 0:
                role_cap = min(role_cap, safety_max)
                deep_cap = min(deep_cap, safety_max)
            target = max(required, min(target, role_cap))
            return HistoryPolicyDecision(
                role=role,
                phase=phase,
                required_bars=required,
                target_bars=target,
                allow_broker_fetch=not market_closed_context,
                sync_runner=True,
                priority=priority,
                minimum_only=False,
                role_cap=role_cap,
                deep_cap=deep_cap,
            )
        ''',
    )


def patch_runner() -> None:
    path = SRC / "strategies" / "runner.py"
    text = _read(path)
    marker = "MIN_EVAL_INTERVAL_SECONDS = 5.0\n"
    addition = (
        "MIN_EVAL_INTERVAL_SECONDS = 5.0\n"
        "_CONTEXT_SESSION_HISTORY_BARS = 400\n"
        "_CONTEXT_HISTORY_PROBE_INTERVAL_SECONDS = 5.0\n"
        "_CONTEXT_TARGET_REQUEST_INTERVAL_SECONDS = 30.0\n"
    )
    text = _replace_required(text, marker, addition, label="runner context constants", count=1)
    _write(path, text)

    _replace_class_method(
        path,
        "StrategyRunner",
        "_get_mdm_bars",
        '''
        def _get_mdm_bars(self, symbol: str, limit: int) -> list[dict[str, Any]]:
            """Read canonical cached bars; broaden only underlying context windows."""
            normalized = self._normalize_symbol(symbol)
            resolved_limit = max(1, int(limit or 1))
            role = self._history_role_for_symbol(normalized)
            if role in {"spot_context", "futures_context"}:
                try:
                    smc_min = max(
                        1, int(HistoryReadinessPolicy.from_env().smc_min_bars)
                    )
                except Exception:
                    smc_min = safe_positive_int_env(
                        "SMC_MIN_BARS_REQUIRED", 30, minimum=1
                    )
                resolved_limit = max(resolved_limit, smc_min)
                if _env_bool("ORB_ENABLED", True):
                    resolved_limit = max(
                        resolved_limit, _CONTEXT_SESSION_HISTORY_BARS
                    )

            for source in (self._market_data, self._data_hub):
                if source is None:
                    continue
                for name in ("get_ohlc_bars", "get_ohlc", "get_recent_bars"):
                    fn = getattr(source, name, None)
                    if not callable(fn):
                        continue
                    try:
                        try:
                            bars = fn(normalized, limit=resolved_limit)
                        except TypeError:
                            bars = fn(normalized)
                        if bars:
                            return [dict(row) for row in list(bars)[-resolved_limit:]]
                    except Exception:
                        continue
            return []
        ''',
    )
    _replace_class_method(
        path,
        "StrategyRunner",
        "_sync_context_history_if_cold",
        '''
        def _sync_context_history_if_cold(
            self, *, source: str = "context_history_sync"
        ) -> None:
            """Keep spot/futures derived histories aligned with CandleEngine via MDM."""
            try:
                smc_min = max(1, int(HistoryReadinessPolicy.from_env().smc_min_bars))
            except Exception:
                smc_min = safe_positive_int_env("SMC_MIN_BARS_REQUIRED", 30, minimum=1)
            minimum = max(1, int(self._context_required_bars or 1), smc_min)
            target = minimum
            if _env_bool("ORB_ENABLED", True):
                target = max(target, _CONTEXT_SESSION_HISTORY_BARS)

            request_state = getattr(self, "_context_structural_request_at", None)
            if not isinstance(request_state, dict):
                request_state = {}
                self._context_structural_request_at = request_state

            for ctx_symbol in self._active_context_symbols_for_history():
                normalized = self._normalize_symbol(ctx_symbol)
                role = self._history_role_for_symbol(normalized)
                if role not in {"spot_context", "futures_context"}:
                    continue
                try:
                    mdm_rows = self._get_mdm_bars(normalized, target)
                    source_count = len(mdm_rows)
                    indicator_count = self._history_count_for_symbol(normalized)
                    runner_count = len(
                        getattr(self, "_symbol_history", {}).get(normalized, []) or []
                    )
                    mdm_last = (
                        self._history_row_timestamp(mdm_rows[-1]) if mdm_rows else None
                    )
                    runner_last = self._runner_history_last_timestamp(normalized)
                    indicator_last = self._indicator_history_last_timestamp(normalized)
                    stale_projection = bool(
                        mdm_last is not None
                        and (
                            runner_last is None
                            or indicator_last is None
                            or runner_last < mdm_last
                            or indicator_last < mdm_last
                        )
                    )
                    needs_sync = bool(
                        indicator_count < minimum
                        or runner_count < minimum
                        or stale_projection
                    )
                    after = indicator_count
                    if needs_sync:
                        after = self._sync_history_from_mdm_cache(
                            normalized,
                            required_bars=minimum,
                            source=source,
                            request_if_short=source_count < minimum,
                        )

                    if source_count < target:
                        now = time.monotonic()
                        last_request = float(request_state.get(normalized, 0.0) or 0.0)
                        if now - last_request >= _CONTEXT_TARGET_REQUEST_INTERVAL_SECONDS:
                            if self._schedule_runtime_history_ensure(
                                normalized,
                                role=role,
                                phase="runner_sync",
                                reason=source,
                                required_bars=minimum,
                                target_bars=target,
                            ):
                                request_state[normalized] = now

                    cold_passes = getattr(self, "_context_cold_passes", None)
                    if cold_passes is None:
                        cold_passes = {}
                        self._context_cold_passes = cold_passes
                    if after < minimum:
                        cold_passes[normalized] = cold_passes.get(normalized, 0) + 1
                        grace = safe_positive_int_env(
                            "CONTEXT_HISTORY_COLD_GRACE_PASSES", 3, minimum=1
                        )
                        if cold_passes[normalized] <= grace:
                            log_throttled(
                                self._logger,
                                f"context_history_pending:{normalized}",
                                "CONTEXT_HISTORY_HYDRATION_PENDING symbol=%s source=%s have=%d need=%d pass=%d",
                                normalized,
                                source,
                                after,
                                minimum,
                                cold_passes[normalized],
                                interval_sec=10.0,
                                extra={
                                    "event": "CONTEXT_HISTORY_HYDRATION_PENDING",
                                    "symbol": normalized,
                                    "source": source,
                                    "indicator_history_count": after,
                                    "required_bars": minimum,
                                    "cold_pass": cold_passes[normalized],
                                },
                            )
                        else:
                            self._logger.warning(
                                "CONTEXT_HISTORY_HYDRATION_FAILED symbol=%s source=%s error=%s",
                                normalized,
                                source,
                                "insufficient_canonical_bars",
                                extra={
                                    "event": "CONTEXT_HISTORY_HYDRATION_FAILED",
                                    "symbol": normalized,
                                    "source": source,
                                    "error": "insufficient_canonical_bars",
                                    "indicator_history_count": after,
                                    "required_bars": minimum,
                                    "cold_pass": cold_passes[normalized],
                                },
                            )
                    else:
                        cold_passes.pop(normalized, None)
                except Exception as exc:  # noqa: BLE001 - fail closed with diagnostics
                    self._logger.warning(
                        "CONTEXT_HISTORY_HYDRATION_FAILED symbol=%s source=%s error=%s",
                        normalized,
                        source,
                        exc,
                        extra={
                            "event": "CONTEXT_HISTORY_HYDRATION_FAILED",
                            "symbol": normalized,
                            "source": source,
                            "error": str(exc),
                        },
                    )
        ''',
    )
    _insert_method_start(
        path,
        "StrategyRunner",
        "_on_tick",
        '''
        try:
            _context_symbol = self._normalize_symbol(symbol)
            if self._history_role_for_symbol(_context_symbol) in {
                "spot_context",
                "futures_context",
            }:
                _probe_state = getattr(self, "_context_history_probe_at", None)
                if not isinstance(_probe_state, dict):
                    _probe_state = {}
                    self._context_history_probe_at = _probe_state
                _probe_now = time.monotonic()
                _last_probe = float(_probe_state.get(_context_symbol, 0.0) or 0.0)
                if (
                    _probe_now - _last_probe
                    >= _CONTEXT_HISTORY_PROBE_INTERVAL_SECONDS
                ):
                    _probe_state[_context_symbol] = _probe_now
                    self._sync_context_history_if_cold(
                        source="context_tick_bar_sync"
                    )
        except Exception:
            pass
        ''',
    )


def patch_strategy_context_imports() -> None:
    signal_path = SRC / "strategies" / "signal_generator.py"
    _remove_functions(signal_path, {"build_strategy_history_context"})
    text = _read(signal_path)
    anchor = "from nifty_scalper_bot.core.signal_arbitrator import SignalArbitrator\n"
    text = _replace_required(
        text,
        anchor,
        anchor
        + "from nifty_scalper_bot.core.strategy_context_builder import build_strategy_history_context\n",
        label="canonical strategy context import",
        count=1,
    )
    _write(signal_path, text)

    manager_path = SRC / "core" / "strategy_manager.py"
    text = _read(manager_path)
    old = (
        "from nifty_scalper_bot.strategies.signal_generator import (\n"
        "    Signal,\n"
        "    build_strategy_history_context,\n"
        ")\n"
    )
    new = (
        "from nifty_scalper_bot.core.strategy_context_builder import (\n"
        "    build_strategy_history_context,\n"
        ")\n"
        "from nifty_scalper_bot.strategies.signal_generator import Signal\n"
    )
    text = _replace_required(text, old, new, label="StrategyManager canonical builder import", count=1)
    _write(manager_path, text)

    safety_path = SRC / "core" / "strategy_live_safety.py"
    _remove_functions(safety_path, {"_install_canonical_history_builder"})
    text = _read(safety_path)
    text = _replace_required(
        text,
        "    _install_canonical_history_builder(strategy_module)\n",
        "",
        label="runtime strategy builder monkeypatch",
        count=1,
    )
    _write(safety_path, text)


def patch_runtime_hardening() -> None:
    path = SRC / "core" / "runtime_history_event_loop_hardening.py"
    text = _read(path)
    text = re.sub(
        r"from nifty_scalper_bot\.data\.ohlc_capacity_contract import \(\n\s*install_mdm_ohlc_capacity_contract,\n\)\n",
        "",
        text,
        count=1,
    )
    if "install_mdm_ohlc_capacity_contract" in text:
        text = text.replace("    install_mdm_ohlc_capacity_contract()\n", "")
    text = text.replace(
        '"""Defer only cold far-context history and preserve canonical ownership."""',
        '"""Defer only cold far-context history; CandleEngine ownership stays native."""',
    )
    text = text.replace(
        '"""Defer cold far-context history and install completed-OHLC capacity."""',
        '"""Defer cold far-context history; CandleEngine ownership stays native."""',
    )
    if "ohlc_capacity_contract" in text or "install_mdm_ohlc_capacity_contract" in text:
        raise RuntimeError("OHLC capacity runtime adapter reference remains")
    _write(path, text)


def patch_dynamic_safety() -> None:
    path = SRC / "core" / "strategy_runner_dynamic_universe_safety.py"
    _remove_functions(path, {"_env_enabled", "_context_history_read_limit", "get_mdm_bars"})
    text = _read(path)
    text = re.sub(
        r"from nifty_scalper_bot\.execution\.readiness import HistoryReadinessPolicy\n",
        "",
        text,
        count=1,
    )
    text = text.replace("_CONTEXT_SESSION_HISTORY_BARS = 400\n", "")
    text = re.sub(r"_TRUE_VALUES = \{[^\n]+\}\n", "", text, count=1)
    text = re.sub(
        r"\s+from nifty_scalper_bot\.core\.context_history_continuity import \(\n\s+apply_patches as _apply_context_history_continuity,\n\s+\)\n",
        "\n",
        text,
        count=1,
    )
    text = text.replace("    _apply_context_history_continuity(app_module)\n", "")
    text = text.replace("    original_get_mdm_bars = StrategyRunner._get_mdm_bars\n", "")
    text = text.replace(
        "    StrategyRunner._dynamic_universe_safety_original_get_mdm_bars = original_get_mdm_bars\n",
        "",
    )
    text = text.replace("    StrategyRunner._get_mdm_bars = get_mdm_bars\n", "")
    text = text.replace('    "_context_history_read_limit",\n', "")
    if "context_history_continuity" in text or "_context_history_read_limit" in text:
        raise RuntimeError("Context history monkeypatch reference remains")
    _write(path, text)


def remove_obsolete_files() -> None:
    for relative in (
        "src/nifty_scalper_bot/data/ohlc_capacity_contract.py",
        "src/nifty_scalper_bot/core/context_history_continuity.py",
        "tests/data/test_ohlc_capacity_contract.py",
    ):
        path = ROOT / relative
        if path.exists():
            path.unlink()


def validate_source() -> None:
    for path in SRC.rglob("*.py"):
        ast.parse(_read(path))
    corpus = "\n".join(_read(path) for path in SRC.rglob("*.py"))
    if "MDM_OHLC_CACHE_LEN" in corpus:
        raise RuntimeError("MDM_OHLC_CACHE_LEN remains in production source")
    if "context_history_continuity" in corpus:
        raise RuntimeError("context_history_continuity remains in production source")
    if "ohlc_capacity_contract" in corpus:
        raise RuntimeError("ohlc_capacity_contract remains in production source")


if __name__ == "__main__":
    patch_mdm()
    patch_history_policy()
    patch_runner()
    patch_strategy_context_imports()
    patch_runtime_hardening()
    patch_dynamic_safety()
    remove_obsolete_files()
    validate_source()
