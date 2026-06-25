#!/usr/bin/env python3
"""Apply the staged AWS Lightsail live-execution safety changes.

This is a temporary, deterministic migration helper used by the branch workflow.
Every replacement is guarded so an upstream drift fails the workflow instead of
silently producing a partial production patch.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _path(relative: str) -> Path:
    return ROOT / relative


def _read(relative: str) -> str:
    return _path(relative).read_text(encoding="utf-8")


def _write(relative: str, text: str) -> None:
    path = _path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def replace_once(relative: str, old: str, new: str, *, sentinel: str | None = None) -> None:
    text = _read(relative)
    if sentinel and sentinel in text:
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{relative}: expected one anchor, found {count}: {old[:120]!r}")
    _write(relative, text.replace(old, new, 1))


def insert_before(relative: str, marker: str, block: str, *, sentinel: str) -> None:
    text = _read(relative)
    if sentinel in text:
        return
    count = text.count(marker)
    if count != 1:
        raise RuntimeError(f"{relative}: expected one marker, found {count}: {marker!r}")
    _write(relative, text.replace(marker, block + marker, 1))


def regex_replace_once(relative: str, pattern: str, replacement: str, *, sentinel: str | None = None) -> None:
    text = _read(relative)
    if sentinel and sentinel in text:
        return
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S | re.M)
    if count != 1:
        raise RuntimeError(f"{relative}: regex anchor did not match exactly once: {pattern[:160]!r}")
    _write(relative, updated)


def stage_quote() -> None:
    readiness = "src/nifty_scalper_bot/execution/readiness.py"
    replace_once(readiness, "from typing import Mapping\n", "from typing import Any, Mapping\n")
    insert_before(
        readiness,
        "def quote_has_tradable_bid_ask(quote: dict | object) -> bool:\n",
        r'''
@dataclass(frozen=True, slots=True)
class ExecutionQuotePolicy:
    """Single source of truth for LIVE option quote execution gates."""

    max_tick_age_ms: float = 2500.0
    max_spread_pct: float = 0.75
    min_real_ticks_last_60s: int = 1
    require_update_version: bool = True
    require_tradable_quote: bool = True
    require_depth: bool = False

    @classmethod
    def from_env(cls, *, require_depth: bool | None = None) -> "ExecutionQuotePolicy":
        def _float(name: str, fallback: float) -> float:
            raw = os.getenv(name)
            if raw is None or not str(raw).strip():
                return float(fallback)
            try:
                return max(float(str(raw).strip()), 0.0)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "INVALID_EXECUTION_QUOTE_ENV name=%s value=%r default=%s",
                    name,
                    raw,
                    fallback,
                    extra={"event": "INVALID_EXECUTION_QUOTE_ENV", "env_name": name},
                )
                return float(fallback)

        def _bool(name: str, fallback: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return fallback
            value = str(raw).strip().lower()
            if value in {"1", "true", "yes", "on"}:
                return True
            if value in {"0", "false", "no", "off"}:
                return False
            return fallback

        max_age_default = _float("LIVE_MAX_TICK_AGE_MS", 2500.0)
        max_spread_default = _float("LIVE_MAX_SPREAD_PCT", 0.75)
        resolved_depth = (
            _bool("REQUIRE_FULL_DEPTH_FOR_EXECUTION", False)
            if require_depth is None
            else bool(require_depth)
        )
        return cls(
            max_tick_age_ms=_float("LIVE_EXECUTION_MAX_TICK_AGE_MS", max_age_default),
            max_spread_pct=_float("LIVE_EXECUTION_MAX_SPREAD_PCT", max_spread_default),
            min_real_ticks_last_60s=_env_int("LIVE_EXECUTION_MIN_REAL_TICKS_60S", 1, minimum=0),
            require_update_version=_bool("LIVE_EXECUTION_REQUIRE_UPDATE_VERSION", True),
            require_tradable_quote=_bool("LIVE_EXECUTION_REQUIRE_TRADABLE_QUOTE", True),
            require_depth=resolved_depth,
        )


@dataclass(frozen=True, slots=True)
class ExecutionQuoteReadiness:
    """Canonical LIVE execution-quote verdict consumed by strategy and runner gates."""

    symbol: str
    allowed: bool
    reason: str
    bid: float | None
    ask: float | None
    spread_pct: float | None
    tick_age_ms: float | None
    quote_update_version: object | None
    real_ticks_last_60s: int | None
    real_tick_count_derived: bool
    depth_available: bool
    tradable_quote: bool
    source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_execution_quote_readiness(
    symbol: str,
    quote: dict | object | None,
    *,
    live_mode: bool,
    policy: ExecutionQuotePolicy | None = None,
) -> ExecutionQuoteReadiness:
    """Evaluate one quote identically at strategy, candidate and final entry gates.

    LIVE mode is fail-closed for missing freshness metadata. A broker timestamp is
    accepted as the update-version proof. If an explicit rolling tick count is not
    available, one fresh versioned quote is treated as one real update; higher
    configured minimums still require an explicit count.
    """

    policy = policy or ExecutionQuotePolicy.from_env()
    base = evaluate_quote_readiness(
        symbol,
        quote,
        max_spread_pct=None,
        require_fresh=False,
    )
    getter = (
        quote.get
        if isinstance(quote, dict)
        else (lambda key, default=None: getattr(quote, key, default))
        if quote is not None
        else (lambda _key, default=None: default)
    )

    tick_age_ms = _quote_float(quote or {}, "tick_age_ms")
    if tick_age_ms is None:
        tick_age_s = _quote_float(quote or {}, "tick_age_s", "age_s", "data_age_seconds")
        if tick_age_s is not None:
            tick_age_ms = max(0.0, tick_age_s * 1000.0)

    quote_update_version: object | None = None
    for key in (
        "quote_update_version",
        "update_version",
        "tick_version",
        "last_tick_ts_ms",
        "timestamp_ms",
        "last_tick_timestamp",
        "timestamp",
    ):
        candidate = getter(key, None)
        if candidate not in (None, "", 0, 0.0):
            quote_update_version = candidate
            break

    real_ticks_raw: Any = None
    real_ticks_present = False
    for key in ("real_ticks_last_60s", "tick_count_60s", "recent_real_tick_count"):
        candidate = getter(key, None)
        if candidate is not None:
            real_ticks_raw = candidate
            real_ticks_present = True
            break
    real_ticks = _safe_non_negative_int(real_ticks_raw, 0) if real_ticks_present else None
    derived_tick_count = False
    if (
        real_ticks is None
        and quote_update_version is not None
        and tick_age_ms is not None
        and tick_age_ms <= policy.max_tick_age_ms
        and base.bid_ask_available
    ):
        real_ticks = 1
        derived_tick_count = True

    explicit_tradable = getter("tradable_quote", None)
    tradable_quote = bool(base.bid_ask_available and explicit_tradable is not False)
    reason = base.reason
    if reason == "ready" and live_mode and tick_age_ms is None:
        reason = "tick_age_missing"
    elif reason == "ready" and tick_age_ms is not None and tick_age_ms > policy.max_tick_age_ms:
        reason = "quote_stale"
    elif (
        reason == "ready"
        and live_mode
        and policy.require_update_version
        and quote_update_version is None
    ):
        reason = "quote_update_version_missing"
    elif (
        reason == "ready"
        and live_mode
        and policy.min_real_ticks_last_60s > 0
        and real_ticks is None
    ):
        reason = "real_tick_count_missing"
    elif (
        reason == "ready"
        and real_ticks is not None
        and real_ticks < policy.min_real_ticks_last_60s
    ):
        reason = "insufficient_real_ticks"
    elif (
        reason == "ready"
        and base.spread_pct is not None
        and base.spread_pct > policy.max_spread_pct
    ):
        reason = "spread_too_wide"
    elif reason == "ready" and policy.require_depth and not base.depth_available:
        reason = "quote_depth_missing"
    elif reason == "ready" and policy.require_tradable_quote and not tradable_quote:
        reason = "quote_not_tradable"

    return ExecutionQuoteReadiness(
        symbol=symbol,
        allowed=reason == "ready",
        reason=reason,
        bid=base.bid,
        ask=base.ask,
        spread_pct=base.spread_pct,
        tick_age_ms=tick_age_ms,
        quote_update_version=quote_update_version,
        real_ticks_last_60s=real_ticks,
        real_tick_count_derived=derived_tick_count,
        depth_available=base.depth_available,
        tradable_quote=tradable_quote,
        source=base.source,
    )


''',
        sentinel="class ExecutionQuotePolicy:",
    )

    base = "src/nifty_scalper_bot/strategies/elite_strategies/base_elite.py"
    replace_once(base, "import inspect\n", "import inspect\nimport os\n")
    replace_once(
        base,
        "from nifty_scalper_bot.utils.logging import get_logger\n",
        "from nifty_scalper_bot.utils.logging import get_logger\nfrom nifty_scalper_bot.utils.market_hours import is_market_hours_cached\n",
    )
    insert_before(
        base,
        "            # ✅ Early exit if capital is exhausted (prevents wasted computation)\n",
        '''            execution_mode = str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper()\n            live_enabled = execution_mode == "LIVE" or str(\n                os.getenv("ENABLE_LIVE", os.getenv("ENABLE_LIVE_TRADING", "false"))\n            ).strip().lower() in {"1", "true", "yes", "on"}\n            option_symbol = str(symbol).upper().endswith(("CE", "PE"))\n            if live_enabled and option_symbol and not is_market_hours_cached():\n                self._no_vote("outside_safe_entry_window")\n                LOGGER.debug(\n                    "ELITE_SIGNAL_BLOCKED strategy=%s symbol=%s reason=outside_safe_entry_window",\n                    self.name,\n                    symbol,\n                    extra={\n                        "event": "ELITE_SIGNAL_BLOCKED",\n                        "strategy": self.name,\n                        "symbol": symbol,\n                        "reason": "outside_safe_entry_window",\n                    },\n                )\n                return None\n\n            if self._last_signal_at is not None and self._config.cooldown_seconds > 0:\n                elapsed = (datetime.now(timezone.utc) - self._last_signal_at).total_seconds()\n                if elapsed < self._config.cooldown_seconds:\n                    self._no_vote("strategy_cooldown_active")\n                    LOGGER.debug(\n                        "ELITE_SIGNAL_BLOCKED strategy=%s symbol=%s reason=strategy_cooldown_active elapsed=%.3f required=%.3f",\n                        self.name,\n                        symbol,\n                        elapsed,\n                        self._config.cooldown_seconds,\n                        extra={\n                            "event": "ELITE_SIGNAL_BLOCKED",\n                            "strategy": self.name,\n                            "symbol": symbol,\n                            "reason": "strategy_cooldown_active",\n                        },\n                    )\n                    return None\n\n''',
        sentinel="outside_safe_entry_window",
    )
    replace_once(
        base,
        '''        self._last_signal_at = elite_signal.timestamp\n        self._last_signal = elite_signal\n        self._signals_generated += 1\n\n        # 1. Consolidate all extra data into metadata\n''',
        '''        signal_role = str(elite_signal.metadata.get("role") or "").strip().lower()\n        context_only = signal_role == "context" and not bool(\n            elite_signal.metadata.get("can_trigger")\n            or elite_signal.metadata.get("trigger_conditions_met")\n            or elite_signal.metadata.get("trigger_eligible")\n        )\n        if not context_only:\n            self._last_signal_at = elite_signal.timestamp\n        self._last_signal = elite_signal\n        self._signals_generated += 1\n\n        # 1. Consolidate all extra data into metadata\n''',
        sentinel="context_only = signal_role",
    )

    order_flow = "src/nifty_scalper_bot/strategies/elite_strategies/order_flow.py"
    replace_once(order_flow, "import os\n", "import os\nimport time\n")
    replace_once(
        order_flow,
        "from nifty_scalper_bot.strategies.signal_quality import resolve_signal_domain\n",
        "from nifty_scalper_bot.execution.readiness import (\n    ExecutionQuotePolicy,\n    evaluate_execution_quote_readiness,\n)\nfrom nifty_scalper_bot.strategies.signal_quality import resolve_signal_domain\n",
    )
    replace_once(
        order_flow,
        "        self._cfg = config\n",
        "        self._cfg = config\n        self._reversal_confirmation: dict[str, dict[str, Any]] = {}\n",
        sentinel="self._reversal_confirmation",
    )
    insert_before(
        order_flow,
        "    def _evaluate_signal(self, symbol: str, indicators: dict[str, Any], current_price: float, position: Any | None = None) -> EliteSignal | None:\n",
        '''    def _reversal_persistence_confirmed(\n        self,\n        *,\n        symbol: str,\n        side: str,\n        quote_update_version: object | None,\n    ) -> bool:\n        """Require multiple distinct fresh quote updates before reversing context bias."""\n        if quote_update_version is None:\n            return False\n        now = time.monotonic()\n        state = self._reversal_confirmation.get(symbol)\n        if state is None or state.get("side") != side:\n            state = {\n                "side": side,\n                "count": 1,\n                "started": now,\n                "last_version": quote_update_version,\n            }\n            self._reversal_confirmation[symbol] = state\n        elif state.get("last_version") != quote_update_version:\n            state["count"] = int(state.get("count") or 0) + 1\n            state["last_version"] = quote_update_version\n        min_updates = max(\n            2,\n            int(float(os.getenv("ORDERFLOW_REVERSAL_MIN_UPDATES", "3") or "3")),\n        )\n        min_persistence_ms = max(\n            0.0,\n            safe_float_env("ORDERFLOW_REVERSAL_MIN_PERSISTENCE_MS", 500.0),\n        )\n        elapsed_ms = max(0.0, now - float(state.get("started") or now)) * 1000.0\n        return int(state.get("count") or 0) >= min_updates and elapsed_ms >= min_persistence_ms\n\n''',
        sentinel="def _reversal_persistence_confirmed",
    )
    replace_once(
        order_flow,
        '''            require_tradable_quote_live = str(os.getenv('ORDERFLOW_REQUIRE_TRADABLE_QUOTE_LIVE', 'true')).strip().lower() in {'1', 'true', 'yes', 'on'}\n            tradable_quote = bool(indicators.get('tradable_quote', True))\n            quote_depth_valid = bool(indicators.get('quote_depth_valid', True))\n            tick_age_ms = float(indicators.get('tick_age_ms') or 0.0)\n            max_tick_age_ms = float(os.getenv('LIVE_MAX_TICK_AGE_MS', '2500') or '2500')\n''',
        '''            require_tradable_quote_live = str(os.getenv('ORDERFLOW_REQUIRE_TRADABLE_QUOTE_LIVE', 'true')).strip().lower() in {'1', 'true', 'yes', 'on'}\n            execution_quote_policy = ExecutionQuotePolicy.from_env(require_depth=True)\n            max_tick_age_ms = execution_quote_policy.max_tick_age_ms\n            tradable_quote = False\n            quote_depth_valid = False\n            tick_age_ms: float | None = None\n            quote_update_version: object | None = None\n''',
        sentinel="execution_quote_policy = ExecutionQuotePolicy",
    )
    insert_before(
        order_flow,
        "            if total_bid + total_ask <= 0:\n",
        '''            quote_payload = dict(indicators)\n            quote_payload.update(\n                {\n                    "ltp": current_price,\n                    "bid": bid,\n                    "ask": ask,\n                    "depth": depth,\n                    "depth_available": depth_available,\n                    "tradable_quote": bool(\n                        indicators.get("tradable_quote") is True\n                        or (bid > 0 and ask > bid)\n                    ),\n                    "spread_pct": spread_pct,\n                }\n            )\n            quote_readiness = evaluate_execution_quote_readiness(\n                symbol,\n                quote_payload,\n                live_mode=is_live_mode,\n                policy=execution_quote_policy,\n            )\n            tradable_quote = quote_readiness.tradable_quote\n            quote_depth_valid = quote_readiness.depth_available\n            tick_age_ms = quote_readiness.tick_age_ms\n            quote_update_version = quote_readiness.quote_update_version\n\n''',
        sentinel="quote_readiness = evaluate_execution_quote_readiness",
    )
    replace_once(
        order_flow,
        '''            microstructure_confirms_side = bool(tick_supports and depth_available and imbalance_confirms)\n            bias_invalidated_by_microstructure = bool(bias_conflict and microstructure_confirms_side)\n            side_alignment_ok = (direction not in {'CE', 'PE'}) or side_aligns or bias_invalidated_by_microstructure\n''',
        '''            raw_microstructure_confirms_side = bool(\n                tick_supports and depth_available and imbalance_confirms\n            )\n            reversal_persistence_confirmed = False\n            if bias_conflict and raw_microstructure_confirms_side:\n                reversal_persistence_confirmed = (\n                    not is_live_mode\n                    or self._reversal_persistence_confirmed(\n                        symbol=symbol,\n                        side=side,\n                        quote_update_version=quote_update_version,\n                    )\n                )\n            else:\n                self._reversal_confirmation.pop(symbol, None)\n            microstructure_confirms_side = bool(\n                raw_microstructure_confirms_side\n                and (not is_live_mode or reversal_persistence_confirmed)\n            )\n            bias_invalidated_by_microstructure = bool(\n                bias_conflict and microstructure_confirms_side\n            )\n            side_alignment_ok = (direction not in {'CE', 'PE'}) or side_aligns or bias_invalidated_by_microstructure\n''',
        sentinel="raw_microstructure_confirms_side",
    )
    # Remove the first of the duplicated stale-bias log blocks; retain one canonical emission.
    text = _read(order_flow)
    duplicate_block = '''            if bias_invalidated_by_microstructure:\n                LOGGER.info(\n                    'ORDERFLOW_STALE_BIAS_INVALIDATED symbol=%s side=%s stale_bias=%s '\n                    'depth_imbalance=%.3f tick_direction=%s score=%.2f',\n                    symbol, side, direction, depth_imbalance, tick_direction, strategy_score,\n                    extra={\n                        'event': 'ORDERFLOW_STALE_BIAS_INVALIDATED',\n                        'symbol': symbol, 'side': side, 'stale_bias': direction,\n                        'depth_imbalance': round(depth_imbalance, 4),\n                        'tick_direction': tick_direction, 'score': strategy_score,\n                    },\n                )\n'''
    if text.count(duplicate_block) == 2:
        text = text.replace(duplicate_block, "", 1)
        _write(order_flow, text)
    elif text.count(duplicate_block) not in {0, 1}:
        raise RuntimeError("order_flow.py: unexpected duplicate stale-bias log count")
    replace_once(
        order_flow,
        '''                and tick_supports\n                and tick_age_ms <= max_tick_age_ms\n            )\n''',
        '''                and tick_supports\n                and quote_readiness.allowed\n                and tick_age_ms is not None\n                and tick_age_ms <= max_tick_age_ms\n            )\n''',
        sentinel="and quote_readiness.allowed",
    )
    replace_once(
        order_flow,
        '''                and bool(tick_supports)\n                and tick_age_ms <= max_tick_age_ms\n                and bool(selected_or_near_atm)\n''',
        '''                and bool(tick_supports)\n                and quote_readiness.allowed\n                and tick_age_ms is not None\n                and tick_age_ms <= max_tick_age_ms\n                and bool(selected_or_near_atm)\n                and (not is_live_mode or bias_invalidated_by_microstructure)\n''',
        sentinel="and (not is_live_mode or bias_invalidated_by_microstructure)",
    )
    replace_once(
        order_flow,
        '''            elif not quote_depth_valid or not depth_available:\n                trigger_block_reason = 'quote_depth_missing'\n            elif is_live_mode and require_tradable_quote_live and not tradable_quote:\n''',
        '''            elif is_live_mode and not quote_readiness.allowed:\n                trigger_block_reason = quote_readiness.reason\n            elif not quote_depth_valid or not depth_available:\n                trigger_block_reason = 'quote_depth_missing'\n            elif is_live_mode and require_tradable_quote_live and not tradable_quote:\n''',
        sentinel="trigger_block_reason = quote_readiness.reason",
    )
    replace_once(
        order_flow,
        "            elif tick_age_ms > max_tick_age_ms:\n",
        "            elif tick_age_ms is None or tick_age_ms > max_tick_age_ms:\n",
    )
    replace_once(
        order_flow,
        "                 'quote_update_version': indicators.get('quote_update_version'),\n",
        "                 'quote_update_version': quote_update_version,\n                 'quote_readiness_allowed': quote_readiness.allowed,\n                 'quote_readiness_reason': quote_readiness.reason,\n                 'real_ticks_last_60s': quote_readiness.real_ticks_last_60s,\n                 'real_tick_count_derived': quote_readiness.real_tick_count_derived,\n                 'reversal_persistence_confirmed': reversal_persistence_confirmed,\n",
        sentinel="'quote_readiness_allowed': quote_readiness.allowed",
    )

    selector = "src/nifty_scalper_bot/strategies/trade_selector.py"
    replace_once(
        selector,
        "from nifty_scalper_bot.risk.expiry_gate import expiry_theta_block, midday_pause_block\n",
        "from nifty_scalper_bot.execution.readiness import (\n    ExecutionQuotePolicy,\n    evaluate_execution_quote_readiness,\n)\nfrom nifty_scalper_bot.risk.expiry_gate import expiry_theta_block, midday_pause_block\n",
    )
    replace_once(
        selector,
        "        allow_ltp_only = os.getenv('ALLOW_LTP_ONLY_CANDIDATE', 'false').lower() in {'1', 'true', 'yes', 'on'}\n        ranked: list[TradeCandidate] = []\n        rejects = {'side_mismatch': 0, 'atm_distance': 0, 'missing_bid_ask': 0, 'premium_out_of_range': 0, 'spread_too_wide': 0, 'tick_stale': 0, 'insufficient_ticks': 0, 'invalid_rr': 0, 'cost_edge_insufficient': 0}\n",
        "        allow_ltp_only = os.getenv('ALLOW_LTP_ONLY_CANDIDATE', 'false').lower() in {'1', 'true', 'yes', 'on'}\n        is_live_mode = str(os.getenv('EXECUTION_MODE', 'SHADOW') or 'SHADOW').strip().upper() == 'LIVE'\n        execution_quote_policy = ExecutionQuotePolicy.from_env(require_depth=False)\n        if is_live_mode:\n            execution_quote_policy = ExecutionQuotePolicy(\n                max_tick_age_ms=min(execution_quote_policy.max_tick_age_ms, max_age * 1000.0),\n                max_spread_pct=min(execution_quote_policy.max_spread_pct, max_spread),\n                min_real_ticks_last_60s=max(execution_quote_policy.min_real_ticks_last_60s, min_ticks),\n                require_update_version=execution_quote_policy.require_update_version,\n                require_tradable_quote=not allow_ltp_only,\n                require_depth=False,\n            )\n        ranked: list[TradeCandidate] = []\n        rejects = {'side_mismatch': 0, 'atm_distance': 0, 'missing_bid_ask': 0, 'premium_out_of_range': 0, 'spread_too_wide': 0, 'tick_stale': 0, 'insufficient_ticks': 0, 'quote_metadata_missing': 0, 'quote_not_tradable': 0, 'invalid_rr': 0, 'cost_edge_insufficient': 0}\n",
        sentinel="execution_quote_policy = ExecutionQuotePolicy.from_env",
    )
    replace_once(
        selector,
        '''            tick_age_s = self._f(s.get('tick_age_s'))\n            if tick_age_s is None or tick_age_s > max_age:\n                rejects['tick_stale'] += 1\n                self._log_reject("tick_stale", symbol, throttle_key_parts=("tick_stale", symbol, int(max_age)), tick_age_s=tick_age_s, max_age_s=max_age)\n                continue\n            real_ticks = int(s.get('real_ticks_last_60s') or 0)\n            if real_ticks < min_ticks:\n                rejects['insufficient_ticks'] += 1\n                self._log_reject("insufficient_ticks", symbol, throttle_key_parts=("insufficient_ticks", symbol, min_ticks), real_ticks_last_60s=real_ticks, min_ticks=min_ticks)\n                continue\n''',
        '''            if is_live_mode:\n                quote_readiness = evaluate_execution_quote_readiness(\n                    symbol,\n                    s,\n                    live_mode=True,\n                    policy=execution_quote_policy,\n                )\n                if not quote_readiness.allowed:\n                    reason_map = {\n                        "tick_age_missing": "tick_stale",\n                        "quote_stale": "tick_stale",\n                        "real_tick_count_missing": "insufficient_ticks",\n                        "insufficient_real_ticks": "insufficient_ticks",\n                        "quote_update_version_missing": "quote_metadata_missing",\n                        "spread_too_wide": "spread_too_wide",\n                        "ltp_missing": "missing_bid_ask",\n                        "bid_ask_missing": "missing_bid_ask",\n                        "bid_ask_crossed": "missing_bid_ask",\n                    }\n                    reject_key = reason_map.get(quote_readiness.reason, "quote_not_tradable")\n                    rejects[reject_key] += 1\n                    self._log_reject(\n                        quote_readiness.reason,\n                        symbol,\n                        throttle_key_parts=(quote_readiness.reason, symbol),\n                        quote_readiness=quote_readiness.to_dict(),\n                    )\n                    continue\n                tick_age_s = (quote_readiness.tick_age_ms or 0.0) / 1000.0\n                real_ticks = int(quote_readiness.real_ticks_last_60s or 0)\n            else:\n                tick_age_s = self._f(s.get('tick_age_s'))\n                if tick_age_s is None or tick_age_s > max_age:\n                    rejects['tick_stale'] += 1\n                    self._log_reject("tick_stale", symbol, throttle_key_parts=("tick_stale", symbol, int(max_age)), tick_age_s=tick_age_s, max_age_s=max_age)\n                    continue\n                real_ticks = int(s.get('real_ticks_last_60s') or 0)\n                if real_ticks < min_ticks:\n                    rejects['insufficient_ticks'] += 1\n                    self._log_reject("insufficient_ticks", symbol, throttle_key_parts=("insufficient_ticks", symbol, min_ticks), real_ticks_last_60s=real_ticks, min_ticks=min_ticks)\n                    continue\n''',
        sentinel="quote_readiness = evaluate_execution_quote_readiness",
    )

    _write(
        "tests/execution/test_execution_quote_readiness.py",
        '''from __future__ import annotations\n\nfrom nifty_scalper_bot.execution.readiness import (\n    ExecutionQuotePolicy,\n    evaluate_execution_quote_readiness,\n)\n\n\ndef _policy(**overrides):\n    values = dict(\n        max_tick_age_ms=2500.0,\n        max_spread_pct=0.75,\n        min_real_ticks_last_60s=1,\n        require_update_version=True,\n        require_tradable_quote=True,\n        require_depth=False,\n    )\n    values.update(overrides)\n    return ExecutionQuotePolicy(**values)\n\n\ndef test_live_quote_missing_age_fails_closed():\n    result = evaluate_execution_quote_readiness(\n        "NFO:NIFTY26JUN24000CE",\n        {"ltp": 100, "bid": 99.9, "ask": 100.1, "quote_update_version": 1},\n        live_mode=True,\n        policy=_policy(),\n    )\n    assert result.allowed is False\n    assert result.reason == "tick_age_missing"\n\n\ndef test_live_quote_missing_update_version_fails_closed():\n    result = evaluate_execution_quote_readiness(\n        "NFO:NIFTY26JUN24000CE",\n        {"ltp": 100, "bid": 99.9, "ask": 100.1, "tick_age_ms": 50},\n        live_mode=True,\n        policy=_policy(),\n    )\n    assert result.allowed is False\n    assert result.reason == "quote_update_version_missing"\n\n\ndef test_fresh_timestamp_is_version_and_single_tick_proof():\n    result = evaluate_execution_quote_readiness(\n        "NFO:NIFTY26JUN24000PE",\n        {\n            "ltp": 120,\n            "bid": 119.9,\n            "ask": 120.1,\n            "data_age_seconds": 0.1,\n            "timestamp_ms": 1234567890123,\n        },\n        live_mode=True,\n        policy=_policy(),\n    )\n    assert result.allowed is True\n    assert result.real_ticks_last_60s == 1\n    assert result.real_tick_count_derived is True\n\n\ndef test_configured_multi_tick_minimum_requires_explicit_count():\n    result = evaluate_execution_quote_readiness(\n        "NFO:NIFTY26JUN24000PE",\n        {\n            "ltp": 120,\n            "bid": 119.9,\n            "ask": 120.1,\n            "tick_age_ms": 100,\n            "quote_update_version": 8,\n        },\n        live_mode=True,\n        policy=_policy(min_real_ticks_last_60s=3),\n    )\n    assert result.allowed is False\n    assert result.reason == "insufficient_real_ticks"\n''',
    )
    _write(
        "tests/strategies/test_base_elite_live_gates.py",
        '''from __future__ import annotations\n\nfrom datetime import datetime, timezone\n\nfrom nifty_scalper_bot.strategies.elite_strategies import base_elite\nfrom nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy\nfrom nifty_scalper_bot.strategies.elite_strategies.config_models import EliteStrategyConfig\n\n\nclass _Strategy(EliteStrategy):\n    def __init__(self, config):\n        super().__init__(config, indicator_engine=None)\n        self.calls = 0\n\n    def _evaluate_signal(self, symbol, indicators, current_price, position=None):\n        self.calls += 1\n        return EliteSignal(\n            symbol=symbol,\n            signal="BUY",\n            confidence=0.9,\n            entry_price=current_price,\n            stop_loss=90.0,\n            target=120.0,\n            strategy_name="Test",\n            metadata={"role": "trigger", "can_trigger": True},\n        )\n\n\ndef test_live_option_is_blocked_outside_safe_entry_window(monkeypatch):\n    monkeypatch.setenv("EXECUTION_MODE", "LIVE")\n    monkeypatch.setattr(base_elite, "is_market_hours_cached", lambda: False)\n    strategy = _Strategy(EliteStrategyConfig(min_confidence=0, cooldown_seconds=0))\n    assert strategy.generate_signal("NFO:NIFTY26JUN24000CE", {}, 100.0) is None\n    assert strategy.calls == 0\n    assert strategy.last_no_vote_reason == "outside_safe_entry_window"\n\n\ndef test_push_generate_signal_honours_strategy_cooldown(monkeypatch):\n    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")\n    strategy = _Strategy(EliteStrategyConfig(min_confidence=0, cooldown_seconds=60))\n    assert strategy.generate_signal("NFO:NIFTY26JUN24000CE", {}, 100.0) is not None\n    assert strategy.generate_signal("NFO:NIFTY26JUN24050CE", {}, 101.0) is None\n    assert strategy.calls == 1\n    assert strategy.last_no_vote_reason == "strategy_cooldown_active"\n\n\ndef test_context_only_vote_does_not_start_trigger_cooldown(monkeypatch):\n    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")\n    strategy = _Strategy(EliteStrategyConfig(min_confidence=0, cooldown_seconds=60))\n    context = EliteSignal(\n        symbol="NFO:NIFTY26JUN24000CE", signal="BUY", confidence=0.8,\n        entry_price=100.0, stop_loss=None, target=None, strategy_name="Test",\n        metadata={"role": "context", "can_trigger": False, "trigger_conditions_met": False},\n        timestamp=datetime.now(timezone.utc),\n    )\n    strategy._process_signal(context)\n    assert strategy._last_signal_at is None\n''',
    )
    _write(
        "tests/strategies/test_order_flow_execution_quote_gate.py",
        '''from __future__ import annotations\n\nfrom nifty_scalper_bot.strategies.elite_strategies.config_models import OrderFlowStrategyConfig\nfrom nifty_scalper_bot.strategies.elite_strategies.order_flow import OrderFlowStrategy\n\n\ndef _indicators(**overrides):\n    value = {\n        "bid": 99.9,\n        "ask": 100.1,\n        "ltp": 100.0,\n        "spread_pct": 0.2,\n        "depth": {\n            "buy": [{"quantity": 900, "price": 99.9}],\n            "sell": [{"quantity": 100, "price": 100.1}],\n        },\n        "tick_direction": "UP",\n        "direction_bias": "CE",\n        "context_age_seconds": 0.1,\n        "tradable_quote": True,\n        "quote_depth_valid": True,\n        "is_selected_option": True,\n        "strike_distance_from_atm": 0,\n        "real_ticks_last_60s": 3,\n        "atr": 2.0,\n    }\n    value.update(overrides)\n    return value\n\n\ndef test_orderflow_live_missing_tick_age_cannot_trigger(monkeypatch):\n    monkeypatch.setenv("EXECUTION_MODE", "LIVE")\n    strategy = OrderFlowStrategy(OrderFlowStrategyConfig(min_confidence=0), None)\n    signal = strategy._evaluate_signal("NFO:NIFTY26JUN24000CE", _indicators(quote_update_version=1), 100.0)\n    assert signal is not None\n    assert signal.metadata["trigger_conditions_met"] is False\n    assert signal.metadata["trigger_block_reason"] == "tick_age_missing"\n\n\ndef test_orderflow_live_fresh_versioned_quote_can_trigger(monkeypatch):\n    monkeypatch.setenv("EXECUTION_MODE", "LIVE")\n    strategy = OrderFlowStrategy(OrderFlowStrategyConfig(min_confidence=0), None)\n    signal = strategy._evaluate_signal(\n        "NFO:NIFTY26JUN24000CE",\n        _indicators(tick_age_ms=100, quote_update_version=1),\n        100.0,\n    )\n    assert signal is not None\n    assert signal.metadata["trigger_conditions_met"] is True\n    assert signal.metadata["quote_readiness_allowed"] is True\n\n\ndef test_bias_reversal_requires_distinct_persistent_updates(monkeypatch):\n    monkeypatch.setenv("EXECUTION_MODE", "LIVE")\n    monkeypatch.setenv("ORDERFLOW_REVERSAL_MIN_UPDATES", "3")\n    monkeypatch.setenv("ORDERFLOW_REVERSAL_MIN_PERSISTENCE_MS", "500")\n    strategy = OrderFlowStrategy(OrderFlowStrategyConfig(min_confidence=0), None)\n    times = iter([10.0, 10.2, 10.7])\n    monkeypatch.setattr("nifty_scalper_bot.strategies.elite_strategies.order_flow.time.monotonic", lambda: next(times))\n    results = []\n    for version in (1, 2, 3):\n        results.append(\n            strategy._evaluate_signal(\n                "NFO:NIFTY26JUN24000CE",\n                _indicators(\n                    direction_bias="PE",\n                    tick_age_ms=100,\n                    quote_update_version=version,\n                ),\n                100.0,\n            )\n        )\n    assert results[0].metadata["trigger_conditions_met"] is False\n    assert results[1].metadata["trigger_conditions_met"] is False\n    assert results[2].metadata["trigger_conditions_met"] is True\n    assert results[2].metadata["reversal_persistence_confirmed"] is True\n''',
    )


def stage_runtime() -> None:
    runtime_om = "src/nifty_scalper_bot/execution/runtime_order_manager.py"
    replace_once(runtime_om, "from typing import Any, Callable\n", "from typing import Any, Callable, Mapping\n")
    insert_before(
        runtime_om,
        "    def _update_from_response(\n",
        '''    @staticmethod\n    def _blocked_health_snapshot(\n        reason: str,\n        *,\n        error: Exception | None = None,\n        original: Mapping[str, Any] | None = None,\n    ) -> dict[str, Any]:\n        snapshot = dict(original or {})\n        snapshot.update(\n            {\n                "ready": False,\n                "order_api_ready": False,\n                "order_api_available": False,\n                "broker_connected": False,\n                "trading_allowed_effect": "live_orders_blocked",\n                "effect": "live_orders_blocked",\n                "block_class": reason,\n                "broker_health_reason": reason,\n            }\n        )\n        if error is not None:\n            snapshot["last_broker_error"] = str(error)\n            snapshot["broker_health_error_type"] = type(error).__name__\n        return snapshot\n\n    def get_broker_health_snapshot(self) -> dict[str, Any]:\n        """Return a mapping in all cases; unknown LIVE broker health fails closed."""\n        try:\n            raw = super().get_broker_health_snapshot()\n        except Exception as exc:\n            return self._blocked_health_snapshot("broker_health_exception", error=exc)\n        if not isinstance(raw, Mapping):\n            return self._blocked_health_snapshot("broker_health_invalid_payload")\n        snapshot = dict(raw)\n        try:\n            live_mode = bool(self.is_live_mode())\n        except Exception:\n            import os\n            live_mode = str(os.getenv("EXECUTION_MODE", "SHADOW")).strip().upper() == "LIVE"\n        if not live_mode:\n            return snapshot\n        effect = snapshot.get("trading_allowed_effect") or snapshot.get("effect")\n        if effect == "live_orders_blocked":\n            return snapshot\n        ready = snapshot.get("ready")\n        order_api_ready = snapshot.get("order_api_ready", snapshot.get("order_api_available"))\n        broker_connected = snapshot.get("broker_connected")\n        if ready is not True or order_api_ready is False or broker_connected is False:\n            return self._blocked_health_snapshot(\n                "broker_health_unknown",\n                original=snapshot,\n            )\n        return snapshot\n\n''',
        sentinel="def _blocked_health_snapshot",
    )

    runner = "src/nifty_scalper_bot/strategies/runner.py"
    replace_once(
        runner,
        "from nifty_scalper_bot.execution.readiness import HistoryReadinessPolicy, resolve_quote_bid_ask_spread\n",
        "from nifty_scalper_bot.execution.readiness import (\n    ExecutionQuotePolicy,\n    HistoryReadinessPolicy,\n    evaluate_execution_quote_readiness,\n    resolve_quote_bid_ask_spread,\n)\n",
    )
    replace_once(
        runner,
        '''        if not callable(health_fn):\n            details["broker_ready"] = True\n            details["broker_ready_assumed"] = True\n            return True, "broker_health_unknown_assumed_ready", details\n''',
        '''        if not callable(health_fn):\n            live_mode = self._resolve_execution_mode_snapshot().is_live_mode\n            details["broker_ready"] = False if live_mode else True\n            details["broker_ready_assumed"] = not live_mode\n            details["broker_health_block_reason"] = "broker_health_unavailable"\n            if live_mode:\n                return False, "broker_health_unknown", details\n            return True, "broker_health_unknown_assumed_ready", details\n''',
        sentinel='details["broker_health_block_reason"] = "broker_health_unavailable"',
    )
    replace_once(
        runner,
        '''        except Exception as exc:\n            details["broker_health_source"] = health_source\n            details["broker_health_error_type"] = type(exc).__name__\n            details["broker_health_error"] = str(exc)\n            details["broker_ready"] = True\n            details["broker_ready_assumed"] = True\n            return True, "broker_health_unknown_assumed_ready", details\n''',
        '''        except Exception as exc:\n            live_mode = self._resolve_execution_mode_snapshot().is_live_mode\n            details["broker_health_source"] = health_source\n            details["broker_health_error_type"] = type(exc).__name__\n            details["broker_health_error"] = str(exc)\n            details["broker_ready"] = False if live_mode else True\n            details["broker_ready_assumed"] = not live_mode\n            details["broker_health_block_reason"] = "broker_health_exception"\n            if live_mode:\n                return False, "broker_health_unknown", details\n            return True, "broker_health_unknown_assumed_ready", details\n''',
        sentinel='details["broker_health_block_reason"] = "broker_health_exception"',
    )
    replace_once(
        runner,
        '''        if not isinstance(raw_health, Mapping):\n            details["broker_health_source"] = health_source\n            details["broker_ready"] = True\n            details["broker_ready_assumed"] = True\n            return True, "broker_health_unknown_assumed_ready", details\n''',
        '''        if not isinstance(raw_health, Mapping):\n            live_mode = self._resolve_execution_mode_snapshot().is_live_mode\n            details["broker_health_source"] = health_source\n            details["broker_ready"] = False if live_mode else True\n            details["broker_ready_assumed"] = not live_mode\n            details["broker_health_block_reason"] = "broker_health_invalid_payload"\n            if live_mode:\n                return False, "broker_health_unknown", details\n            return True, "broker_health_unknown_assumed_ready", details\n''',
        sentinel='details["broker_health_block_reason"] = "broker_health_invalid_payload"',
    )
    regex_replace_once(
        runner,
        r"    def verify_state\(self\) -> bool:\n.*?\n    def _risk_kill_switch_triggered\(self\) -> bool:\n",
        '''    def verify_state(self) -> bool:\n        """Validate broker position state; LIVE uncertainty always blocks entries."""\n        try:\n            try:\n                live_mode = self._resolve_execution_mode_snapshot().is_live_mode\n            except Exception:\n                live_mode = str(os.getenv("EXECUTION_MODE", "SHADOW")).strip().upper() == "LIVE"\n            broker = getattr(self, "_broker", None) or getattr(\n                self._order_manager, "_broker", None\n            )\n            if broker is None or not hasattr(broker, "get_positions"):\n                self._logger.warning(\n                    "BROKER_STATE_VERIFICATION_UNAVAILABLE live_mode=%s",\n                    live_mode,\n                    extra={"event": "BROKER_STATE_VERIFICATION_UNAVAILABLE", "live_mode": live_mode},\n                )\n                return not live_mode\n\n            sync_broker = getattr(\n                broker,\n                "client",\n                getattr(broker, "_broker", broker),\n            )\n            method = getattr(sync_broker, "get_positions", None)\n            if method is None:\n                return not live_mode\n\n            broker_positions = method()\n            if asyncio.iscoroutine(broker_positions):\n                broker_positions.close()\n                self._logger.error(\n                    "BROKER_STATE_VERIFICATION_ASYNC_RESULT_BLOCKED",\n                    extra={"event": "BROKER_STATE_VERIFICATION_ASYNC_RESULT_BLOCKED"},\n                )\n                return not live_mode\n            if broker_positions is None:\n                return not live_mode\n            return True\n        except Exception as exc:\n            self._logger.error(\n                "State verification failed: %s",\n                exc,\n                extra={"event": "BROKER_STATE_VERIFICATION_FAILED", "error_type": type(exc).__name__},\n                exc_info=True,\n            )\n            return False\n\n    def _risk_kill_switch_triggered(self) -> bool:\n''',
        sentinel="BROKER_STATE_VERIFICATION_ASYNC_RESULT_BLOCKED",
    )
    insert_before(
        runner,
        "        risk_kill_switch = self._risk_kill_switch_triggered()\n",
        '''        execution_quote_policy = ExecutionQuotePolicy.from_env(\n            require_depth=_env_bool("REQUIRE_FULL_DEPTH_FOR_EXECUTION", False)\n        )\n        quote_readiness = evaluate_execution_quote_readiness(\n            symbol_norm,\n            quote,\n            live_mode=mode_snapshot.is_live_mode,\n            policy=execution_quote_policy,\n        )\n''',
        sentinel="execution_quote_policy = ExecutionQuotePolicy.from_env",
    )
    replace_once(
        runner,
        '            "bid_ask_source": bid_ask_source,\n        }\n',
        '            "bid_ask_source": bid_ask_source,\n            "quote_readiness_allowed": quote_readiness.allowed,\n            "quote_readiness_reason": quote_readiness.reason,\n            "quote_tick_age_ms": quote_readiness.tick_age_ms,\n            "quote_update_version": quote_readiness.quote_update_version,\n            "real_ticks_last_60s": quote_readiness.real_ticks_last_60s,\n            "real_tick_count_derived": quote_readiness.real_tick_count_derived,\n        }\n',
        sentinel='"quote_readiness_allowed": quote_readiness.allowed',
    )
    replace_once(
        runner,
        '''        quote_fresh = self._is_option_symbol_tick_fresh(symbol, max_age_s=60.0)\n        details["quote_fresh"] = quote_fresh\n        if not quote_fresh:\n            return _finish(False, "option_tick_stale")\n''',
        '''        quote_fresh = quote_readiness.allowed\n        details["quote_fresh"] = quote_fresh\n        if not quote_fresh:\n            return _finish(False, quote_readiness.reason)\n''',
        sentinel="return _finish(False, quote_readiness.reason)",
    )
    insert_before(
        runner,
        "                        signal = self._strategy_manager.generate_signal(\n",
        '''                        if (\n                            self._resolve_execution_mode_snapshot().is_live_mode\n                            and self._is_tradable_symbol(symbol)\n                            and not is_market_hours_cached()\n                        ):\n                            self._emit_runner_eval_decision(\n                                symbol=symbol,\n                                stage="phase9",\n                                reason="outside_safe_entry_window",\n                                allowed=False,\n                                trace_id=trace_id,\n                            )\n                            return\n''',
        sentinel='reason="outside_safe_entry_window"',
    )
    replace_once(
        runner,
        '                    "SIGNAL_GENERATED symbol=%s action=%s reason=%s trace_id=%s",\n',
        '                    "SIGNAL_PREPARATION_REQUESTED symbol=%s action=%s reason=%s trace_id=%s",\n',
    )
    replace_once(
        runner,
        '                        "event": "SIGNAL_GENERATED",\n',
        '                        "event": "SIGNAL_PREPARATION_REQUESTED",\n',
    )
    insert_before(
        runner,
        '        self._logger.info(\n            f"🔴 1. SIGNAL HANDLER ENTERED: {signal.symbol} {signal.action}"\n        )\n',
        '''        self._logger.info(\n            "SIGNAL_GENERATED symbol=%s action=%s reason=%s trace_id=%s",\n            signal.symbol,\n            signal.action,\n            signal.reason,\n            trace_id,\n            extra={\n                "event": "SIGNAL_GENERATED",\n                "symbol": signal.symbol,\n                "action": signal.action,\n                "reason": signal.reason,\n                "trace_id": trace_id,\n                "execution_stage": "handler_accepted",\n            },\n        )\n''',
        sentinel='"execution_stage": "handler_accepted"',
    )

    candle = "src/nifty_scalper_bot/data/candle_engine.py"
    replace_once(
        candle,
        "    LOGGER.info('data_integrity_error', extra={'event': 'data_integrity_error', 'symbol': symbol, 'reason': 'repair_with_backfill_deprecated'})\n",
        "    LOGGER.info('DATA_INTEGRITY_ERROR symbol=%s reason=repair_with_backfill_deprecated', symbol, extra={'event': 'data_integrity_error', 'symbol': symbol, 'reason': 'repair_with_backfill_deprecated'})\n",
    )
    replace_once(
        candle,
        "            'data_integrity_error',\n            extra={'event': 'data_integrity_error', 'symbol': symbol, 'attempt': attempt, 'reason': 'historical_validation_failed'},\n",
        "            'DATA_INTEGRITY_ERROR symbol=%s reason=historical_validation_failed attempt=%s',\n            symbol,\n            attempt,\n            extra={'event': 'data_integrity_error', 'symbol': symbol, 'attempt': attempt, 'reason': 'historical_validation_failed'},\n",
    )
    replace_once(
        candle,
        "        LOGGER.error('data_integrity_error', extra={'event': 'data_integrity_error', 'symbol': symbol, 'reason': 'insufficient_historical'})\n",
        "        LOGGER.error('DATA_INTEGRITY_ERROR symbol=%s reason=insufficient_historical min_required=%s', symbol, min_required, extra={'event': 'data_integrity_error', 'symbol': symbol, 'reason': 'insufficient_historical', 'min_required': min_required})\n",
    )

    _write(
        "tests/execution/test_runtime_order_manager_health_fail_closed.py",
        '''from __future__ import annotations\n\nfrom nifty_scalper_bot.execution import order_manager_core\nfrom nifty_scalper_bot.execution.runtime_order_manager import RuntimeOrderManager\n\n\ndef test_runtime_order_manager_health_exception_fails_closed(monkeypatch):\n    manager = RuntimeOrderManager.__new__(RuntimeOrderManager)\n    manager.is_live_mode = lambda: True\n    monkeypatch.setattr(\n        order_manager_core.OrderManager,\n        "get_broker_health_snapshot",\n        lambda _self: (_ for _ in ()).throw(RuntimeError("boom")),\n    )\n    result = manager.get_broker_health_snapshot()\n    assert result["ready"] is False\n    assert result["trading_allowed_effect"] == "live_orders_blocked"\n    assert result["block_class"] == "broker_health_exception"\n\n\ndef test_runtime_order_manager_unknown_live_health_fails_closed(monkeypatch):\n    manager = RuntimeOrderManager.__new__(RuntimeOrderManager)\n    manager.is_live_mode = lambda: True\n    monkeypatch.setattr(\n        order_manager_core.OrderManager,\n        "get_broker_health_snapshot",\n        lambda _self: {"ready": None},\n    )\n    result = manager.get_broker_health_snapshot()\n    assert result["ready"] is False\n    assert result["block_class"] == "broker_health_unknown"\n''',
    )
    _write(
        "tests/strategies/test_runner_live_fail_closed.py",
        '''from __future__ import annotations\n\nfrom types import SimpleNamespace\n\nfrom nifty_scalper_bot.strategies.runner import StrategyRunner\n\n\nclass _Logger:\n    def warning(self, *args, **kwargs): pass\n    def error(self, *args, **kwargs): pass\n\n\ndef _runner():\n    runner = StrategyRunner.__new__(StrategyRunner)\n    runner._logger = _Logger()\n    runner._resolve_execution_mode_snapshot = lambda: SimpleNamespace(is_live_mode=True)\n    return runner\n\n\ndef test_unknown_order_manager_health_blocks_live_entry():\n    runner = _runner()\n    runner._order_manager = SimpleNamespace(is_live_mode=lambda: True, is_kill_switch_active=lambda: False)\n    runner._order_manager_kill_switch_status_for_entry = lambda: (False, {})\n    allowed, reason, details = runner._resolve_order_manager_health_for_entry()\n    assert allowed is False\n    assert reason == "broker_health_unknown"\n    assert details["broker_ready"] is False\n\n\ndef test_missing_broker_position_api_blocks_live_state_verification():\n    runner = _runner()\n    runner._order_manager = SimpleNamespace(_broker=None)\n    runner._broker = None\n    assert runner.verify_state() is False\n''',
    )


def stage_dashboard() -> None:
    event_buffer = "dashboard/event_buffer.py"
    replace_once(
        event_buffer,
        "import re\n",
        "import re\nfrom typing import Iterable\n",
    )
    replace_once(
        event_buffer,
        '''    if any(x in upper for x in ("ERROR","FAIL","TRACEBACK","REJECTED")):\n        kind = "ERROR"\n    elif any(x in upper for x in ("WARN","DEGRADED")):\n''',
        '''    strong_errors = (\n        "TRACEBACK", "UNHANDLED EXCEPTION", "CRITICAL", "RUNNER_ON_TICK_ERROR",\n        "HANDLER CRASHED", "ORDER_FAILED", "STARTUP_FAILED", "_ERROR ",\n    )\n    if any(x in upper for x in strong_errors):\n        kind = "ERROR"\n    elif any(x in upper for x in ("WARN","DEGRADED")):\n''',
        sentinel="strong_errors = (",
    )
    insert_before(
        event_buffer,
        "class EventRing:\n",
        '''_TRACE_RE = re.compile(r"\\btrace_id=([^\\s,}]+)")\n\n\ndef deduplicate_events(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:\n    """Remove duplicate terminal outcomes while preserving event order."""\n    output: list[dict[str, str]] = []\n    terminal_seen: set[tuple[str, str]] = set()\n    for row in rows:\n        message = row.get("message", "")\n        if "SIGNAL_EXECUTION_RESULT" in message:\n            match = _TRACE_RE.search(message)\n            if match:\n                outcome = re.search(r"accepted=([^\\s]+).*?reason=([^\\s]+)", message)\n                signature = outcome.group(0) if outcome else message\n                key = (match.group(1), signature)\n                if key in terminal_seen:\n                    continue\n                terminal_seen.add(key)\n        output.append(row)\n    return output\n\n\n''',
        sentinel="def deduplicate_events",
    )
    replace_once(
        event_buffer,
        "                    if event:\n                        with self.lock:\n                            if not self.rows or self.rows[-1] != event:\n                                self.rows.append(event)\n                                self.last_event = time.time()\n",
        "                    if event:\n                        with self.lock:\n                            candidate = deduplicate_events([*self.rows, event])\n                            if len(candidate) > len(self.rows):\n                                self.rows.append(event)\n                                self.last_event = time.time()\n",
        sentinel="candidate = deduplicate_events",
    )

    console = "dashboard/operations_console.py"
    replace_once(
        console,
        "from event_buffer import EventRing, parse_event\n",
        "from event_buffer import EventRing, deduplicate_events, parse_event\n",
    )
    replace_once(
        console,
        "    return EventRing(SERVICE, max_events=3000)\n",
        "    return EventRing(SERVICE, max_events=max(3000, int(os.getenv('BOT_EVENT_BUFFER_MAX', '10000') or '10000')))\n",
    )
    replace_once(
        console,
        '''    try:\n        response = http_session().get(API + path, timeout=1.2)\n        response.raise_for_status()\n        value = response.json()\n        return value if isinstance(value, dict) else None\n    except Exception:\n        return None\n''',
        '''    try:\n        response = http_session().get(API + path, timeout=1.2)\n        value = response.json()\n        if not isinstance(value, dict):\n            return None\n        value = dict(value)\n        value["_http_status"] = response.status_code\n        return value\n    except (requests.RequestException, ValueError):\n        return None\n''',
        sentinel='value["_http_status"] = response.status_code',
    )
    replace_once(
        console,
        "    return [\n        event\n        for line in result.stdout.splitlines()\n        if (event := parse_event(line))\n    ], None\n",
        "    rows = [\n        event\n        for line in result.stdout.splitlines()\n        if (event := parse_event(line))\n    ]\n    return deduplicate_events(rows), None\n",
    )
    replace_once(
        console,
        '''    broker_ready = bool(broker.get("ready"))\n    auth_invalid = bool(broker.get("auth_invalid"))\n    reconciled = bool(recon.get("completed"))\n''',
        '''    trading_available = trading is not None\n    broker_available = trading_available and bool(broker)\n    recon_available = trading_available and bool(recon)\n    broker_ready = bool(broker.get("ready")) if broker_available else None\n    auth_invalid = bool(broker.get("auth_invalid")) if broker_available else None\n    reconciled = bool(recon.get("completed")) if recon_available else None\n''',
        sentinel="broker_available = trading_available",
    )
    replace_once(
        console,
        '''        + state_item("Broker", "READY" if broker_ready else "NOT READY", "good" if broker_ready else "bad-text")\n        + state_item("Balance", short_value(broker.get("balance")))\n        + state_item("Reconciled", "YES" if reconciled else "NO", "good" if reconciled else "warn-text")\n        + state_item("Authentication", "INVALID" if auth_invalid else "OK", "bad-text" if auth_invalid else "good")\n''',
        '''        + state_item("Broker", "UNKNOWN" if broker_ready is None else ("READY" if broker_ready else "NOT READY"), "warn-text" if broker_ready is None else ("good" if broker_ready else "bad-text"))\n        + state_item("Balance", short_value(broker.get("balance")) if broker_available else "UNKNOWN")\n        + state_item("Reconciled", "UNKNOWN" if reconciled is None else ("YES" if reconciled else "NO"), "warn-text" if reconciled is None else ("good" if reconciled else "warn-text"))\n        + state_item("Authentication", "UNKNOWN" if auth_invalid is None else ("INVALID" if auth_invalid else "OK"), "warn-text" if auth_invalid is None else ("bad-text" if auth_invalid else "good"))\n''',
        sentinel='state_item("Authentication", "UNKNOWN"',
    )
    replace_once(
        console,
        "        '<div class=\"status-card\"><div class=\"card-title\">Deployment</div>'\n",
        "        '<div class=\"status-card\"><div class=\"card-title\">Deployment</div>'\n        f'<div class=\"deploy-row\"><span class=\"deploy-key\">Platform</span><span class=\"deploy-value\">{html.escape(os.getenv("DEPLOYMENT_PLATFORM", "AWS Lightsail / systemd"))}</span></div>'\n",
        sentinel='deploy-key">Platform',
    )

    _write(
        "tests/dashboard/test_event_buffer.py",
        '''from __future__ import annotations\n\nimport importlib.util\nfrom pathlib import Path\n\nMODULE_PATH = Path(__file__).resolve().parents[2] / "dashboard" / "event_buffer.py"\nspec = importlib.util.spec_from_file_location("dashboard_event_buffer", MODULE_PATH)\nmodule = importlib.util.module_from_spec(spec)\nassert spec.loader is not None\nspec.loader.exec_module(module)\n\n\ndef test_candidate_rejection_is_not_misclassified_as_error():\n    event = module.parse_event(\n        "[2026-06-25 15:00:00 IST] CANDIDATE_REJECTED symbol=X reason=tick_stale"\n    )\n    assert event is not None\n    assert event["type"] != "ERROR"\n\n\ndef test_actual_runner_error_is_error():\n    event = module.parse_event(\n        "[2026-06-25 15:00:00 IST] RUNNER_ON_TICK_ERROR symbol=X error=boom"\n    )\n    assert event is not None\n    assert event["type"] == "ERROR"\n\n\ndef test_duplicate_terminal_execution_result_is_removed():\n    rows = [\n        {"timestamp_ist": "2026-06-25 15:00:00 IST", "type": "SIGNAL", "message": "SIGNAL_EXECUTION_RESULT symbol=X accepted=False reason=no_execution_ready_candidate trace_id=t1"},\n        {"timestamp_ist": "2026-06-25 15:00:00 IST", "type": "TRADE", "message": "SIGNAL_EXECUTION_RESULT symbol=X accepted=False reason=no_execution_ready_candidate trace_id=t1"},\n    ]\n    assert len(module.deduplicate_events(rows)) == 1\n''',
    )


def stage_lightsail() -> None:
    setup = "deploy/lightsail_setup.sh"
    replace_once(
        setup,
        "sudo apt-get install -y -qq python3 python3-venv python3-pip git\n",
        "sudo apt-get install -y -qq python3 python3-venv python3-pip git curl\n",
    )
    replace_once(
        setup,
        "pip install --quiet -e .\npip install --quiet python-multipart uvicorn fastapi\n",
        "pip install --quiet -e \".[dev]\"\npip install --quiet python-multipart uvicorn fastapi\n",
    )
    insert_before(
        setup,
        "# --- systemd service: auto-start, auto-restart, survives reboot ---\n",
        '''# Add new safe defaults without overwriting operator-defined values.\nensure_env_default() {\n  local key="$1" value="$2"\n  if ! grep -qE "^${key}=" "$ENV_FILE"; then\n    printf '%s=%s\\n' "$key" "$value" >> "$ENV_FILE"\n  fi\n}\nensure_env_default DEPLOYMENT_PLATFORM aws_lightsail\nensure_env_default LIVE_ENTRY_SELECTED_ONLY true\nensure_env_default LIVE_EXECUTION_MAX_TICK_AGE_MS 2500\nensure_env_default LIVE_EXECUTION_MAX_SPREAD_PCT 0.75\nensure_env_default LIVE_EXECUTION_MIN_REAL_TICKS_60S 1\nensure_env_default LIVE_EXECUTION_REQUIRE_UPDATE_VERSION true\nensure_env_default ORDERFLOW_REVERSAL_MIN_UPDATES 3\nensure_env_default ORDERFLOW_REVERSAL_MIN_PERSISTENCE_MS 500\nensure_env_default BOT_EVENT_BUFFER_MAX 10000\n\n''',
        sentinel="ensure_env_default DEPLOYMENT_PLATFORM",
    )
    replace_once(
        setup,
        "EnvironmentFile=$ENV_FILE\nExecStart=$APP_DIR/.venv/bin/python -m uvicorn nifty_scalper_bot.main:app --host 0.0.0.0 --port $PORT\nRestart=always\nRestartSec=3\n",
        "EnvironmentFile=$ENV_FILE\nEnvironment=PYTHONUNBUFFERED=1\nExecStart=$APP_DIR/.venv/bin/python -m uvicorn nifty_scalper_bot.main:app --host 0.0.0.0 --port $PORT\nRestart=on-failure\nRestartSec=3\nTimeoutStopSec=30\nKillSignal=SIGINT\n",
    )
    regex_replace_once(
        setup,
        r"sudo tee /usr/local/bin/niftybot-autodeploy\.sh >/dev/null <<EOF\n.*?\nEOF\nsudo chmod \+x /usr/local/bin/niftybot-autodeploy\.sh",
        '''sudo tee /usr/local/bin/niftybot-autodeploy.sh >/dev/null <<EOF\n#!/usr/bin/env bash\nset -euo pipefail\nexec 9>/tmp/niftybot-autodeploy.lock\nflock -n 9 || exit 0\ncd $APP_DIR\nSTATUS_FILE=$APP_DIR/data/auto_update_status.json\nmkdir -p $APP_DIR/data\nwrite_status() {\n  local state=\"\\$1\" message=\"\\$2\"\n  printf '{\"state\":\"%s\",\"message\":\"%s\",\"updated_at\":\"%s\"}\\n' \\\n    \"\\$state\" \"\\${message//\"/\\\\\"}\" \"\\$(date -Is)\" > \"\\$STATUS_FILE.tmp\"\n  mv \"\\$STATUS_FILE.tmp\" \"\\$STATUS_FILE\"\n}\nBEFORE=\\$(git rev-parse HEAD 2>/dev/null || echo none)\ngit fetch --quiet origin main || { write_status fetch_failed \"git fetch failed\"; exit 0; }\nAFTER=\\$(git rev-parse origin/main 2>/dev/null || echo none)\nif [ \"\\$BEFORE\" = \"\\$AFTER\" ]; then\n  write_status current \"running \\${BEFORE:0:7}\"\n  exit 0\nfi\nwrite_status validating \"validating \\${AFTER:0:7}\"\nCANDIDATE=/tmp/niftybot-candidate-\\${AFTER:0:12}\nrm -rf \"\\$CANDIDATE\"\ngit worktree add --detach --quiet \"\\$CANDIDATE\" \"\\$AFTER\"\ncleanup() { git worktree remove --force \"\\$CANDIDATE\" >/dev/null 2>&1 || true; }\ntrap cleanup EXIT\nPYTHONPATH=\"\\$CANDIDATE/src\" $APP_DIR/.venv/bin/python -m compileall -q \"\\$CANDIDATE/src\" || {\n  write_status validation_failed \"compile failed for \\${AFTER:0:7}\"; exit 0;\n}\nTARGETED_TESTS=(\n  tests/execution/test_execution_quote_readiness.py\n  tests/execution/test_runtime_order_manager_health_fail_closed.py\n  tests/strategies/test_base_elite_live_gates.py\n  tests/strategies/test_order_flow_execution_quote_gate.py\n  tests/strategies/test_runner_live_fail_closed.py\n  tests/dashboard/test_event_buffer.py\n)\nEXISTING_TESTS=()\nfor test_path in \"\\${TARGETED_TESTS[@]}\"; do\n  [ -f \"\\$CANDIDATE/\\$test_path\" ] && EXISTING_TESTS+=(\"\\$CANDIDATE/\\$test_path\")\ndone\nif [ \"\\${#EXISTING_TESTS[@]}\" -gt 0 ]; then\n  PYTHONPATH=\"\\$CANDIDATE/src\" $APP_DIR/.venv/bin/python -m pytest -q \"\\${EXISTING_TESTS[@]}\" || {\n    write_status validation_failed \"targeted tests failed for \\${AFTER:0:7}\"; exit 0;\n  }\nfi\ngit reset --hard --quiet \"\\$AFTER\"\n$APP_DIR/.venv/bin/pip install --quiet -e \".[dev]\" || {\n  git reset --hard --quiet \"\\$BEFORE\"\n  $APP_DIR/.venv/bin/pip install --quiet -e \".[dev]\" || true\n  write_status install_failed \"dependency install failed; rolled back to \\${BEFORE:0:7}\"\n  exit 0\n}\nsudo systemctl restart ${SERVICE}\nfor _ in \\$(seq 1 15); do\n  if curl -fsS --max-time 2 http://127.0.0.1:${PORT}/livez >/dev/null; then\n    write_status deployed \"deployed \\${AFTER:0:7}\"\n    logger -t niftybot-autodeploy \"validated and deployed \\$BEFORE -> \\$AFTER\"\n    exit 0\n  fi\n  sleep 2\ndone\ngit reset --hard --quiet \"\\$BEFORE\"\n$APP_DIR/.venv/bin/pip install --quiet -e \".[dev]\" || true\nsudo systemctl restart ${SERVICE}\nwrite_status rolled_back \"health check failed; rolled back to \\${BEFORE:0:7}\"\nlogger -t niftybot-autodeploy \"health check failed; rolled back \\$AFTER -> \\$BEFORE\"\nEOF\nsudo chmod +x /usr/local/bin/niftybot-autodeploy.sh''',
        sentinel="write_status validating",
    )

    architecture = "ARCHITECTURE_TRADING_PATH.md"
    text = _read(architecture)
    if "AWS Lightsail deployment authority" not in text:
        header = '''# Nifty Scalper Bot — Production Trading Path\n\n## AWS Lightsail deployment authority\n\nThe production authority is a single Ubuntu AWS Lightsail instance managed by\n`systemd` service `niftybot`. It starts `uvicorn nifty_scalper_bot.main:app` from\n`/home/ubuntu/nifty_scalper_bot`. Railway is not part of the production path.\nThe Lightsail auto-deployer validates candidate commits in an isolated git\nworktree, runs compile and targeted safety tests, restarts only after validation,\nand rolls back automatically if `/livez` does not recover.\n\n'''
        if text.startswith("#"):
            text = re.sub(r"\A#.*?\n", header, text, count=1)
        else:
            text = header + text
        text = text.replace("Railway using `deployment_main.py`", "AWS Lightsail using `nifty_scalper_bot.main:app`")
        text = text.replace("Railway", "AWS Lightsail")
        _write(architecture, text)

    _write(
        "deploy/LIGHTSAIL_DEPLOYMENT.md",
        '''# AWS Lightsail production deployment\n\n## Authority\n\n- Host: one Ubuntu AWS Lightsail instance.\n- Service: `niftybot.service` under systemd.\n- Entrypoint: `python -m uvicorn nifty_scalper_bot.main:app --host 0.0.0.0 --port 8080`.\n- Source authority: `origin/main`.\n- Railway is not used.\n\n## Staged auto-deployment\n\n`deploy/lightsail_setup.sh` installs a two-minute systemd timer. A changed `main`\ncommit is first checked out into an isolated worktree. The deployment proceeds only\nafter Python compilation and the focused live-safety tests pass. The service is then\nrestarted and `/livez` is polled. If the API does not recover, the host resets to the\nprevious commit, reinstalls the editable package and restarts the prior release.\n\n## Live-entry defaults appended without overwriting operator settings\n\n- selected-option-only execution\n- fail-closed quote age/update-version checks\n- one real update minimum\n- persistent three-update OrderFlow reversal confirmation\n- 10,000-event dashboard ring\n\nProtective exits remain outside these new-entry gates.\n''',
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("quote", "runtime", "dashboard", "lightsail"), required=True)
    args = parser.parse_args()
    {"quote": stage_quote, "runtime": stage_runtime, "dashboard": stage_dashboard, "lightsail": stage_lightsail}[args.stage]()


if __name__ == "__main__":
    main()
