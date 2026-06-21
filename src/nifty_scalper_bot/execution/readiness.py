"""Pure readiness helpers used by the live arming gate.

Keeping the readiness decision in a side-effect-free helper makes the live
trading arming logic easy to unit-test and ensures the same rules are applied
consistently from app startup, health endpoints, and supervisor checks.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import os
import logging
from datetime import datetime, timezone
from typing import Mapping

LOGGER = logging.getLogger(__name__)


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return max(default, minimum)
    try:
        return max(int(float(str(raw).strip())), minimum)
    except (TypeError, ValueError):
        LOGGER.warning(
            "INVALID_HISTORY_POLICY_ENV name=%s value=%r default=%s",
            name,
            raw,
            default,
            extra={"event": "INVALID_HISTORY_POLICY_ENV", "env_name": name, "value": raw, "default": default},
        )
        return max(default, minimum)


def _safe_positive_int(value: object, fallback: int) -> int:
    fallback_value = max(int(fallback), 1)
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return fallback_value
    if parsed <= 0:
        return fallback_value
    return parsed


def _safe_non_negative_int(value: object, fallback: int = 0) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return max(int(fallback), 0)
    return max(parsed, 0)


_READINESS_PRIORITY = [
    "emergency_stop_active",
    "kill_switch_active",
    "broker_auth_invalid",
    "broker_session_invalid",
    "broker_balance_unavailable",
    "position_reconciliation_failed",
    "position_reconciliation_incomplete",
    "unresolved_exit_position",
    "unprotected_broker_position",
    "risk_halt",
    "daily_loss_limit",
    "market_closed",
    "exchange_holiday",
    "outside_session",
    "broker_health_block",
    "futures_history_missing",
    "context_exec_not_ready",
    "selected_contract_missing",
    "selected_option_subscription_missing",
    "selected_option_quote_missing",
    "selected_option_depth_missing",
    "selected_option_history_cold",
    "strategy_not_ready",
    "order_manager_not_ready",
]

_BLOCKER_ALIASES = {
    "selected_ce_missing": "selected_contract_missing",
    "selected_pe_missing": "selected_contract_missing",
    "selected_options_missing": "selected_contract_missing",
    "selected_ce_quote_missing": "selected_option_quote_missing",
    "selected_pe_quote_missing": "selected_option_quote_missing",
    "selected_option_bid_ask_missing": "selected_option_quote_missing",
    "selected_ce_depth_missing": "selected_option_depth_missing",
    "selected_pe_depth_missing": "selected_option_depth_missing",
    "selected_ce_history_insufficient": "selected_option_history_cold",
    "selected_pe_history_insufficient": "selected_option_history_cold",
    "ce_eval_bars_missing": "selected_option_history_cold",
    "pe_eval_bars_missing": "selected_option_history_cold",
    "ce_exec_bars_missing": "selected_option_history_cold",
    "pe_exec_bars_missing": "selected_option_history_cold",
    "ce_exec_quote_or_history_not_ready": "selected_option_history_cold",
    "pe_exec_quote_or_history_not_ready": "selected_option_history_cold",
    "runner_not_running": "strategy_not_ready",
    "strategy_runner_not_running": "strategy_not_ready",
    "eval_not_ready": "strategy_not_ready",
    "broker_not_ready": "broker_health_block",
    "broker_connectivity_unknown": "broker_health_block",
    "selected_ce_subscription_not_live": "selected_option_subscription_missing",
    "selected_pe_subscription_not_live": "selected_option_subscription_missing",
}


@dataclass(frozen=True, slots=True)
class ReadinessDecision:
    generation: int = 0
    calculated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    primary_blocker: str | None = None
    blocker_list: tuple[str, ...] = ()
    secondary_blockers: tuple[str, ...] = ()
    live_orders_armed: bool = False
    evaluation_ready: bool = False
    execution_ready: bool = False
    broker_ready: bool = False
    reconciliation_ready: bool = False
    market_state: str = "unknown"
    human_reason: str = "not_evaluated"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if isinstance(self.calculated_at, datetime):
            payload["calculated_at"] = self.calculated_at.astimezone(timezone.utc).isoformat()
        payload["blocker_list"] = list(self.blocker_list)
        payload["secondary_blockers"] = list(self.secondary_blockers)
        return payload


def _normalize_market_state_name(market_state: object) -> str:
    value = getattr(market_state, "value", market_state)
    return str(value or "").strip().lower()


def _canonical_blocker(reason: object) -> str:
    text = str(reason or "").strip()
    if not text:
        return ""
    if ":" in text:
        text = text.split(":", 1)[-1]
    return _BLOCKER_ALIASES.get(text, text)


def normalize_readiness_blockers(
    blockers: list[str] | tuple[str, ...] | set[str],
    market_state: object = None,
    emergency_state: Mapping[str, object] | None = None,
    broker_state: Mapping[str, object] | None = None,
    risk_state: Mapping[str, object] | None = None,
    *,
    live_mode: bool = True,
    evaluation_ready: bool = False,
    execution_ready: bool = False,
) -> ReadinessDecision:
    """Apply deterministic readiness blocker priority and closed-market dominance."""

    canonical = [b for b in (_canonical_blocker(item) for item in blockers or []) if b]
    market_name = _normalize_market_state_name(market_state)
    emergency_state = emergency_state or {}
    broker_state = broker_state or {}
    risk_state = risk_state or {}
    if emergency_state.get("emergency_stop_active"):
        canonical.append("emergency_stop_active")
    if emergency_state.get("kill_switch_active"):
        canonical.append("kill_switch_active")
    if broker_state.get("broker_auth_invalid"):
        canonical.append("broker_auth_invalid")
    if broker_state.get("broker_session_invalid"):
        canonical.append("broker_session_invalid")
    if broker_state.get("broker_balance_unavailable") or broker_state.get("broker_balance_valid") is False:
        canonical.append("broker_balance_unavailable")
    if risk_state.get("risk_halt"):
        canonical.append("risk_halt")
    if risk_state.get("daily_loss_limit"):
        canonical.append("daily_loss_limit")
    if live_mode and market_name and market_name not in {"open", "marketstate.open"}:
        if market_name in {"holiday", "closed_holiday"}:
            canonical.append("exchange_holiday")
        elif market_name in {"preopen", "pre_market", "premarket"}:
            canonical.append("outside_session")
        else:
            canonical.append("market_closed")
    canonical = list(dict.fromkeys(canonical))

    high_priority = {"emergency_stop_active", "kill_switch_active", "broker_auth_invalid", "broker_session_invalid", "broker_balance_unavailable", "position_reconciliation_failed", "position_reconciliation_incomplete", "unresolved_exit_position", "unprotected_broker_position"}
    has_high_priority = any(item in canonical for item in high_priority)
    market_blocker = "exchange_holiday" if "exchange_holiday" in canonical else "outside_session" if "outside_session" in canonical else "market_closed" if "market_closed" in canonical else None
    secondary: list[str] = []
    visible = canonical
    if market_blocker and not has_high_priority:
        secondary = [item for item in canonical if item != market_blocker]
        visible = [market_blocker]

    def _rank(item: str) -> int:
        try:
            return _READINESS_PRIORITY.index(item)
        except ValueError:
            return len(_READINESS_PRIORITY)

    ordered_visible = sorted(list(dict.fromkeys(visible)), key=_rank)
    ordered_secondary = sorted(list(dict.fromkeys(secondary)), key=_rank)
    primary = ordered_visible[0] if ordered_visible else None
    armed = bool(live_mode and not primary and evaluation_ready and execution_ready)
    human = "ready" if primary is None else primary
    return ReadinessDecision(
        primary_blocker=primary,
        blocker_list=tuple(ordered_visible),
        secondary_blockers=tuple(ordered_secondary),
        live_orders_armed=armed,
        evaluation_ready=bool(evaluation_ready and not primary),
        execution_ready=bool(execution_ready and not primary),
        broker_ready=not any(item in canonical for item in {"broker_auth_invalid", "broker_session_invalid", "broker_balance_unavailable", "broker_health_block"}),
        reconciliation_ready=not any(item in canonical for item in {"position_reconciliation_failed", "position_reconciliation_incomplete", "unresolved_exit_position", "unprotected_broker_position"}),
        market_state=market_name or "unknown",
        human_reason=human,
    )


@dataclass(slots=True)
class HydrationStatus:
    """Canonical full-path hydration contract for startup/readiness gates."""

    symbol: str
    role: str
    token: int | None = None
    tradingsymbol: str | None = None
    exchange: str | None = None
    required_bars: int = 0
    historical_rows_returned: int = 0
    historical_rows_accepted: int = 0
    mdm_bars: int = 0
    datahub_bars: int = 0
    runner_bars: int = 0
    indicator_bars: int = 0
    live_tick_fresh: bool = False
    tradable_quote: bool = False
    depth_available: bool = False
    bid: float | None = None
    ask: float | None = None
    spread_pct: float | None = None
    ready_for_evaluation: bool = False
    ready_for_execution: bool = False
    blocker_reasons: list[str] = field(default_factory=list)
    last_historical_fetch_error: str | None = None
    last_historical_fetch_at: datetime | None = None
    first_bar_ts: datetime | None = None
    last_bar_ts: datetime | None = None
    live_merge_applied: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly hydration snapshot."""
        payload = asdict(self)
        for key in ("last_historical_fetch_at", "first_bar_ts", "last_bar_ts"):
            value = payload.get(key)
            if isinstance(value, datetime):
                payload[key] = value.astimezone(timezone.utc).isoformat()
        return payload


@dataclass(frozen=True)
class HistoryReadinessPolicy:
    option_eval_min_bars: int = 5
    option_entry_min_bars: int = 5
    context_min_bars: int = 50
    smc_min_bars: int = 30
    allow_synthetic_option_bars_for_eval: bool = False

    @classmethod
    def from_env(cls) -> "HistoryReadinessPolicy":
        return cls(
            option_eval_min_bars=_env_int("OPTION_EVAL_MIN_BARS", 5),
            option_entry_min_bars=_env_int("OPTION_ENTRY_MIN_BARS", 5),
            context_min_bars=_env_int("CONTEXT_MIN_BARS", 50),
            smc_min_bars=_env_int("SMC_MIN_BARS_REQUIRED", 30),
            allow_synthetic_option_bars_for_eval=str(
                os.getenv("ALLOW_SYNTHETIC_OPTION_BARS_FOR_EVAL", "false")
            ).strip().lower() in {"1", "true", "yes", "on"},
        )


def _quote_float(payload: dict | object, *keys: str) -> float | None:
    """Return first positive-compatible float field from a quote-like mapping."""
    getter = payload.get if isinstance(payload, dict) else lambda key, default=None: getattr(payload, key, default)
    for key in keys:
        value = getter(key, None)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number:
            return number
    return None


def resolve_quote_bid_ask_spread(quote: dict | object) -> tuple[float | None, float | None, float | None, str]:
    """Resolve bid/ask/spread from top-level fields or Zerodha depth.

    Fresh WebSocket FULL quotes are tradable when either top-level bid/ask or
    depth.buy[0]/depth.sell[0] contains a valid crossed-safe best bid/ask.
    """
    if quote is None:
        return None, None, None, "missing"
    getter = quote.get if isinstance(quote, dict) else lambda key, default=None: getattr(quote, key, default)
    bid: float | None = None
    ask: float | None = None
    source = "missing"

    raw_bid = _quote_float(quote, "bid", "bid_price")
    raw_ask = _quote_float(quote, "ask", "ask_price")
    if raw_bid is not None and raw_bid > 0 and raw_ask is not None and raw_ask > raw_bid:
        bid, ask, source = raw_bid, raw_ask, "top_level"

    if bid is None:
        raw_bid = _quote_float(quote, "best_bid", "best_bid_price")
        raw_ask = _quote_float(quote, "best_ask", "best_ask_price")
        if raw_bid is not None and raw_bid > 0 and raw_ask is not None and raw_ask > raw_bid:
            bid, ask, source = raw_bid, raw_ask, "best_bid_ask"

    if bid is None:
        depth = getter("depth", None)
        buy_levels = sell_levels = []
        if isinstance(depth, dict):
            buy_levels = depth.get("buy") or []
            sell_levels = depth.get("sell") or []
        buy_top = buy_levels[0] if isinstance(buy_levels, list) and buy_levels else {}
        sell_top = sell_levels[0] if isinstance(sell_levels, list) and sell_levels else {}
        raw_bid = _quote_float(buy_top, "price") if isinstance(buy_top, dict) else None
        raw_ask = _quote_float(sell_top, "price") if isinstance(sell_top, dict) else None
        if raw_bid is not None and raw_bid > 0 and raw_ask is not None and raw_ask > raw_bid:
            bid, ask, source = raw_bid, raw_ask, "depth"

    spread_pct: float | None = None
    if bid is not None and ask is not None and bid > 0 and ask > bid:
        mid = (bid + ask) / 2.0
        spread_pct = ((ask - bid) / mid) * 100.0
    else:
        precomputed = _quote_float(quote, "spread_pct")
        has_quote_proof = bool(getter("tradable_quote", False) or getter("depth_available", False) or getter("depth", None))
        if precomputed is not None and precomputed > 0 and has_quote_proof:
            spread_pct = precomputed
            if source == "missing":
                source = "derived_only"
    return bid, ask, spread_pct, source



@dataclass(frozen=True, slots=True)
class QuoteReadiness:
    """Canonical quote readiness split into LTP, depth, bid/ask, and tradable states."""

    symbol: str
    ltp_ready: bool
    depth_available: bool
    bid_ask_available: bool
    tradable_quote_ready: bool
    reason: str
    bid: float | None = None
    ask: float | None = None
    spread_pct: float | None = None
    source: str = "missing"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_quote_readiness(
    symbol: str,
    quote: dict | object | None,
    *,
    max_spread_pct: float | None = None,
    require_fresh: bool = True,
    max_age_s: float | None = None,
) -> QuoteReadiness:
    """Return canonical quote readiness for startup, runtime, execution, Telegram, and health.

    Tradable quote readiness is stricter than LTP readiness: bid and ask must be
    present, positive, non-crossed, fresh when age is known, and within the
    configured spread limit.
    """
    if quote is None:
        return QuoteReadiness(symbol=symbol, ltp_ready=False, depth_available=False, bid_ask_available=False, tradable_quote_ready=False, reason="quote_missing")
    getter = quote.get if isinstance(quote, dict) else lambda key, default=None: getattr(quote, key, default)
    ltp = _quote_float(quote, "ltp", "last_price", "last_traded_price")
    ltp_ready = bool(ltp is not None and ltp > 0)
    depth = getter("depth", None)
    depth_available = bool(getter("depth_available", False) or depth)
    bid, ask, spread_pct, source = resolve_quote_bid_ask_spread(quote)
    bid_ask_available = bool(bid is not None and ask is not None and bid > 0 and ask > bid)
    if not ltp_ready:
        reason = "ltp_missing"
    elif not bid_ask_available:
        raw_bid = _quote_float(quote, "bid", "bid_price", "best_bid", "best_bid_price")
        raw_ask = _quote_float(quote, "ask", "ask_price", "best_ask", "best_ask_price")
        if raw_bid is not None and raw_ask is not None and raw_bid > 0 and raw_ask > 0 and raw_ask <= raw_bid:
            reason = "bid_ask_crossed"
        else:
            reason = "bid_ask_missing"
    else:
        reason = "ready"
    if reason == "ready" and require_fresh:
        age = _quote_float(quote, "tick_age_s", "age_s")
        if age is None:
            ts = _quote_float(quote, "timestamp_ms", "last_tick_ts_ms")
            if ts and ts > 10_000_000_000:
                age = max(0.0, datetime.now(timezone.utc).timestamp() - (ts / 1000.0))
        if age is None:
            reason = "quote_age_unknown"
        elif max_age_s is not None and age > max_age_s:
            reason = "quote_stale"
    if reason == "ready" and max_spread_pct is not None and spread_pct is not None and spread_pct > max_spread_pct:
        reason = "spread_too_wide"
    return QuoteReadiness(
        symbol=symbol,
        ltp_ready=ltp_ready,
        depth_available=depth_available,
        bid_ask_available=bid_ask_available,
        tradable_quote_ready=reason == "ready",
        reason=reason,
        bid=bid,
        ask=ask,
        spread_pct=spread_pct,
        source=source,
    )

def quote_has_tradable_bid_ask(quote: dict | object) -> bool:
    """Return True when quote has valid top-level/depth bid-ask proof."""
    bid, ask, _spread, _source = resolve_quote_bid_ask_spread(quote)
    return bool(bid is not None and ask is not None and bid > 0 and ask > bid)


def compute_live_readiness(
    *,
    live_mode: bool,
    hard_ready: bool,
    quote_available: bool,
    ws_quote_proof: bool,
    market_open: bool,
    runner_running: bool,
    selected_ce: str | None = None,
    selected_pe: str | None = None,
    ce_bars: int = 0,
    pe_bars: int = 0,
    option_exec_min_bars: int = 0,
    ce_quote_ready: bool = False,
    pe_quote_ready: bool = False,
) -> tuple[bool, list[str]]:
    """Decide whether live trading should be armed.

    The bot may arm live trading when:
      * the configured execution mode is LIVE,
      * the data pipeline reports ``hard_ready``,
      * the market session is open, and
      * either REST quote capability OR a fresh WebSocket tradable-quote
        proof is available — the two are interchangeable here so a transient
        Zerodha 403 cannot block trading while WS is healthy.

    Args:
        live_mode: True when EXECUTION_MODE is LIVE / ENABLE_LIVE is set.
        hard_ready: MarketDataManager hard readiness flag.
        quote_available: Broker REST quote capability snapshot.
        ws_quote_proof: WebSocket tradable-quote proof flag.
        market_open: True when the exchange session is currently open.
        runner_running: True when StrategyRunner is actively running.

    Returns:
        Tuple ``(armed, reasons)`` where ``armed`` indicates whether live
        trading should be enabled and ``reasons`` lists every blocking
        condition (empty when armed).
    """

    if not live_mode:
        return False, ["not_live_mode"]
    market_data_proof = bool(quote_available or ws_quote_proof)
    reasons: list[str] = []
    if not hard_ready:
        reasons.append("startup_pipeline_incomplete")
    if not market_data_proof:
        reasons.append("market_data_proof_unavailable")
    if not market_open:
        reasons.append("market_closed")
    if not runner_running:
        reasons.append("strategy_runner_not_running")
    if not selected_ce or not selected_pe:
        reasons.append("selected_options_missing")
    policy = HistoryReadinessPolicy.from_env()
    min_bars = _safe_positive_int(option_exec_min_bars, policy.option_entry_min_bars)
    ce_count = _safe_non_negative_int(ce_bars, 0)
    pe_count = _safe_non_negative_int(pe_bars, 0)
    if ce_count < min_bars:
        reasons.append("selected_ce_history_insufficient")
    if pe_count < min_bars:
        reasons.append("selected_pe_history_insufficient")
    if not ce_quote_ready:
        reasons.append("selected_ce_quote_missing")
    if not pe_quote_ready:
        reasons.append("selected_pe_quote_missing")
    return (len(reasons) == 0), reasons
