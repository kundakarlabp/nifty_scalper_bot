"""Readiness helpers used by the live arming gate.

The decision functions keep arming semantics deterministic while emitting compact,
rate-controlled diagnostics for operator validation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Mapping

from nifty_scalper_bot.config.env_utils import parse_float_env

LOGGER = logging.getLogger(__name__)
_UNUSABLE_QUOTE_TIMESTAMP_QUALITIES = {"synthetic", "unknown", "invalid"}
_TRUTHY = {"1", "true", "yes", "y", "on"}


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
            extra={
                "event": "INVALID_HISTORY_POLICY_ENV",
                "env_name": name,
                "value": raw,
                "default": default,
            },
        )
        return max(default, minimum)


def _safe_positive_int(value: object, fallback: int) -> int:
    fallback_value = max(int(fallback), 1)
    try:
        parsed = int(float(str(value)))
    except (TypeError, ValueError):
        return fallback_value
    if parsed <= 0:
        return fallback_value
    return parsed


def _safe_non_negative_int(value: object, fallback: int = 0) -> int:
    try:
        parsed = int(float(str(value)))
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
    "selected_ce_history_cold",
    "selected_pe_history_cold",
    "spot_history_cold",
    "futures_history_cold",
    "history_token_missing",
    "history_broker_error",
    "history_authentication_failed",
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
    "selected_ce_history_insufficient": "selected_ce_history_cold",
    "selected_pe_history_insufficient": "selected_pe_history_cold",
    "ce_eval_bars_missing": "selected_ce_history_cold",
    "pe_eval_bars_missing": "selected_pe_history_cold",
    "ce_exec_bars_missing": "selected_ce_history_cold",
    "pe_exec_bars_missing": "selected_pe_history_cold",
    "ce_exec_quote_or_history_not_ready": "selected_ce_history_cold",
    "pe_exec_quote_or_history_not_ready": "selected_pe_history_cold",
    "runner_not_running": "strategy_not_ready",
    "strategy_runner_not_running": "strategy_not_ready",
    "eval_not_ready": "strategy_not_ready",
    "broker_not_ready": "broker_health_block",
}


@dataclass(frozen=True, slots=True)
class ReadinessDecision:
    primary_blocker: str | None
    blocker_list: list[str]
    live_orders_armed: bool
    evaluation_ready: bool
    execution_ready: bool
    human_reason: str
    secondary_blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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


def _checklist_logging_enabled() -> bool:
    raw = os.getenv("LIVE_VALIDATION_CHECKLIST_LOGS", "true")
    return str(raw or "true").strip().lower() in _TRUTHY


def build_live_validation_checklist(
    decision: ReadinessDecision,
    *,
    market_name: str,
    canonical_blockers: list[str],
    emergency_state: Mapping[str, object],
    broker_state: Mapping[str, object],
    risk_state: Mapping[str, object],
    live_mode: bool,
    evaluation_ready: bool,
    execution_ready: bool,
) -> dict[str, object]:
    """Return compact operator checklist for live arming validation."""

    market_open = market_name in {"open", "marketstate.open"}
    broker_auth_ok = not bool(
        broker_state.get("broker_auth_invalid")
        or broker_state.get("broker_session_invalid")
    )
    broker_balance_ok = not bool(
        broker_state.get("broker_balance_unavailable")
        or broker_state.get("broker_balance_valid") is False
    )
    emergency_clear = not bool(
        emergency_state.get("emergency_stop_active")
        or emergency_state.get("kill_switch_active")
    )
    risk_green = not bool(
        risk_state.get("risk_halt") or risk_state.get("daily_loss_limit")
    )
    return {
        "live_mode": bool(live_mode),
        "market_open": bool(market_open),
        "evaluation_ready": bool(evaluation_ready),
        "execution_ready": bool(execution_ready),
        "broker_auth_ok": bool(broker_auth_ok),
        "broker_balance_ok": bool(broker_balance_ok),
        "emergency_clear": bool(emergency_clear),
        "risk_green": bool(risk_green),
        "primary_blocker": decision.primary_blocker,
        "blockers": list(decision.blocker_list),
        "secondary_blockers": list(decision.secondary_blockers),
        "raw_blockers": list(dict.fromkeys(canonical_blockers)),
        "live_orders_armed": bool(decision.live_orders_armed),
    }


def _emit_live_validation_checklist(
    decision: ReadinessDecision,
    *,
    market_name: str,
    canonical_blockers: list[str],
    emergency_state: Mapping[str, object],
    broker_state: Mapping[str, object],
    risk_state: Mapping[str, object],
    live_mode: bool,
    evaluation_ready: bool,
    execution_ready: bool,
) -> None:
    if not live_mode or not _checklist_logging_enabled():
        return
    checklist = build_live_validation_checklist(
        decision,
        market_name=market_name,
        canonical_blockers=canonical_blockers,
        emergency_state=emergency_state,
        broker_state=broker_state,
        risk_state=risk_state,
        live_mode=live_mode,
        evaluation_ready=evaluation_ready,
        execution_ready=execution_ready,
    )
    LOGGER.info(
        "LIVE_VALIDATION_CHECKLIST live_mode=%s market_open=%s "
        "evaluation_ready=%s execution_ready=%s broker_auth_ok=%s "
        "broker_balance_ok=%s emergency_clear=%s risk_green=%s "
        "live_orders_armed=%s primary_blocker=%s",
        checklist["live_mode"],
        checklist["market_open"],
        checklist["evaluation_ready"],
        checklist["execution_ready"],
        checklist["broker_auth_ok"],
        checklist["broker_balance_ok"],
        checklist["emergency_clear"],
        checklist["risk_green"],
        checklist["live_orders_armed"],
        checklist["primary_blocker"],
        extra={"event": "LIVE_VALIDATION_CHECKLIST", **checklist},
    )


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
    if (
        broker_state.get("broker_balance_unavailable")
        or broker_state.get("broker_balance_valid") is False
    ):
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

    high_priority = {
        "emergency_stop_active",
        "kill_switch_active",
        "broker_auth_invalid",
        "broker_session_invalid",
        "broker_balance_unavailable",
        "position_reconciliation_failed",
        "position_reconciliation_incomplete",
        "unresolved_exit_position",
        "unprotected_broker_position",
    }
    has_high_priority = any(item in canonical for item in high_priority)
    market_blocker = (
        "exchange_holiday"
        if "exchange_holiday" in canonical
        else (
            "outside_session"
            if "outside_session" in canonical
            else "market_closed" if "market_closed" in canonical else None
        )
    )
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
    decision = ReadinessDecision(
        primary_blocker=primary,
        blocker_list=ordered_visible,
        secondary_blockers=ordered_secondary,
        live_orders_armed=armed,
        evaluation_ready=bool(evaluation_ready and not primary),
        execution_ready=bool(execution_ready and not primary),
        human_reason=human,
    )
    _emit_live_validation_checklist(
        decision,
        market_name=market_name,
        canonical_blockers=canonical,
        emergency_state=emergency_state,
        broker_state=broker_state,
        risk_state=risk_state,
        live_mode=live_mode,
        evaluation_ready=evaluation_ready,
        execution_ready=execution_ready,
    )
    return decision


@dataclass(slots=True)
class HydrationStatus:
    """Canonical full-path hydration contract for startup/readiness gates."""

    symbol: str
    role: str
    token: int | None = None
    tradingsymbol: str | None = None
    exchange: str | None = None
    required_bars: int = 0
    historical_rows_returned: int = (
        0  # Back-compat: latest broker fetch row count, not cache size.
    )
    historical_rows_accepted: int = (
        0  # Back-compat: newly imported rows, not cache size.
    )
    fetch_returned_rows: int = 0
    import_accepted_new_rows: int = 0
    import_idempotent_rows: int = 0
    validation_rejected_rows: int = 0
    final_cache_rows: int = 0
    latest_import_status: str | None = None
    latest_import_reason: str | None = None
    latest_import_error: str | None = None
    latest_import_at: datetime | None = None
    history_provider_error: str | None = None
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
    expected_latest_closed_ts: datetime | None = None
    latest_bar_age_seconds: float | None = None
    latest_bar_fresh: bool = True
    recent_window_contiguous: bool = True
    missing_expected_minute_count: int = 0
    largest_intraday_gap_minutes: int = 0
    propagation_consistent: bool = False
    live_merge_applied: bool = False

    def __post_init__(self) -> None:
        if (
            self.role not in {"selected_ce", "selected_pe"}
            or not self.ready_for_execution
        ):
            return
        bid_ask_valid = bool(
            self.bid is not None
            and self.ask is not None
            and self.bid > 0
            and self.ask > self.bid
        )
        if bid_ask_valid:
            return
        self.ready_for_execution = False
        blocker = f"{self.role}_quote_missing"
        if blocker not in self.blocker_reasons:
            self.blocker_reasons.append(blocker)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly hydration snapshot."""
        payload = asdict(self)
        for key in (
            "last_historical_fetch_at",
            "latest_import_at",
            "first_bar_ts",
            "last_bar_ts",
            "expected_latest_closed_ts",
        ):
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
            )
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
        )


def _quote_getter(payload: dict | object):
    return (
        payload.get
        if isinstance(payload, dict)
        else lambda key, default=None: getattr(payload, key, default)
    )


def quote_timestamp_quality_allows_hard_readiness(quote: dict | object | None) -> bool:
    """Return False when a quote explicitly carries synthetic/invalid time proof.

    Missing timestamp_quality remains backward-compatible; explicit unusable
    quality is treated as hard-negative for bid/ask readiness and live arming.
    """

    if quote is None:
        return True
    getter = _quote_getter(quote)
    quality = str(getter("timestamp_quality", "") or "").strip().lower()
    return quality not in _UNUSABLE_QUOTE_TIMESTAMP_QUALITIES


def _quote_float(payload: dict | object, *keys: str) -> float | None:
    """Return first positive-compatible float field from a quote-like mapping."""
    getter = _quote_getter(payload)
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


def resolve_max_quote_age_seconds(
    seconds_env: str,
    legacy_ms_env: str,
    *,
    default_seconds: float,
) -> float:
    """Resolve quote max-age config once into canonical seconds.

    Prefer the seconds setting. Fall back to a legacy millisecond setting for
    deployment compatibility. Malformed/commented values safely use defaults.
    """
    raw_seconds = os.getenv(seconds_env)
    if raw_seconds is not None and raw_seconds.strip():
        return max(
            0.0,
            parse_float_env(raw_seconds, default_seconds),
        )
    legacy_default_ms = default_seconds * 1000.0
    legacy_ms = parse_float_env(
        os.getenv(legacy_ms_env),
        legacy_default_ms,
    )
    return max(0.0, legacy_ms / 1000.0)


def resolve_quote_age_seconds(payload: dict | object) -> float | None:
    age_ms = _quote_float(
        payload,
        "tick_age_ms",
        "quote_age_ms",
        "last_tick_age_ms",
        "market_data_age_ms",
    )
    if age_ms is not None:
        return max(0.0, age_ms / 1000.0)
    age_s = _quote_float(
        payload,
        "tick_age_s",
        "quote_age_s",
        "data_age_seconds",
        "age_s",
        "age_seconds",
        "last_tick_age_s",
        "market_data_age_s",
    )
    if age_s is not None:
        return max(0.0, age_s)
    return None


def resolve_quote_bid_ask_spread(
    quote: dict | object,
) -> tuple[float | None, float | None, float | None, str]:
    """Resolve bid/ask/spread from top-level fields or Zerodha depth.

    Fresh WebSocket FULL quotes are tradable when either top-level bid/ask or
    depth.buy[0]/depth.sell[0] contains a valid crossed-safe best bid/ask.
    Quotes explicitly tagged synthetic/unknown/invalid cannot prove readiness.
    """
    if quote is None:
        return None, None, None, "missing"
    if not quote_timestamp_quality_allows_hard_readiness(quote):
        return None, None, None, "timestamp_quality_unusable"
    getter = _quote_getter(quote)
    bid: float | None = None
    ask: float | None = None
    source = "missing"

    raw_bid = _quote_float(quote, "bid", "bid_price")
    raw_ask = _quote_float(quote, "ask", "ask_price")
    if (
        raw_bid is not None
        and raw_bid > 0
        and raw_ask is not None
        and raw_ask > raw_bid
    ):
        bid, ask, source = raw_bid, raw_ask, "top_level"

    if bid is None:
        raw_bid = _quote_float(quote, "best_bid", "best_bid_price")
        raw_ask = _quote_float(quote, "best_ask", "best_ask_price")
        if (
            raw_bid is not None
            and raw_bid > 0
            and raw_ask is not None
            and raw_ask > raw_bid
        ):
            bid, ask, source = raw_bid, raw_ask, "best_bid_ask"

    if bid is None:
        depth = getter("depth", None)
        buy_levels: list[object] = []
        sell_levels: list[object] = []
        if isinstance(depth, dict):
            buy_levels = list(depth.get("buy") or [])
            sell_levels = list(depth.get("sell") or [])
        buy_top = buy_levels[0] if isinstance(buy_levels, list) and buy_levels else {}
        sell_top = (
            sell_levels[0] if isinstance(sell_levels, list) and sell_levels else {}
        )
        raw_bid = _quote_float(buy_top, "price") if isinstance(buy_top, dict) else None
        raw_ask = (
            _quote_float(sell_top, "price") if isinstance(sell_top, dict) else None
        )
        if (
            raw_bid is not None
            and raw_bid > 0
            and raw_ask is not None
            and raw_ask > raw_bid
        ):
            bid, ask, source = raw_bid, raw_ask, "depth"

    spread_pct: float | None = None
    if bid is not None and ask is not None and bid > 0 and ask > bid:
        mid = (bid + ask) / 2.0
        spread_pct = ((ask - bid) / mid) * 100.0
    else:
        precomputed = _quote_float(quote, "spread_pct")
        has_quote_proof = bool(
            getter("tradable_quote", False)
            or getter("depth_available", False)
            or getter("depth", None)
        )
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
    """Return canonical quote readiness for runtime, execution, and health.

    Tradable quote readiness is stricter than LTP readiness: bid and ask must be
    present, positive, non-crossed, fresh when age is known, and within the
    configured spread limit.
    """
    if quote is None:
        return QuoteReadiness(
            symbol=symbol,
            ltp_ready=False,
            depth_available=False,
            bid_ask_available=False,
            tradable_quote_ready=False,
            reason="quote_missing",
        )
    getter = _quote_getter(quote)
    ltp = _quote_float(quote, "ltp", "last_price", "last_traded_price")
    ltp_ready = bool(ltp is not None and ltp > 0)
    depth = getter("depth", None)
    depth_available = bool(getter("depth_available", False) or depth)
    bid, ask, spread_pct, source = resolve_quote_bid_ask_spread(quote)
    bid_ask_available = bool(
        bid is not None and ask is not None and bid > 0 and ask > bid
    )
    timestamp_quality_ok = quote_timestamp_quality_allows_hard_readiness(quote)
    if not ltp_ready:
        reason = "ltp_missing"
    elif not timestamp_quality_ok:
        reason = "timestamp_quality_unusable"
    elif not bid_ask_available:
        raw_bid = _quote_float(quote, "bid", "bid_price", "best_bid", "best_bid_price")
        raw_ask = _quote_float(quote, "ask", "ask_price", "best_ask", "best_ask_price")
        if (
            raw_bid is not None
            and raw_ask is not None
            and raw_bid > 0
            and raw_ask > 0
            and raw_ask <= raw_bid
        ):
            reason = "bid_ask_crossed"
        else:
            reason = "bid_ask_missing"
    else:
        reason = "ready"
    if reason == "ready" and require_fresh:
        age = resolve_quote_age_seconds(quote)
        if age is None:
            ts = _quote_float(quote, "timestamp_ms", "last_tick_ts_ms")
            if ts and ts > 10_000_000_000:
                age = max(0.0, datetime.now(timezone.utc).timestamp() - (ts / 1000.0))
        if age is None:
            reason = "quote_age_unknown"
        elif max_age_s is not None and age > max_age_s:
            reason = "quote_stale"
    if (
        reason == "ready"
        and max_spread_pct is not None
        and spread_pct is not None
        and spread_pct > max_spread_pct
    ):
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
