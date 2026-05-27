"""Pure readiness helpers used by the live arming gate.

Keeping the readiness decision in a side-effect-free helper makes the live
trading arming logic easy to unit-test and ensures the same rules are applied
consistently from app startup, health endpoints, and supervisor checks.
"""

from __future__ import annotations
from dataclasses import dataclass
import os
import logging

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
            extra={"event": "INVALID_HISTORY_POLICY_ENV", "name": name, "value": raw, "default": default},
        )
        return max(default, minimum)


def _safe_positive_int(value: object, fallback: int) -> int:
    try:
        return max(int(float(value)), 1)
    except (TypeError, ValueError):
        return max(int(fallback), 1)


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
    min_bars = _safe_positive_int(option_exec_min_bars, HistoryReadinessPolicy.from_env().option_entry_min_bars)
    if int(ce_bars) < min_bars:
        reasons.append("selected_ce_history_insufficient")
    if int(pe_bars) < min_bars:
        reasons.append("selected_pe_history_insufficient")
    if not ce_quote_ready:
        reasons.append("selected_ce_quote_missing")
    if not pe_quote_ready:
        reasons.append("selected_pe_quote_missing")
    return (len(reasons) == 0), reasons
