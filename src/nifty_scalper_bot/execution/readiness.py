"""Pure readiness helpers used by the live arming gate.

Keeping the readiness decision in a side-effect-free helper makes the live
trading arming logic easy to unit-test and ensures the same rules are applied
consistently from app startup, health endpoints, and supervisor checks.
"""

from __future__ import annotations


def compute_live_readiness(
    *,
    live_mode: bool,
    hard_ready: bool,
    quote_available: bool,
    ws_quote_proof: bool,
    market_open: bool,
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
    return (len(reasons) == 0), reasons
