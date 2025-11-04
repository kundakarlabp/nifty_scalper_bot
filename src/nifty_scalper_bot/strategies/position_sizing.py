"""Position sizing utilities for option strategies."""

from __future__ import annotations

from typing import Any, Mapping, cast

from nifty_scalper_bot.utils.logging import get_logger

logger = get_logger(__name__)


def calculate_position_size(
    *,
    account_balance: float,
    risk_pct: float,
    or_high: float,
    or_low: float,
    option_price: float,
    lot_size: int,
    tradingsymbol: str,
    market_data: Mapping[str, Any],
) -> int:
    """Calculate option quantity using live delta-based stop distance.

    Args:
        account_balance: Total deployable capital for sizing decisions.
        risk_pct: Percentage of balance risked per trade (0-100).
        or_high: Opening range high used for stop placement.
        or_low: Opening range low used for stop placement.
        option_price: Current option premium for entry sizing.
        lot_size: Exchange-mandated lot size for the contract.
        tradingsymbol: Identifier for the option contract.
        market_data: Snapshot containing greeks information.

    Returns:
        Integer quantity (multiple of lot_size) respecting risk constraints.

    Raises:
        ValueError: If input parameters are invalid.
    """

    logger.debug(
        "Entered calculate_position_size",
        extra={
            "event": "calculate_position_size_enter",
            "tradingsymbol": tradingsymbol,
        },
    )

    if account_balance <= 0:
        raise ValueError("account_balance must be positive")
    if risk_pct <= 0:
        raise ValueError("risk_pct must be positive")
    if lot_size <= 0:
        raise ValueError("lot_size must be positive")
    if option_price <= 0:
        raise ValueError("option_price must be positive")
    if or_high <= or_low:
        raise ValueError("or_high must be greater than or_low")

    try:
        greeks_snapshot = market_data.get("greeks") if market_data else None
        if not isinstance(greeks_snapshot, Mapping):
            logger.info(
                "Condition met: greeks_missing",
                extra={
                    "event": "calculate_position_size_missing_greeks",
                    "tradingsymbol": tradingsymbol,
                },
            )
            return 0
        normalized_symbol = (tradingsymbol or "").strip().upper()
        greeks_entry = greeks_snapshot.get(normalized_symbol)
        if not isinstance(greeks_entry, Mapping):
            logger.info(
                "Condition met: delta_missing",
                extra={
                    "event": "calculate_position_size_missing_delta",
                    "tradingsymbol": normalized_symbol,
                },
            )
            return 0
        raw_delta = greeks_entry.get("delta")
        if raw_delta is None:
            logger.info(
                "Condition met: delta_missing",
                extra={
                    "event": "calculate_position_size_missing_delta",
                    "tradingsymbol": normalized_symbol,
                },
            )
            return 0
        if not isinstance(raw_delta, (int, float, str)):
            logger.info(
                "Condition met: delta_invalid",
                extra={
                    "event": "calculate_position_size_invalid_delta",
                    "tradingsymbol": normalized_symbol,
                },
            )
            return 0
        try:
            delta = abs(float(cast(float | str, raw_delta)))
        except (TypeError, ValueError):
            logger.info(
                "Condition met: delta_invalid",
                extra={
                    "event": "calculate_position_size_invalid_delta",
                    "tradingsymbol": normalized_symbol,
                },
            )
            return 0
        stop_distance_option = (or_high - or_low) * delta
        if stop_distance_option <= 0:
            logger.info(
                "Condition met: stop_distance_non_positive",
                extra={
                    "event": "calculate_position_size_invalid_stop",
                    "tradingsymbol": normalized_symbol,
                    "stop_distance_option": stop_distance_option,
                },
            )
            return 0
        risk_amount = account_balance * (risk_pct / 100.0)
        if risk_amount <= 0:
            logger.info(
                "Condition met: risk_amount_non_positive",
                extra={
                    "event": "calculate_position_size_invalid_risk",
                    "tradingsymbol": normalized_symbol,
                    "risk_amount": risk_amount,
                },
            )
            return 0
        per_lot_risk = stop_distance_option * lot_size
        if per_lot_risk <= 0:
            logger.info(
                "Condition met: per_lot_risk_non_positive",
                extra={
                    "event": "calculate_position_size_invalid_per_lot",
                    "tradingsymbol": normalized_symbol,
                    "per_lot_risk": per_lot_risk,
                },
            )
            return 0
        max_lots = int(risk_amount // per_lot_risk)
        if max_lots <= 0:
            logger.info(
                "Condition met: risk_budget_insufficient",
                extra={
                    "event": "calculate_position_size_no_lots",
                    "tradingsymbol": normalized_symbol,
                    "risk_amount": risk_amount,
                    "per_lot_risk": per_lot_risk,
                },
            )
            return 0
        position_size = max_lots * lot_size
        logger.info(
            "Condition met: position_size_computed",
            extra={
                "event": "calculate_position_size_success",
                "tradingsymbol": normalized_symbol,
                "position_size": position_size,
                "delta": delta,
                "risk_amount": risk_amount,
                "per_lot_risk": per_lot_risk,
            },
        )
        return position_size
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failure in calculate_position_size: %s",
            exc,
            extra={
                "event": "calculate_position_size_error",
                "tradingsymbol": tradingsymbol,
            },
        )
        raise
