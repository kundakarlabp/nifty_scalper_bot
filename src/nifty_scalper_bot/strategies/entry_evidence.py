"""Canonical entry-evidence helpers.

This module contains objective execution evidence only; admission remains
structural and fail-closed: direction, setup validity, market-data freshness,
quote executability, post-cost economics and risk retain separate owners.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, cast

from nifty_scalper_bot.config.entry_policy import resolve_entry_policy


@dataclass(frozen=True, slots=True)
class ExecutionEvidence:
    """Objective option-execution evidence; never directional alpha."""

    contract_side: str
    direction_aligned: bool
    bid_ask_valid: bool
    spread_observed: bool
    spread_ok: bool | None
    spread_pct: float | None
    spread_limit_pct: float
    depth_valid: bool
    tradable_quote: bool
    stale_data_used: bool
    required_data_present: bool

    @property
    def executable(self) -> bool:
        """Return whether the observed quote is suitable for an order plan."""
        return bool(
            self.required_data_present
            and not self.stale_data_used
            and (self.tradable_quote or self.bid_ask_valid)
            and self.spread_ok is not False
        )

    def to_metadata(self) -> dict[str, object]:
        payload = asdict(self)
        payload["execution_evidence_pass"] = self.executable
        return payload


def infer_option_side(symbol: str, metadata: Mapping[str, object] | None = None) -> str:
    """Return the contract side without inferring underlying direction from premium."""
    upper = str(symbol or "").strip().upper()
    if upper.endswith("CE"):
        return "CE"
    if upper.endswith("PE"):
        return "PE"
    payload = dict(metadata or {})
    side = str(
        payload.get("contract_side")
        or payload.get("trade_side")
        or payload.get("side")
        or ""
    ).strip().upper()
    return side if side in {"CE", "PE"} else "UNKNOWN"


def resolve_signal_domain(
    symbol: str,
    metadata: Mapping[str, object] | None = None,
) -> tuple[str, bool, bool]:
    """Return (contract_side, option_premium_domain, underlying_domain)."""
    payload = dict(metadata or {})
    contract_side = infer_option_side(symbol, payload)
    source_symbol = str(payload.get("source_symbol") or "").strip().upper()
    option_symbol = str(symbol or "").upper().endswith(("CE", "PE"))
    option_premium_domain = bool(option_symbol and not source_symbol)
    underlying_domain = bool(source_symbol)
    return contract_side, option_premium_domain, underlying_domain


def canonical_max_spread_pct() -> float:
    """Return the single execution-spread limit."""
    return resolve_entry_policy().execution_max_spread_pct


def build_execution_evidence(
    indicators: Mapping[str, object] | None,
    *,
    side: str,
) -> dict[str, object]:
    """Build objective execution evidence from the current market snapshot."""
    payload = dict(indicators or {})
    resolved_side = str(side or "").strip().upper()
    direction = str(
        payload.get("underlying_direction_bias")
        or payload.get("direction_bias")
        or ""
    ).strip().upper()

    try:
        bid = float(cast(Any, payload.get("bid") or 0.0))
        ask = float(cast(Any, payload.get("ask") or 0.0))
    except (TypeError, ValueError):
        bid = ask = 0.0
    bid_ask_valid = bid > 0.0 and ask >= bid

    spread_observed = payload.get("spread_pct") is not None
    try:
        spread_pct = (
            float(cast(Any, payload.get("spread_pct")))
            if spread_observed
            else None
        )
    except (TypeError, ValueError):
        spread_pct = None
        spread_observed = False
    if not spread_observed and bid_ask_valid:
        midpoint = (bid + ask) / 2.0
        if midpoint > 0:
            spread_pct = ((ask - bid) / midpoint) * 100.0
            spread_observed = True

    spread_limit = canonical_max_spread_pct()
    spread_ok = (
        bool(spread_pct <= spread_limit)
        if spread_observed and spread_pct is not None
        else None
    )
    depth_valid = bool(payload.get("quote_depth_valid"))
    tradable_quote = bool(payload.get("tradable_quote"))
    stale_data_used = bool(payload.get("stale_data_used"))
    required_data_present = bool(payload.get("required_data_present", True))
    evidence = ExecutionEvidence(
        contract_side=resolved_side,
        direction_aligned=bool(
            resolved_side in {"CE", "PE"} and direction == resolved_side
        ),
        bid_ask_valid=bid_ask_valid,
        spread_observed=spread_observed,
        spread_ok=spread_ok,
        spread_pct=spread_pct,
        spread_limit_pct=spread_limit,
        depth_valid=depth_valid,
        tradable_quote=tradable_quote,
        stale_data_used=stale_data_used,
        required_data_present=required_data_present,
    )
    return evidence.to_metadata()


__all__ = [
    "ExecutionEvidence",
    "build_execution_evidence",
    "canonical_max_spread_pct",
    "infer_option_side",
    "resolve_signal_domain",
]
