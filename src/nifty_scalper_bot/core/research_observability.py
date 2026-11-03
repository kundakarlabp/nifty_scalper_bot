"""Bounded, read-only evidence for selected CE/PE parity and strategy no-votes.

Live evaluation owns no new tick writer: the canonical replay archive remains the
only high-frequency quote store. Samples are minute-bounded journald records,
shipped by the existing off-hot-path daily log archiver. Missing evidence never
constitutes a pass and cannot change signal or broker admission.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from nifty_scalper_bot.core.option_volume_diagnostics import (
    _minute_key,
    describe_option_volume_evidence,
)
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.symbols import normalize_symbol

LOGGER = get_logger(__name__)
_STRATEGIES = {"SMC", "VWAPPro", "ORBPro"}


def _completed(rows: Any) -> list[Mapping[str, Any]]:
    if not isinstance(rows, (tuple, list)):
        return []
    return [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("is_complete") is not False
        and row.get("is_provisional") is not True
        and _minute_key(row.get("timestamp") or row.get("start"))
    ]


def sample_selected_option_parity(
    owner: Any, *, at: datetime | None = None
) -> list[dict[str, Any]]:
    """Compare BOTH selected contracts on actual matching completed UTC minutes.

    Called after a strategy invocation. The one-sample-per-UTC-minute latch is
    independent of whether VWAP produced a vote or a zero-volume rejection.
    """
    hub = getattr(owner, "_data_hub", None)
    engine = getattr(owner, "_indicator_engine", None)
    selector = getattr(hub, "get_selected_option_symbols", None)
    history = getattr(engine, "get_history", None)
    canonical = getattr(hub, "get_ohlc_bars", None)
    if not all(callable(x) for x in (selector, history, canonical)):
        return []
    try:
        selected = selector()
        options = tuple(
            dict.fromkeys(
                normalize_symbol(str(symbol))
                for symbol in selected
                if symbol and normalize_symbol(str(symbol)).endswith(("CE", "PE"))
            )
        )
    except (TypeError, ValueError, AttributeError):
        return []
    if not options:
        return []

    now = at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        return []
    minute = now.astimezone(timezone.utc).replace(second=0, microsecond=0).isoformat()
    key = (minute, options)
    lock = getattr(owner, "_signal_observability_lock", None)
    if lock is None:
        return []
    with lock:
        if getattr(owner, "_research_parity_sample_key", None) == key:
            return []
        owner._research_parity_sample_key = key

    results: list[dict[str, Any]] = []
    for symbol in options:
        try:
            indicator_rows = _completed(history(symbol, count=5, field="bars"))
            canonical_rows = _completed(canonical(symbol, limit=5))
            indicator = indicator_rows[-1] if indicator_rows else None
            indicator_minute = (
                _minute_key(indicator.get("timestamp") or indicator.get("start"))
                if indicator is not None
                else None
            )
            same_minute_canonical = (
                next(
                    (
                        row
                        for row in reversed(canonical_rows)
                        if _minute_key(row.get("timestamp") or row.get("start"))
                        == indicator_minute
                    ),
                    None,
                )
                if indicator_minute is not None
                else None
            )
            reference = same_minute_canonical or (
                canonical_rows[-1] if canonical_rows else None
            )
            evidence = describe_option_volume_evidence(
                {"volume": indicator.get("volume") if indicator else None},
                latest_bar=indicator,
                indicator_completed_bar=indicator,
                canonical_completed_bar=reference,
            )
            status = evidence["completed_bar_parity"]
            if (
                status == "same_minute_volume_match"
                and (evidence["indicator_completed_volume"] or 0) <= 0
            ):
                status = "zero_volume_match_unverified"
            record = {
                "v": 1,
                "symbol": symbol,
                "checked_at": now.isoformat(),
                "status": status,
                "indicator_minute": evidence["indicator_completed_ts"],
                "canonical_minute": evidence["canonical_completed_ts"],
                "indicator_volume": evidence["indicator_completed_volume"],
                "canonical_volume": evidence["canonical_completed_volume"],
                "indicator_bars": len(indicator_rows),
                "canonical_bars": len(canonical_rows),
            }
        except Exception as exc:  # noqa: BLE001 - diagnostics cannot affect entry
            record = {
                "v": 1,
                "symbol": symbol,
                "checked_at": now.isoformat(),
                "status": "parity_observation_error",
                "error_type": type(exc).__name__,
            }
        LOGGER.info(
            "OPTION_COMPLETED_VOLUME_PARITY %s",
            json.dumps(record, separators=(",", ":"), allow_nan=False),
            extra={"event": "OPTION_COMPLETED_VOLUME_PARITY", "symbol": symbol},
        )
        results.append(record)
    return results


def record_strategy_rejection_sample(
    owner: Any,
    *,
    strategy: str,
    symbol: str,
    reason: str,
    indicators: Mapping[str, Any],
    at: datetime | None = None,
) -> dict[str, Any] | None:
    """Log the first no-vote per strategy, exact contract and UTC minute.

    An absent structural setup ID is NEVER replaced by a fake setup ID. These
    are minute-sampled evaluations, not independent SMC/VWAP/ORB setups.
    """
    if strategy not in _STRATEGIES:
        return None
    contract = normalize_symbol(str(symbol))
    if not contract.startswith("NFO:") or not contract.endswith(("CE", "PE")):
        return None
    now = at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        return None
    minute = now.astimezone(timezone.utc).replace(second=0, microsecond=0).isoformat()
    lock = getattr(owner, "_signal_observability_lock", None)
    if lock is None:
        return None
    with lock:
        if getattr(owner, "_research_rejections_minute", None) != minute:
            owner._research_rejections_minute = minute
            owner._research_rejections_seen = set()
        seen = owner._research_rejections_seen
        identity = (strategy, contract)
        if identity in seen:
            return None
        seen.add(identity)

    bar_time = indicators.get("latest_bar_ts") or indicators.get("bar_ts")
    record = {
        "v": 1,
        "sampling": "first_no_vote_per_contract_strategy_utc_minute",
        "decision_at": now.isoformat(),
        "evaluation_minute": minute,
        "strategy": strategy,
        "symbol": contract,
        "side": contract[-2:],
        "reason": str(reason or "none")[:100],
        "bar_ts": (
            bar_time.isoformat()
            if isinstance(bar_time, datetime)
            else str(bar_time)[:45] if bar_time else None
        ),
    }
    LOGGER.info(
        "RESEARCH_STRATEGY_REJECTION %s",
        json.dumps(record, separators=(",", ":"), allow_nan=False),
        extra={"event": "RESEARCH_STRATEGY_REJECTION", "symbol": contract},
    )
    return record
