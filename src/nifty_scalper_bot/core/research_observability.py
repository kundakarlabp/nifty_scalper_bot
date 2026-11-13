"""Bounded, read-only evidence for selected CE/PE parity and strategy no-votes.

Responsibilities: Preserve once-per-minute rejected SMC, VWAP and ORB
decision-time scalar evidence and selected-option volume parity in journald.
Canonical bar callback: observe_completed_selected_pair checks paired selected
closed minutes before calling sample_selected_option_parity; no strategy vote.
Boundaries: ReplayArchive remains the only high-frequency full-depth quote tape;
the existing daily log archiver mirrors samples without a second database.
Safety: Missing or nonfinite evidence is omitted, never assumed to pass, and
cannot change order admission, broker actions or protective risk constraints.
"""

from __future__ import annotations

import json
import math
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
# Frozen same-decision scalar context, never full books or credentials.
_REJECTION_CONTEXT_KEYS = (
    "ltp",
    "price",
    "close",
    "open",
    "high",
    "low",
    "bid",
    "ask",
    "spread_pct",
    "timestamp_ms",
    "last_tick_timestamp",
    "quote_update_version",
    "volume",
    "avg_volume",
    "atr",
    "vwap",
    "session_vwap",
    "exchange_vwap",
    "direction_bias",
    "underlying_direction_bias",
    "underlying_direction_confidence",
    "context_age_seconds",
    "regime",
    "futures_price",
    "futures_vwap",
    "futures_vwap_slope",
    "futures_volume_ratio",
    "ema_fast",
    "ema_slow",
    "bos_confirmed",
    "choch_confirmed",
    "retest_confirmed",
    "premium_reclaim",
    "selected_ce",
    "selected_pe",
    "is_selected_option",
    "lot_size",
    "stale_data_used",
    "data_age_seconds",
)


def _research_scalar(value: Any) -> Any:
    """Retain finite scalar metadata only; never arrays or arbitrary mappings."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:100]
    if isinstance(value, datetime):
        return value.isoformat()
    return None


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


def observe_completed_selected_pair(
    owner: Any, bar: Mapping[str, Any], mdm: Any
) -> list[dict[str, Any]]:
    """Audit both selected options on canonical bar completion, not strategy votes.

    Both contracts must have finalized the triggering minute. The normal
    parity sampler owns the once-per-minute latch; nothing here mutates
    the execution path or forces any additional market-data request.
    """
    hub = getattr(owner, "_data_hub", None)
    selector = getattr(hub, "get_selected_option_symbols", None)
    latest = getattr(mdm, "get_latest_closed_bar", None)
    if not callable(selector) or not callable(latest):
        return []
    try:
        selected = tuple(
            dict.fromkeys(
                normalize_symbol(str(value))
                for value in selector()
                if value and normalize_symbol(str(value)).endswith(("CE", "PE"))
            )
        )
        observed_symbol = normalize_symbol(str(bar.get("symbol") or ""))
        if len(selected) != 2 or observed_symbol not in selected:
            return []
        minute = _minute_key(bar.get("timestamp") or bar.get("start"))
        if minute is None:
            return []
        for option in selected:
            row = latest(option)
            if not isinstance(row, Mapping):
                return []
            if _minute_key(row.get("timestamp") or row.get("start")) != minute:
                return []
    except (AttributeError, TypeError, ValueError):
        return []
    return sample_selected_option_parity(owner)


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
    if not callable(selector) or not callable(history) or not callable(canonical):
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
            # The two completed-bar consumers advance on independent clocks:
            # the indicator may receive a bar before CandleEngine's flush grace
            # closes it. Compare the NEWEST COMMON UTC minute, not two tails
            # that happen to be read on opposite sides of that boundary.
            indicator_latest = indicator_rows[-1] if indicator_rows else None
            canonical_latest = canonical_rows[-1] if canonical_rows else None
            indicator_latest_minute = (
                _minute_key(
                    indicator_latest.get("timestamp") or indicator_latest.get("start")
                )
                if indicator_latest is not None
                else None
            )
            canonical_latest_minute = (
                _minute_key(
                    canonical_latest.get("timestamp") or canonical_latest.get("start")
                )
                if canonical_latest is not None
                else None
            )
            canonical_by_minute = {
                _minute_key(row.get("timestamp") or row.get("start")): row
                for row in canonical_rows
            }
            common_indicator = next(
                (
                    row
                    for row in reversed(indicator_rows)
                    if _minute_key(row.get("timestamp") or row.get("start"))
                    in canonical_by_minute
                ),
                None,
            )
            indicator = common_indicator or indicator_latest
            compared_minute = (
                _minute_key(indicator.get("timestamp") or indicator.get("start"))
                if indicator is not None
                else None
            )
            reference = (
                canonical_by_minute[compared_minute]
                if common_indicator is not None
                else canonical_latest
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
                "latest_indicator_minute": indicator_latest_minute,
                "latest_canonical_minute": canonical_latest_minute,
                "latest_sources_aligned": (
                    indicator_latest_minute == canonical_latest_minute
                    and indicator_latest_minute is not None
                ),
                "compared_common_minute": (
                    compared_minute if common_indicator is not None else None
                ),
                "comparison_used_older_common_minute": (
                    common_indicator is not None
                    and (
                        compared_minute != indicator_latest_minute
                        or compared_minute != canonical_latest_minute
                    )
                ),
                "volume_difference": (
                    evidence["indicator_completed_volume"]
                    - evidence["canonical_completed_volume"]
                    if evidence["indicator_completed_volume"] is not None
                    and evidence["canonical_completed_volume"] is not None
                    and evidence["indicator_completed_ts"]
                    == evidence["canonical_completed_ts"]
                    else None
                ),
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
        "context": {
            key: scalar
            for key in _REJECTION_CONTEXT_KEYS
            if key in indicators
            if (scalar := _research_scalar(indicators[key])) is not None
        },
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
