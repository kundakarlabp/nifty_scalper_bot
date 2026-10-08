from __future__ import annotations

import os
import time
import zlib
from typing import Any, Mapping

from nifty_scalper_bot.execution.quote_readiness import evaluate_execution_quote
from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    OrderFlowStrategyConfig,
)
from nifty_scalper_bot.strategies.entry_evidence import resolve_signal_domain
from nifty_scalper_bot.strategies.runtime_context_contract import (
    resolve_context_age_seconds,
)
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


def safe_float_env(name: str, default: float) -> float:
    from nifty_scalper_bot.config.env_utils import parse_float_env

    return parse_float_env(os.getenv(name), default)


def _depth_supports_side(
    depth_imbalance: float,
    *,
    side: str,
    option_premium_domain: bool,
    threshold: float,
) -> bool:
    """Return whether signed book pressure supports the candidate side.

    For an option premium book, both CE and PE are long-premium candidates, so
    positive bid-side pressure supports the option while negative pressure is
    adverse. For an underlying-domain book, CE/PE retain directional sign.
    """
    threshold = max(0.0, float(threshold))
    if option_premium_domain:
        return depth_imbalance >= threshold
    return bool(
        (side == "CE" and depth_imbalance >= threshold)
        or (side == "PE" and depth_imbalance <= -threshold)
    )


def _normalised_depth_thresholds(
    config: OrderFlowStrategyConfig,
) -> tuple[float, float]:
    """Return canonical support/strong-support thresholds from strategy config."""
    support = max(0.05, min(0.50, float(config.large_order_threshold_pct) / 100.0))
    ratio = max(1.0, float(config.imbalance_ratio_min))
    ratio_threshold = (ratio - 1.0) / (ratio + 1.0) if ratio > 1.0 else 0.0
    strong = max(support, min(0.85, ratio_threshold))
    return support, strong


def _depth_conflicts_side(
    depth_imbalance: float,
    *,
    side: str,
    option_premium_domain: bool,
    threshold: float,
) -> bool:
    """Return whether signed book pressure is materially adverse to the side."""
    threshold = max(0.0, float(threshold))
    if option_premium_domain:
        return depth_imbalance <= -threshold
    return bool(
        (side == "CE" and depth_imbalance <= -threshold)
        or (side == "PE" and depth_imbalance >= threshold)
    )


def _safe_float_value(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _stable_quote_version(value: Any) -> int:
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        numeric = 0
    if numeric > 0:
        return numeric
    return int(zlib.crc32(str(value).encode("utf-8")) & 0x7FFFFFFF) or 1


def _stamp_quote_update_identity(
    metadata: dict[str, Any], indicators: Mapping[str, Any]
) -> None:
    """Preserve a real quote version or stable microstructure fingerprint."""
    for source in (metadata, indicators):
        for key in (
            "quote_update_version",
            "update_version",
            "tick_version",
            "last_tick_ts_ms",
            "timestamp_ms",
            "last_tick_timestamp",
        ):
            value = source.get(key)
            if value not in (None, "", 0, 0.0):
                metadata["quote_update_version"] = _stable_quote_version(value)
                metadata.setdefault("quote_update_version_source", key)
                return

    bid = _safe_float_value(metadata.get("bid") or indicators.get("bid"))
    ask = _safe_float_value(metadata.get("ask") or indicators.get("ask"))
    imbalance = _safe_float_value(
        metadata.get("depth_imbalance") or indicators.get("depth_imbalance")
    )
    tick_direction = str(
        metadata.get("tick_direction") or indicators.get("tick_direction") or ""
    ).upper()
    if bid is None and ask is None and imbalance is None and not tick_direction:
        return
    raw = (
        f"{bid if bid is not None else 'na'}:"
        f"{ask if ask is not None else 'na'}:"
        f"{imbalance if imbalance is not None else 'na'}:{tick_direction or 'na'}"
    )
    metadata["quote_update_version"] = _stable_quote_version(raw)
    metadata["quote_update_version_source"] = "microstructure_fingerprint"


class OrderFlowStrategy(EliteStrategy):
    """Order-flow vote using spread, depth imbalance and tick direction."""

    MIN_BARS_REQUIRED = 5

    def __init__(self, config: OrderFlowStrategyConfig, indicator_engine: Any) -> None:
        """Args: config, indicator_engine. Returns: None. Raises: Exception."""
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config
        # Only one owner tracks whether a live tick-only confirmation persists.
        # Temporal OFI already measures persistence upstream.
        self._last_live_depth_support: dict[str, tuple[int, float]] = {}

    def get_required_indicators(self) -> set[str]:
        """Args: none. Returns: indicator keys. Raises: Exception."""
        return {
            "bid",
            "ask",
            "depth",
            "tick_direction",
            "buy_qty",
            "sell_qty",
            "direction_bias",
            "spread_pct",
            "atr",
        }

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: dict[str, Any],
        current_price: float,
        position: Any | None = None,
    ) -> EliteSignal | None:
        """Return fresh microstructure context; OrderFlow never owns entries."""
        del position
        try:
            self._no_vote("stale_or_invalid_data")
            if current_price <= 0 or bool(indicators.get("stale_data_used")):
                return None

            bid = float(indicators.get("bid") or 0.0)
            ask = float(indicators.get("ask") or 0.0)
            if bid <= 0.0 or ask <= bid:
                self._no_vote("missing_bid_ask")
                return None
            spread_pct = float(
                indicators.get("spread_pct")
                or (((ask - bid) / ((ask + bid) / 2.0)) * 100.0)
            )

            depth = indicators.get("depth") or {}
            bids = depth.get("buy", []) if isinstance(depth, dict) else []
            asks = depth.get("sell", []) if isinstance(depth, dict) else []
            depth_available = bool(bids and asks)
            total_bid = (
                sum(float(level.get("quantity", 0.0)) for level in bids[:5])
                if depth_available
                else 0.0
            )
            total_ask = (
                sum(float(level.get("quantity", 0.0)) for level in asks[:5])
                if depth_available
                else 0.0
            )
            if total_bid + total_ask <= 0.0:
                self._no_vote("missing_depth")
                return None

            execution_mode = (
                str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper()
            )
            is_live = execution_mode == "LIVE"
            max_spread_pct = safe_float_env(
                "ORDERFLOW_CONTEXT_MAX_SPREAD_PCT",
                0.75 if is_live else 12.0,
            )
            max_tick_age_ms = safe_float_env("LIVE_MAX_TICK_AGE_MS", 2500.0)
            quote_payload = dict(indicators)
            quote_payload.update(
                {
                    "bid": bid,
                    "ask": ask,
                    "depth": depth,
                    "depth_available": depth_available,
                    "spread_pct": spread_pct,
                }
            )
            quote_readiness = evaluate_execution_quote(
                symbol,
                quote_payload,
                live_mode=is_live,
                max_tick_age_ms=max_tick_age_ms,
                max_spread_pct=max_spread_pct,
                require_depth=True,
            )
            # OrderFlow is context-only. Preserve observable evidence even when
            # execution quote readiness fails; the producer-owned
            # context_quality_eligible flag below prevents unusable context from
            # confirming an entry.
            contract_side, option_premium_domain, _ = resolve_signal_domain(
                symbol, indicators
            )
            direction = str(
                indicators.get("underlying_direction_bias")
                or indicators.get("direction_bias")
                or ""
            ).upper()
            if option_premium_domain and contract_side not in {"CE", "PE"}:
                self._no_vote("unknown_contract_side")
                return None

            depth_imbalance = (total_bid - total_ask) / max(total_bid + total_ask, 1.0)
            side = (
                contract_side
                if option_premium_domain
                else ("CE" if depth_imbalance > 0 else "PE")
            )
            support_threshold, strong_support_threshold = _normalised_depth_thresholds(
                self._cfg
            )
            depth_supports_side = _depth_supports_side(
                depth_imbalance,
                side=side,
                option_premium_domain=option_premium_domain,
                threshold=support_threshold,
            )
            strong_depth_supports_side = _depth_supports_side(
                depth_imbalance,
                side=side,
                option_premium_domain=option_premium_domain,
                threshold=strong_support_threshold,
            )
            depth_conflicts_side = _depth_conflicts_side(
                depth_imbalance,
                side=side,
                option_premium_domain=option_premium_domain,
                threshold=support_threshold,
            )
            strong_depth_conflicts_side = _depth_conflicts_side(
                depth_imbalance,
                side=side,
                option_premium_domain=option_premium_domain,
                threshold=strong_support_threshold,
            )

            tick_direction = str(indicators.get("tick_direction") or "").upper()
            tick_supports_side = (
                tick_direction in {"UP", "BUY"}
                if option_premium_domain
                else (
                    (side == "CE" and tick_direction in {"UP", "BUY"})
                    or (side == "PE" and tick_direction in {"DOWN", "SELL"})
                )
            )
            ofi_ready = bool(indicators.get("ofi_ready"))
            ofi_value = _safe_float_value(indicators.get("ofi_1s_normalized"))
            ofi_threshold = max(
                0.01, safe_float_env("ORDERFLOW_OFI_NORMALIZED_MIN", 0.10)
            )
            ofi_directional = bool(
                ofi_ready and ofi_value is not None and abs(ofi_value) >= ofi_threshold
            )
            ofi_supports_side = bool(
                ofi_directional
                and _depth_supports_side(
                    float(ofi_value or 0.0),
                    side=side,
                    option_premium_domain=option_premium_domain,
                    threshold=ofi_threshold,
                )
            )
            ofi_conflicts_side = bool(ofi_directional and not ofi_supports_side)
            tick_directional = tick_direction in {"UP", "BUY", "DOWN", "SELL"}
            tick_conflicts_side = bool(
                not ofi_directional and tick_directional and not tick_supports_side
            )
            flow_supports_side = (
                ofi_supports_side if ofi_directional else tick_supports_side
            )
            flow_conflicts_side = ofi_conflicts_side or tick_conflicts_side
            flow_confirmation_source = (
                "temporal_ofi" if ofi_directional else "tick_direction"
            )

            context_age_seconds = resolve_context_age_seconds(indicators)
            max_context_age = safe_float_env("ORDERFLOW_MAX_CONTEXT_AGE_SECONDS", 5.0)
            context_fresh = context_age_seconds <= max_context_age and not bool(
                indicators.get("stale_data_used")
            )
            direction_available = direction in {"CE", "PE"}
            side_aligns = bool(direction_available and direction == side)

            # Context never overturns the canonical underlying direction. It can
            # confirm a same-side setup or explicitly conflict with it.
            context_quality_eligible = bool(
                quote_readiness.allowed
                and depth_available
                and context_fresh
                and spread_pct <= max_spread_pct
                and direction_available
            )
            context_alignment_source: str | None = None
            if (
                ofi_directional
                and ofi_supports_side
                and not strong_depth_conflicts_side
            ):
                context_alignment_source = "temporal_ofi"
            elif depth_supports_side and flow_supports_side and not flow_conflicts_side:
                context_alignment_source = "depth_plus_flow"
            elif strong_depth_supports_side and not flow_conflicts_side:
                context_alignment_source = "strong_depth"
            alignment_candidate_source = context_alignment_source
            # A single order-book imbalance plus the last tick can flip on
            # one quote update. In LIVE mode require two distinct, fresh
            # quote versions separated in time unless upstream temporal OFI
            # provides its own multi-tick confirmation.
            if (
                is_live
                and context_alignment_source == "depth_plus_flow"
                and not ofi_directional
                and context_quality_eligible
                and side_aligns
            ):
                raw_version = quote_readiness.quote_update_version or indicators.get(
                    "quote_update_version"
                )
                try:
                    quote_version = (
                        int(float(raw_version)) if raw_version is not None else 0
                    )
                except (TypeError, ValueError, OverflowError):
                    quote_version = 0
                now = time.monotonic()
                previous = self._last_live_depth_support.get(symbol)
                persisted = bool(
                    quote_version > 0
                    and previous is not None
                    and previous[0] != quote_version
                    and 1.0 <= now - previous[1] <= 12.0
                )
                # Anchor the start of continuous support, not the newest
                # quote. Subsecond quote versions must not perpetually reset
                # the one-second persistence window.
                if quote_version > 0 and (
                    previous is None or now - previous[1] > 12.0
                ):
                    self._last_live_depth_support[symbol] = (
                        quote_version,
                        now,
                    )
                if not persisted:
                    context_alignment_source = None
            elif is_live:
                self._last_live_depth_support.pop(symbol, None)

            microstructure_supports_side = context_alignment_source is not None
            microstructure_conflicts_side = bool(
                (ofi_directional and ofi_conflicts_side) or strong_depth_conflicts_side
            )
            effective_context_alignment = bool(
                context_quality_eligible
                and side_aligns
                and microstructure_supports_side
                and not microstructure_conflicts_side
            )
            effective_context_conflict = bool(
                context_quality_eligible
                and (
                    (direction_available and direction != side)
                    or (side_aligns and microstructure_conflicts_side)
                )
            )

            reasons: list[str] = []
            if depth_supports_side:
                reasons.append("depth_imbalance_support")
            if strong_depth_supports_side:
                reasons.append("strong_depth_imbalance_support")
            if flow_supports_side:
                reasons.append(f"{flow_confirmation_source}_alignment")
            if side_aligns:
                reasons.append("underlying_direction_alignment")
            if context_fresh:
                reasons.append("fresh_context")

            metadata = {
                "strategy": "OrderFlow",
                "strategy_name": "OrderFlow",
                "role": "context",
                "source_domain": "market_microstructure",
                "side": side,
                "trade_side": side,
                "contract_side": side,
                "direction_bias": direction if direction_available else None,
                "underlying_direction_bias": direction if direction_available else None,
                "setup_pass": context_quality_eligible,
                "setup_type": "microstructure_confirmation",
                "setup_reasons": reasons,
                "required_data_present": depth_available,
                "stale_data_used": bool(indicators.get("stale_data_used")),
                "candidate_symbol": symbol,
                "bid": bid,
                "ask": ask,
                "spread_pct": round(spread_pct, 4),
                "depth_available": depth_available,
                "quote_depth_valid": bool(quote_readiness.depth_available),
                "tradable_quote": bool(quote_readiness.tradable_quote),
                "tick_age_ms": quote_readiness.tick_age_ms,
                "quote_update_version": quote_readiness.quote_update_version,
                "quote_readiness_allowed": quote_readiness.allowed,
                "quote_readiness_reason": quote_readiness.reason,
                "depth_imbalance": round(depth_imbalance, 4),
                "depth_supports_side": depth_supports_side,
                "strong_depth_supports_side": strong_depth_supports_side,
                "depth_conflicts_side": depth_conflicts_side,
                "strong_depth_conflicts_side": strong_depth_conflicts_side,
                "depth_support_threshold": round(support_threshold, 4),
                "strong_depth_support_threshold": round(strong_support_threshold, 4),
                "tick_direction": tick_direction,
                "tick_supports_side": tick_supports_side,
                "ofi_ready": ofi_ready,
                "ofi_1s_normalized": ofi_value,
                "ofi_threshold": ofi_threshold,
                "ofi_directional": ofi_directional,
                "ofi_supports_side": ofi_supports_side,
                "ofi_conflicts_side": ofi_conflicts_side,
                "tick_conflicts_side": tick_conflicts_side,
                "flow_confirmation_source": flow_confirmation_source,
                "flow_supports_side": flow_supports_side,
                "flow_conflicts_side": flow_conflicts_side,
                "microstructure_supports_side": microstructure_supports_side,
                "microstructure_conflicts_side": microstructure_conflicts_side,
                "context_alignment_source": context_alignment_source,
                "context_alignment_candidate_source": alignment_candidate_source,
                "context_age_seconds": context_age_seconds,
                "context_fresh": context_fresh,
                "context_quality_eligible": context_quality_eligible,
                "effective_context_alignment": effective_context_alignment,
                "effective_context_conflict": effective_context_conflict,
                "context_role": "confirmation",
                "vote_timestamp": time.time(),
                "vote_monotonic": time.monotonic(),
                "trigger_conditions_met": False,
                "trigger_block_reason": "context_only_role",
                "can_trigger": False,
                "premium_stop_distance": max(
                    0.8
                    * max(
                        float(indicators.get("atr") or 0.0),
                        current_price * 0.01,
                        1.0,
                    ),
                    current_price * 0.02,
                    1.0,
                ),
                "premium_target_rr": 1.8,
            }
            signal = EliteSignal(
                symbol=symbol,
                signal="BUY",
                confidence=1.0,
                entry_price=current_price,
                stop_loss=None,
                target=None,
                quantity=self._cfg.quantity or 1,
                strategy_name="OrderFlow",
                metadata=metadata,
            )
            _stamp_quote_update_identity(signal.metadata, indicators)
            LOGGER.info(
                "ORDERFLOW_CONTEXT_EVIDENCE symbol=%s side=%s eligible=%s "
                "aligned=%s conflict=%s source=%s ofi_ready=%s ofi_norm=%s "
                "depth=%.3f strong_depth_conflict=%s spread_pct=%.3f age_s=%.3f",
                symbol,
                side,
                context_quality_eligible,
                effective_context_alignment,
                effective_context_conflict,
                context_alignment_source,
                ofi_ready,
                ofi_value,
                depth_imbalance,
                strong_depth_conflicts_side,
                spread_pct,
                context_age_seconds,
                extra={
                    "event": "ORDERFLOW_CONTEXT_EVIDENCE",
                    "symbol": symbol,
                    "side": side,
                    "context_quality_eligible": context_quality_eligible,
                    "effective_context_alignment": effective_context_alignment,
                    "effective_context_conflict": effective_context_conflict,
                    "context_alignment_source": context_alignment_source,
                    "ofi_ready": ofi_ready,
                    "ofi_1s_normalized": ofi_value,
                    "depth_imbalance": round(depth_imbalance, 4),
                    "strong_depth_conflicts_side": strong_depth_conflicts_side,
                },
            )
            return signal
        except Exception as exc:
            LOGGER.error(
                "Failure in OrderFlowStrategy._evaluate_signal: %s",
                exc,
                exc_info=exc,
            )
            return None


__all__ = ["OrderFlowStrategy"]
