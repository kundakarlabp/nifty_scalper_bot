from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from typing import Any, Mapping

from nifty_scalper_bot.execution.readiness import HistoryReadinessPolicy
from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    SMCStrategyConfig,
)
from nifty_scalper_bot.strategies.runtime_context_contract import (
    resolve_context_age_seconds,
)
from nifty_scalper_bot.utils.logging import get_logger, log_throttled

LOGGER = get_logger(__name__)


def _safe_history_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_not_none(*values: int | None) -> int | None:
    return next((value for value in values if value is not None), None)


def safe_float_env(name: str, default: float) -> float:
    from nifty_scalper_bot.config.env_utils import parse_float_env

    return parse_float_env(os.getenv(name), default)


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(float(os.getenv(name, str(default)) or default)))
    except (TypeError, ValueError):
        return max(minimum, int(default))


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, (int, float)):
        number = float(value)
        if number > 10_000_000_000:
            number /= 1000.0
        result = datetime.fromtimestamp(number, tz=timezone.utc)
    elif isinstance(value, str) and value.strip():
        try:
            result = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc).replace(microsecond=0)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class SMCStrategy(EliteStrategy):
    """Underlying-led liquidity sweep/reclaim trigger for selected NIFTY options.

    The executable symbol remains the selected option. NIFTY futures are the
    primary market-structure source, with spot as a context-only fallback.
    Option-premium structure may add confirmation but cannot manufacture a
    liquidity sweep in LIVE trading.
    """

    # Only a small option history is needed for premium/quality context. The
    # SMC-specific 30-bar requirement is enforced on the underlying structure
    # source inside _underlying_snapshot.
    MIN_BARS_REQUIRED = 5
    ROLE = "trigger"
    TRIGGER_KEY = "smc_liquidity_v2"

    def __init__(self, config: SMCStrategyConfig, indicator_engine: Any) -> None:
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config
        self._events: dict[tuple[str, str], dict[str, Any]] = {}
        self._last_emitted_bar: dict[tuple[str, str], datetime] = {}
        self.last_sweep_diagnostics: dict[str, Any] = {}

    def get_required_indicators(self) -> set[str]:
        return {
            "high",
            "low",
            "close",
            "open",
            "atr",
            "direction_bias",
            "underlying_direction_bias",
            "bos_confirmed",
            "choch_confirmed",
            "retest_confirmed",
            "premium_reclaim",
            "spread_pct",
            "tradable_quote",
            "quote_depth_valid",
            "stale_data_used",
            "futures_symbol",
            "spot_symbol",
        }

    def _read_completed_bars(self, symbol: str) -> list[dict[str, Any]]:
        engine = self._indicator_engine
        if engine is None or not symbol or not hasattr(engine, "get_history"):
            return []
        try:
            rows = engine.get_history(symbol, field="bars")
        except TypeError:
            rows = engine.get_history(symbol)
        except Exception:
            return []

        completed: list[dict[str, Any]] = []
        for raw in rows or ():
            if not isinstance(raw, Mapping):
                continue
            if raw.get("is_provisional") is True or raw.get("is_complete") is False:
                continue
            ts = _coerce_datetime(raw.get("timestamp"))
            open_price = _safe_float(raw.get("open"))
            high = _safe_float(raw.get("high"))
            low = _safe_float(raw.get("low"))
            close = _safe_float(raw.get("close"))
            volume = _safe_float(raw.get("volume")) or 0.0
            if ts is None or None in {open_price, high, low, close}:
                continue
            assert (
                open_price is not None
                and high is not None
                and low is not None
                and close is not None
            )
            if min(open_price, high, low, close) <= 0:
                continue
            completed.append(
                {
                    "timestamp": ts,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": max(0.0, volume),
                }
            )
        completed.sort(key=lambda row: row["timestamp"])
        return completed

    @staticmethod
    def _atr(rows: list[dict[str, Any]], period: int = 14) -> float:
        if not rows:
            return 1.0
        sample = rows[-max(2, period + 1) :]
        true_ranges: list[float] = []
        previous_close: float | None = None
        for row in sample:
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            true_range = high - low
            if previous_close is not None:
                true_range = max(
                    true_range,
                    abs(high - previous_close),
                    abs(low - previous_close),
                )
            true_ranges.append(max(0.0, true_range))
            previous_close = close
        selected = true_ranges[-period:]
        average = sum(selected) / len(selected) if selected else 0.0
        return max(1.0, average)

    @staticmethod
    def _volume_ratio(rows: list[dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        current = float(rows[-1].get("volume") or 0.0)
        prior = [
            float(row.get("volume") or 0.0)
            for row in rows[-21:-1]
            if float(row.get("volume") or 0.0) > 0.0
        ]
        if current <= 0 or not prior:
            return 0.0
        average = sum(prior) / len(prior)
        return current / average if average > 0 else 0.0

    @staticmethod
    def _latest_confirmed_pivots(
        rows: list[dict[str, Any]], *, strength: int, lookback: int
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Return latest confirmed pivot low/high before the evaluation bar."""
        if len(rows) < (strength * 2) + 2:
            return None, None
        history = rows[:-1]
        start = max(strength, len(history) - max(lookback, strength * 2 + 1))
        stop = len(history) - strength
        pivot_low: dict[str, Any] | None = None
        pivot_high: dict[str, Any] | None = None

        for index in range(start, stop):
            left = history[index - strength : index]
            right = history[index + 1 : index + 1 + strength]
            if len(left) < strength or len(right) < strength:
                continue
            row = history[index]
            low = float(row["low"])
            high = float(row["high"])
            neighbor_lows = [float(item["low"]) for item in (*left, *right)]
            neighbor_highs = [float(item["high"]) for item in (*left, *right)]
            if low <= min(neighbor_lows) and low < max(neighbor_lows):
                pivot_low = row
            if high >= max(neighbor_highs) and high > min(neighbor_highs):
                pivot_high = row
        return pivot_low, pivot_high

    def _underlying_snapshot(
        self, indicators: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        required = max(10, int(HistoryReadinessPolicy.from_env().smc_min_bars))
        strength = _env_int("SMC_PIVOT_STRENGTH", 2)
        lookback = _env_int(
            "SMC_PIVOT_LOOKBACK", 30, minimum=(strength * 2) + 1
        )
        candidates = (
            (str(indicators.get("futures_symbol") or "").strip(), "futures"),
            (str(indicators.get("spot_symbol") or "").strip(), "spot_fallback"),
        )
        option_anchor = _coerce_datetime(
            indicators.get("latest_bar_ts")
            or indicators.get("bar_timestamp")
            or indicators.get("setup_candle_timestamp")
        )
        max_lag_seconds = max(
            60.0,
            safe_float_env("SMC_UNDERLYING_MAX_LAG_SECONDS", 120.0),
        )

        for underlying_symbol, source in candidates:
            if not underlying_symbol:
                continue
            rows = self._read_completed_bars(underlying_symbol)
            if len(rows) < required:
                continue
            current = rows[-1]
            current_ts = current["timestamp"]
            if (
                option_anchor is not None
                and abs((option_anchor - current_ts).total_seconds())
                > max_lag_seconds
            ):
                continue
            pivot_low, pivot_high = self._latest_confirmed_pivots(
                rows, strength=strength, lookback=lookback
            )
            return {
                "source": source,
                "symbol": underlying_symbol,
                "rows": rows,
                "current": current,
                "current_ts": current_ts,
                "atr": self._atr(rows),
                "volume_ratio": self._volume_ratio(rows),
                "pivot_low": pivot_low,
                "pivot_high": pivot_high,
                "history_count": len(rows),
            }

        # Backward-compatible diagnostic/replay path only. It is deliberately
        # unavailable in LIVE, so option-premium OHLC can never become the
        # authoritative SMC structure source.
        execution_mode = str(
            os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW"
        ).upper()
        source_symbol = str(indicators.get("source_symbol") or "").strip()
        source_upper = source_symbol.upper()
        explicit_underlying = bool(
            source_symbol
            and not source_upper.endswith(("CE", "PE"))
            and (
                source_upper.startswith("NSE:")
                or "FUT" in source_upper
                or str(indicators.get("history_domain_used") or "").lower()
                in {"spot", "underlying"}
            )
        )
        if execution_mode != "LIVE" and explicit_underlying:
            current_ts = option_anchor or datetime.now(timezone.utc).replace(
                microsecond=0
            )
            current = {
                "timestamp": current_ts,
                "open": float(indicators.get("open") or 0.0),
                "high": float(indicators.get("high") or 0.0),
                "low": float(indicators.get("low") or 0.0),
                "close": float(indicators.get("close") or 0.0),
                "volume": float(indicators.get("volume") or 0.0),
            }
            pivot_low_value = _safe_float(
                indicators.get("prior_swing_low")
                if indicators.get("prior_swing_low") is not None
                else indicators.get("swing_low")
            )
            pivot_high_value = _safe_float(
                indicators.get("prior_swing_high")
                if indicators.get("prior_swing_high") is not None
                else indicators.get("swing_high")
            )
            pivot_low = (
                {"timestamp": current_ts, "low": pivot_low_value}
                if pivot_low_value is not None
                else None
            )
            pivot_high = (
                {"timestamp": current_ts, "high": pivot_high_value}
                if pivot_high_value is not None
                else None
            )
            if min(
                float(current["open"]),
                float(current["high"]),
                float(current["low"]),
                float(current["close"]),
            ) > 0:
                return {
                    "source": "legacy_shadow_underlying_payload",
                    "symbol": source_symbol,
                    "rows": [current],
                    "current": current,
                    "current_ts": current_ts,
                    "atr": max(
                        1.0,
                        float(
                            indicators.get("underlying_atr")
                            or indicators.get("atr")
                            or 1.0
                        ),
                    ),
                    "volume_ratio": float(
                        indicators.get("volume_spike_ratio") or 0.0
                    ),
                    "pivot_low": pivot_low,
                    "pivot_high": pivot_high,
                    "history_count": int(
                        indicators.get("underlying_history_count") or 0
                    ),
                }
        return None

    def _sweep_diagnostics(
        self, snapshot: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        current = snapshot["current"]
        atr = max(1.0, float(snapshot["atr"]))
        min_sweep_atr = max(
            0.0, safe_float_env("SMC_MIN_SWEEP_ATR", 0.08)
        )
        max_sweep_atr = max(
            min_sweep_atr,
            safe_float_env("SMC_MAX_SWEEP_ATR", 0.75),
        )
        reclaim_buffer_atr = max(
            0.0, safe_float_env("SMC_RECLAIM_BUFFER_ATR", 0.03)
        )
        # sweep_distance_points was previously dead configuration. In v2 it is
        # an absolute cap on the ATR-normalised minimum, preventing shock-vol
        # regimes from demanding an unrealistically large minimum penetration.
        configured_cap = max(
            0.05, float(self._cfg.sweep_distance_points or 0.05)
        )
        effective_min = max(
            0.05, min(configured_cap, atr * min_sweep_atr)
        )
        reclaim_buffer = atr * reclaim_buffer_atr

        bullish: dict[str, Any] = {
            "exists": False,
            "valid": False,
            "reason": "missing_pivot",
        }
        pivot_low = snapshot.get("pivot_low")
        if isinstance(pivot_low, Mapping) and pivot_low.get("low") is not None:
            level = float(pivot_low["low"])
            depth = level - float(current["low"])
            reclaim = float(current["close"]) - level
            bullish = {
                "exists": depth > 0,
                "valid": bool(
                    depth >= effective_min
                    and depth <= atr * max_sweep_atr
                    and reclaim >= reclaim_buffer
                ),
                "level": level,
                "depth_points": max(0.0, depth),
                "depth_atr": max(0.0, depth) / atr,
                "reclaim_points": reclaim,
                "reclaim_atr": reclaim / atr,
                "too_shallow": bool(0 < depth < effective_min),
                "too_deep": bool(depth > atr * max_sweep_atr),
                "reclaim_failed": bool(
                    depth > 0 and reclaim < reclaim_buffer
                ),
            }

        bearish: dict[str, Any] = {
            "exists": False,
            "valid": False,
            "reason": "missing_pivot",
        }
        pivot_high = snapshot.get("pivot_high")
        if isinstance(pivot_high, Mapping) and pivot_high.get("high") is not None:
            level = float(pivot_high["high"])
            depth = float(current["high"]) - level
            reclaim = level - float(current["close"])
            bearish = {
                "exists": depth > 0,
                "valid": bool(
                    depth >= effective_min
                    and depth <= atr * max_sweep_atr
                    and reclaim >= reclaim_buffer
                ),
                "level": level,
                "depth_points": max(0.0, depth),
                "depth_atr": max(0.0, depth) / atr,
                "reclaim_points": reclaim,
                "reclaim_atr": reclaim / atr,
                "too_shallow": bool(0 < depth < effective_min),
                "too_deep": bool(depth > atr * max_sweep_atr),
                "reclaim_failed": bool(
                    depth > 0 and reclaim < reclaim_buffer
                ),
            }

        self.last_sweep_diagnostics = {
            "effective_min_sweep_points": effective_min,
            "min_sweep_atr": min_sweep_atr,
            "max_sweep_atr": max_sweep_atr,
            "reclaim_buffer_atr": reclaim_buffer_atr,
            "configured_sweep_distance_points": configured_cap,
            "underlying_atr": atr,
            "bullish": dict(bullish),
            "bearish": dict(bearish),
        }
        return bullish, bearish

    @staticmethod
    def _side_from_symbol(symbol: str) -> str:
        upper = str(symbol or "").strip().upper()
        if upper.endswith("CE"):
            return "CE"
        if upper.endswith("PE"):
            return "PE"
        return ""

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: dict[str, Any],
        current_price: float,
        position: Any | None = None,
    ) -> EliteSignal | None:
        del position
        try:
            self._no_vote("stale_or_invalid_data")
            direction = str(indicators.get("direction_bias") or "").upper()
            underlying_direction = str(
                indicators.get("underlying_direction_bias") or ""
            ).upper()
            effective_direction = underlying_direction or direction
            execution_mode = str(
                os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW"
            ).strip().upper()
            is_live = execution_mode == "LIVE"

            if is_live and not effective_direction:
                self._no_vote("direction_context_not_ready")
                log_throttled(
                    LOGGER,
                    f"smc_direction_context_not_ready:{symbol}",
                    "STRATEGY_NO_VOTE strategy=SMC symbol=%s reason=direction_context_not_ready direction_bias=%s underlying_direction_bias=%s",
                    symbol,
                    direction,
                    underlying_direction,
                    interval_sec=float(
                        os.getenv(
                            "SMC_DIRECTION_CONTEXT_NO_VOTE_LOG_THROTTLE_SECONDS",
                            "45",
                        )
                        or "45"
                    ),
                    level=logging.WARNING,
                    extra={
                        "event": "STRATEGY_NO_VOTE",
                        "strategy": "SMC",
                        "symbol": symbol,
                        "reason": "direction_context_not_ready",
                    },
                )
                return None

            stale_data = bool(indicators.get("stale_data_used")) or float(
                indicators.get("data_age_seconds") or 0.0
            ) > 120.0
            if stale_data or current_price <= 0:
                self._no_vote("stale_or_invalid_data")
                return None

            # Execution identity is owned by the evaluated candidate symbol.
            # Underlying/futures context may describe structure, but it cannot
            # turn a validated CE/PE candidate into a non-option instrument.
            contract_side = self._side_from_symbol(symbol)
            if contract_side not in {"CE", "PE"}:
                self._no_vote("smc_executable_symbol_not_option")
                return None

            snapshot = self._underlying_snapshot(indicators)
            if snapshot is None:
                self._no_vote("underlying_context_not_ready")
                return None

            underlying_symbol = str(snapshot["symbol"])
            current = snapshot["current"]
            current_ts = snapshot["current_ts"]
            event_key = (underlying_symbol, contract_side)
            if self._last_emitted_bar.get(event_key) == current_ts:
                self._no_vote("smc_duplicate_confirmation_bar")
                return None

            event = self._events.get(event_key)
            if event is not None:
                max_age_minutes = max(
                    1.0,
                    safe_float_env("SMC_CONFIRMATION_MAX_MINUTES", 5.0),
                )
                age_minutes = max(
                    0.0,
                    (current_ts - event["sweep_ts"]).total_seconds() / 60.0,
                )
                if age_minutes > max_age_minutes:
                    self._events.pop(event_key, None)
                    self._no_vote("smc_sweep_expired")
                    return None

                side = str(event["side"])
                if side == "CE" and float(current["close"]) <= float(
                    event["sweep_extreme"]
                ):
                    self._events.pop(event_key, None)
                    self._no_vote("smc_sweep_invalidated")
                    return None
                if side == "PE" and float(current["close"]) >= float(
                    event["sweep_extreme"]
                ):
                    self._events.pop(event_key, None)
                    self._no_vote("smc_sweep_invalidated")
                    return None

                if current_ts <= event["sweep_ts"]:
                    self._no_vote("smc_awaiting_confirmation")
                    return None

                atr = max(1.0, float(snapshot["atr"]))
                body = abs(float(current["close"]) - float(current["open"]))
                displacement_score = body / atr
                displacement_min = max(
                    0.05,
                    safe_float_env(
                        "SMC_CONFIRMATION_DISPLACEMENT_ATR", 0.25
                    ),
                )
                if side == "CE":
                    price_confirmation = float(current["close"]) > float(
                        event["sweep_bar_high"]
                    )
                else:
                    price_confirmation = float(current["close"]) < float(
                        event["sweep_bar_low"]
                    )
                displacement_confirmed = bool(
                    price_confirmation
                    and displacement_score >= displacement_min
                )
                if not displacement_confirmed:
                    self._no_vote("smc_awaiting_confirmation")
                    return None

                if (
                    effective_direction in {"CE", "PE"}
                    and effective_direction != contract_side
                ):
                    self._events.pop(event_key, None)
                    self._no_vote("smc_direction_conflict")
                    return None

                bos_confirmed = bool(indicators.get("bos_confirmed"))
                choch_confirmed = bool(indicators.get("choch_confirmed"))
                structure_confirmed = bool(bos_confirmed or choch_confirmed)
                retest_confirmed = bool(
                    indicators.get("retest_confirmed")
                    or indicators.get("mitigation_confirmed")
                )
                premium_reclaim = bool(indicators.get("premium_reclaim"))
                direction_aligned = (
                    effective_direction in {"CE", "PE"}
                    and effective_direction == contract_side
                )

                score = 5.0
                reasons = [
                    "underlying_liquidity_sweep",
                    "reclaim",
                    "displacement_confirmation",
                ]
                if direction_aligned:
                    score += 1.5
                    reasons.append("direction_alignment")
                if bool(event["volume_confirmation"]):
                    score += 1.0
                    reasons.append("volume_confirmation")
                if structure_confirmed:
                    score += 1.0
                    reasons.append("structure_confirmation")
                if retest_confirmed:
                    score += 0.5
                    reasons.append("retest_mitigation")
                if premium_reclaim:
                    score += 0.5
                    reasons.append("premium_reclaim_support")
                depth_atr = float(event["depth_atr"])
                if 0.12 <= depth_atr <= 0.50:
                    score += 0.5
                    reasons.append("balanced_sweep_depth")

                strategy_score = max(0.0, min(10.0, score))
                min_score = float(
                    os.getenv("SMC_MIN_SCORE_LIVE", "6.5")
                    if is_live
                    else os.getenv("SMC_MIN_SCORE_SHADOW", "4.5")
                )
                if strategy_score < min_score:
                    self._no_vote("smc_low_score")
                    return None

                option_atr = max(
                    float(indicators.get("atr") or 0.0),
                    current_price * 0.01,
                    1.0,
                )
                invalidation_buffer = atr * max(
                    0.02,
                    safe_float_env("SMC_INVALIDATION_BUFFER_ATR", 0.05),
                )
                if contract_side == "CE":
                    underlying_invalidation = (
                        float(event["sweep_extreme"])
                        - invalidation_buffer
                    )
                else:
                    underlying_invalidation = (
                        float(event["sweep_extreme"])
                        + invalidation_buffer
                    )

                feature_names = (
                    "premium_reclaim",
                    "bullish_reversal",
                    "bearish_reversal",
                    "choch_confirmed",
                    "bos_confirmed",
                    "retest_confirmed",
                )
                feature_completeness = sum(
                    1
                    for name in feature_names
                    if indicators.get(name) is not None
                ) / float(len(feature_names))
                context_age_seconds = resolve_context_age_seconds(indicators)
                metadata = {
                    "strategy": "SMC",
                    "strategy_name": "SMC",
                    "role": "trigger",
                    "signal_family": "directional_trigger",
                    "trade_side": contract_side,
                    "side": contract_side,
                    "contract_side": contract_side,
                    "direction_bias": contract_side,
                    "underlying_direction_bias": (
                        underlying_direction
                        if underlying_direction in {"CE", "PE"}
                        else None
                    ),
                    "context_age_seconds": context_age_seconds,
                    "source_domain": "underlying_price",
                    "structure_source": snapshot["source"],
                    "structure_symbol": underlying_symbol,
                    "preliminary_only": True,
                    "requires_runner_final_score": True,
                    "requires_orderflow_confirmation": True,
                    "orderflow_confirmation_owner": "StrategyManager",
                    "direction_score": strategy_score,
                    "strategy_score": strategy_score,
                    "data_score": 8.0,
                    "score_reasons": reasons,
                    "setup_quality": strategy_score,
                    "setup_type": "liquidity_sweep_reclaim_confirmation",
                    "required_data_present": True,
                    "stale_data_used": stale_data,
                    "candidate_symbol": symbol,
                    "rejection_reasons": [],
                    "sweep_level": float(event["sweep_level"]),
                    "sweep_extreme": float(event["sweep_extreme"]),
                    "sweep_depth_points": float(event["depth_points"]),
                    "sweep_depth_atr": depth_atr,
                    "reclaim_distance_points": float(
                        event["reclaim_points"]
                    ),
                    "reclaim_distance_atr": float(event["reclaim_atr"]),
                    "displacement_score": round(displacement_score, 3),
                    "structure_confirmed": structure_confirmed,
                    "momentum_confirmed": True,
                    "structure_or_momentum_confirmed": True,
                    "smc_sweep_type": (
                        "bullish" if contract_side == "CE" else "bearish"
                    ),
                    "structure_confirmation_used": structure_confirmed,
                    "premium_reclaim_used": premium_reclaim,
                    "retest_confirmed": retest_confirmed,
                    "volume_ratio": float(event["volume_ratio"]),
                    "volume_confirmation": bool(
                        event["volume_confirmation"]
                    ),
                    "volume_spike_threshold": float(
                        self._cfg.volume_spike_mult
                    ),
                    "effective_min_sweep_points": float(
                        event["effective_min_sweep_points"]
                    ),
                    "underlying_atr": atr,
                    "underlying_invalidation_level": underlying_invalidation,
                    "premium_stop_distance": max(
                        option_atr,
                        current_price * 0.02,
                        1.0,
                    ),
                    "premium_target_rr": 2.0,
                    "partial_features_used": False,
                    "feature_completeness": feature_completeness,
                    "smc_quality_score": strategy_score,
                    "smc_block_reason": "",
                    "latest_bar_ts": current_ts,
                    "setup_candle_timestamp": current_ts,
                    "sweep_timestamp": event["sweep_ts"],
                }
                self._events.pop(event_key, None)
                self._last_emitted_bar[event_key] = current_ts
                LOGGER.info(
                    "STRATEGY_VOTE strategy=SMC side=%s score=%.2f source=%s "
                    "sweep_depth_atr=%.3f displacement_atr=%.3f",
                    contract_side,
                    strategy_score,
                    snapshot["source"],
                    depth_atr,
                    displacement_score,
                )
                return EliteSignal(
                    symbol=symbol,
                    signal="BUY",
                    confidence=max(
                        0.1, min(0.88, strategy_score / 10.0)
                    ),
                    entry_price=current_price,
                    stop_loss=None,
                    target=None,
                    quantity=self._cfg.quantity or 1,
                    strategy_name="SMC",
                    metadata=metadata,
                )

            bullish, bearish = self._sweep_diagnostics(snapshot)
            desired = bullish if contract_side == "CE" else bearish
            opposite = bearish if contract_side == "CE" else bullish

            if bool(desired.get("too_shallow")):
                self._no_vote("smc_sweep_too_shallow")
                return None
            if bool(desired.get("too_deep")):
                self._no_vote("smc_sweep_too_deep")
                return None
            if bool(desired.get("reclaim_failed")):
                self._no_vote("smc_reclaim_failed")
                return None
            if not bool(desired.get("valid")):
                if bool(opposite.get("valid")):
                    self._no_vote("smc_underlying_side_mismatch")
                else:
                    self._no_vote("underlying_no_liquidity_sweep")
                return None

            side = contract_side
            if (
                effective_direction in {"CE", "PE"}
                and effective_direction != side
            ):
                self._no_vote("smc_direction_conflict")
                return None

            volume_ratio = float(snapshot["volume_ratio"])
            volume_threshold = max(
                0.0, float(self._cfg.volume_spike_mult or 0.0)
            )
            volume_confirmation = bool(
                volume_threshold > 0 and volume_ratio >= volume_threshold
            )
            sweep_extreme = (
                float(current["low"])
                if side == "CE"
                else float(current["high"])
            )
            self._events[event_key] = {
                "side": side,
                "sweep_ts": current_ts,
                "sweep_level": float(desired["level"]),
                "sweep_extreme": sweep_extreme,
                "sweep_bar_high": float(current["high"]),
                "sweep_bar_low": float(current["low"]),
                "depth_points": float(desired["depth_points"]),
                "depth_atr": float(desired["depth_atr"]),
                "reclaim_points": float(desired["reclaim_points"]),
                "reclaim_atr": float(desired["reclaim_atr"]),
                "volume_ratio": volume_ratio,
                "volume_confirmation": volume_confirmation,
                "effective_min_sweep_points": float(
                    self.last_sweep_diagnostics[
                        "effective_min_sweep_points"
                    ]
                ),
            }
            self._no_vote("smc_awaiting_confirmation")
            LOGGER.info(
                "SMC_SWEEP_ARMED symbol=%s option_side=%s "
                "structure_symbol=%s source=%s sweep_level=%.2f "
                "depth_points=%.2f depth_atr=%.3f reclaim_points=%.2f "
                "volume_ratio=%.2f",
                symbol,
                side,
                underlying_symbol,
                snapshot["source"],
                float(desired["level"]),
                float(desired["depth_points"]),
                float(desired["depth_atr"]),
                float(desired["reclaim_points"]),
                volume_ratio,
                extra={
                    "event": "SMC_SWEEP_ARMED",
                    "symbol": symbol,
                    "side": side,
                    "structure_symbol": underlying_symbol,
                    "structure_source": snapshot["source"],
                    "sweep_level": float(desired["level"]),
                    "sweep_depth_points": float(desired["depth_points"]),
                    "sweep_depth_atr": float(desired["depth_atr"]),
                    "reclaim_points": float(desired["reclaim_points"]),
                    "volume_ratio": volume_ratio,
                    "volume_confirmation": volume_confirmation,
                },
            )
            return None
        except Exception as exc:
            LOGGER.error(
                "Failure in SMCStrategy._evaluate_signal symbol=%s "
                "last_no_vote_reason=%s error=%s",
                symbol,
                getattr(self, "last_no_vote_reason", None),
                exc,
                exc_info=exc,
            )
            self._no_vote("evaluation_failed")
            return None


__all__ = ["SMCStrategy"]