from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
import os
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    ORBProStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)
_INDIA_TZ = ZoneInfo("Asia/Kolkata")
_MARKET_OPEN = time(hour=9, minute=15)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default)) or default))
    except (TypeError, ValueError):
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


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


class ORBProStrategy(EliteStrategy):
    """Underlying-led NIFTY opening-range breakout trigger.

    Spot/futures remain context-only. The strategy evaluates the selected option
    symbol but derives opening-range structure from already-hydrated NIFTY
    futures history, with spot as a fail-safe context fallback. It never selects
    or executes an underlying instrument.
    """

    MIN_BARS_REQUIRED = 5
    ROLE = "trigger"

    def __init__(self, config: ORBProStrategyConfig, indicator_engine: Any) -> None:
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config
        self._events: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._last_bar_by_key: dict[tuple[str, str, str], datetime] = {}
        self._event_count_by_key: dict[tuple[str, str, str], int] = {}

    def get_required_indicators(self) -> set[str]:
        """Return strategy-facing option and underlying context requirements."""
        return {
            "close",
            "open",
            "high",
            "low",
            "atr",
            "direction_bias",
            "underlying_direction_bias",
            "regime",
            "spread_pct",
            "quote_depth_valid",
            "tradable_quote",
            "stale_data_used",
            "futures_symbol",
            "futures_price",
            "spot_symbol",
            "spot_price",
            "futures_vwap_slope",
            # Legacy option ORB is retained only for diagnostics/backward telemetry.
            "orb_high",
            "orb_low",
            "orb_ready",
        }

    def _read_completed_bars(self, symbol: str) -> list[dict[str, Any]]:
        engine = self._indicator_engine
        if engine is None or not symbol or not hasattr(engine, "get_history"):
            return []
        rows = engine.get_history(symbol, field="bars")
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
            assert open_price is not None and high is not None and low is not None and close is not None
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
    def _underlying_atr(rows: list[dict[str, Any]], period: int = 14) -> float:
        if not rows:
            return 1.0
        sample = rows[-max(2, period + 1) :]
        true_ranges: list[float] = []
        previous_close: float | None = None
        for row in sample:
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            tr = high - low
            if previous_close is not None:
                tr = max(tr, abs(high - previous_close), abs(low - previous_close))
            true_ranges.append(max(0.0, tr))
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
        return current / (sum(prior) / len(prior))

    def _underlying_snapshot(
        self, indicators: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        orb_minutes = max(1, int(self._cfg.orb_minutes or 1))
        candidates = (
            (str(indicators.get("futures_symbol") or "").strip(), "futures"),
            (str(indicators.get("spot_symbol") or "").strip(), "spot_fallback"),
        )
        for underlying_symbol, source in candidates:
            if not underlying_symbol:
                continue
            rows = self._read_completed_bars(underlying_symbol)
            if len(rows) < 2:
                continue
            latest = rows[-1]
            latest_ts = latest["timestamp"]
            local_ts = latest_ts.astimezone(_INDIA_TZ)
            session_open_local = local_ts.replace(
                hour=_MARKET_OPEN.hour,
                minute=_MARKET_OPEN.minute,
                second=0,
                microsecond=0,
            )
            if local_ts.time() < _MARKET_OPEN:
                session_open_local -= timedelta(days=1)
            session_open = session_open_local.astimezone(timezone.utc)
            cutoff = session_open + timedelta(minutes=orb_minutes)

            # Minute bars are timestamped by bar start, therefore a 15-minute OR
            # is [09:15, 09:30), not [09:15, 09:30].
            range_rows = [
                row for row in rows if session_open <= row["timestamp"] < cutoff
            ]
            if not range_rows or latest_ts < cutoff:
                continue
            prior_rows = [row for row in rows if row["timestamp"] < latest_ts]
            if not prior_rows:
                continue
            previous = prior_rows[-1]
            orb_high = max(float(row["high"]) for row in range_rows)
            orb_low = min(float(row["low"]) for row in range_rows)
            if orb_high <= orb_low:
                continue
            atr = self._underlying_atr(rows)
            body_range = max(float(latest["high"]) - float(latest["low"]), 1e-9)
            body_ratio = abs(float(latest["close"]) - float(latest["open"])) / body_range
            return {
                "source": source,
                "symbol": underlying_symbol,
                "session_date": session_open_local.date().isoformat(),
                "session_open": session_open,
                "cutoff": cutoff,
                "orb_minutes": orb_minutes,
                "orb_high": orb_high,
                "orb_low": orb_low,
                "previous": previous,
                "current": latest,
                "current_ts": latest_ts,
                "atr": atr,
                "body_ratio": body_ratio,
                "volume_ratio": self._volume_ratio(rows),
                "minutes_after_range": max(
                    0.0, (latest_ts - cutoff).total_seconds() / 60.0
                ),
            }
        return None

    @staticmethod
    def _side_from_symbol(symbol: str) -> str:
        upper = str(symbol or "").strip().upper()
        if upper.endswith("CE"):
            return "CE"
        if upper.endswith("PE"):
            return "PE"
        return ""

    def _quality_score(
        self,
        *,
        side: str,
        branch: str,
        direction: str,
        penetration_atr: float,
        volume_ratio: float,
        indicators: Mapping[str, Any],
    ) -> tuple[float, list[str]]:
        score = 4.0
        reasons = ["underlying_opening_range_complete", "fresh_underlying_breakout"]
        score += 2.0
        reasons.append("retest_hold" if branch == "retest" else "momentum_acceptance")
        if direction == side:
            score += 1.0
            reasons.append("underlying_direction_alignment")
        if volume_ratio >= _env_float("ORB_VOLUME_CONFIRM_RATIO", 1.2):
            score += 1.0
            reasons.append("underlying_volume_confirmation")
        if penetration_atr >= _env_float("ORB_PENETRATION_CONFIRM_ATR", 0.2):
            score += 1.0
            reasons.append("normalized_breakout_penetration")
        slope = _safe_float(indicators.get("futures_vwap_slope"))
        if slope is not None and ((side == "CE" and slope > 0) or (side == "PE" and slope < 0)):
            score += 1.0
            reasons.append("futures_vwap_slope_alignment")
        return max(0.0, min(10.0, score)), reasons

    def _build_signal(
        self,
        *,
        symbol: str,
        side: str,
        current_price: float,
        indicators: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        event: Mapping[str, Any],
        branch: str,
        retest_timestamp: datetime | None,
    ) -> EliteSignal:
        option_atr = max(float(indicators.get("atr") or 0.0), current_price * 0.01, 1.0)
        max_stop_pct = max(0.5, _env_float("ORB_PREMIUM_STOP_MAX_PCT", 8.0)) / 100.0
        atr_stop = max(0.5, _env_float("ORB_PREMIUM_STOP_ATR_MULT", 0.75) * option_atr)
        premium_stop_distance = min(atr_stop, max(0.5, current_price * max_stop_pct))
        stop_loss = max(0.05, current_price - premium_stop_distance)
        target_rr = max(1.0, _env_float("ORB_TARGET_RR", 1.8))
        target = current_price + premium_stop_distance * target_rr

        boundary = float(event["boundary"])
        underlying_atr = float(snapshot["atr"])
        invalidation_buffer = max(
            0.05 * underlying_atr,
            _env_float("ORB_RETEST_TOLERANCE_ATR", 0.15) * underlying_atr,
        )
        underlying_invalidation = (
            boundary - invalidation_buffer if side == "CE" else boundary + invalidation_buffer
        )
        current_underlying = float(snapshot["current"]["close"])
        penetration_atr = abs(current_underlying - boundary) / max(underlying_atr, 1e-9)
        direction = str(
            indicators.get("underlying_direction_bias")
            or indicators.get("direction_bias")
            or ""
        ).upper()
        strategy_score, reasons = self._quality_score(
            side=side,
            branch=branch,
            direction=direction,
            penetration_atr=penetration_atr,
            volume_ratio=float(snapshot["volume_ratio"]),
            indicators=indicators,
        )
        breakout_ts = event["breakout_timestamp"]
        source = str(snapshot["source"])
        setup_id = (
            f"orbv2:{snapshot['session_date']}:{snapshot['symbol']}:{side}:"
            f"{breakout_ts.isoformat()}"
        )
        metadata = {
            "strategy": "ORBPro",
            "strategy_name": "ORBPro",
            "role": "trigger",
            "trade_side": side,
            "side": side,
            "contract_side": side,
            "candidate_symbol": symbol,
            "setup_id": setup_id,
            "setup_type": "underlying_opening_range_breakout",
            "signal_domain": "NIFTY_FUTURES" if source == "futures" else "NIFTY_SPOT_FALLBACK",
            "source_domain": "underlying_orb",
            "opening_range_source": source,
            "underlying_symbol": snapshot["symbol"],
            "orb_window_minutes": int(snapshot["orb_minutes"]),
            "opening_range_high": float(snapshot["orb_high"]),
            "opening_range_low": float(snapshot["orb_low"]),
            "opening_range_complete": True,
            "breakout_side": side,
            "breakout_timestamp": breakout_ts.timestamp(),
            "breakout_age_seconds": max(
                0.0,
                (snapshot["current_ts"] - breakout_ts).total_seconds(),
            ),
            "entry_branch": branch,
            "retest_confirmed": branch == "retest",
            "retest_timestamp": retest_timestamp.timestamp() if retest_timestamp else None,
            "underlying_atr": underlying_atr,
            "underlying_breakout_body_pct": round(float(event["body_ratio"]), 4),
            "underlying_volume_ratio": round(float(snapshot["volume_ratio"]), 4),
            "underlying_penetration_atr": round(penetration_atr, 4),
            "underlying_entry": current_underlying,
            "underlying_invalidation": underlying_invalidation,
            "premium_stop_distance": premium_stop_distance,
            "premium_target_rr": target_rr,
            "raw_setup_score": strategy_score,
            "strategy_score": strategy_score,
            "setup_quality": strategy_score,
            "confidence_semantics": "setup_quality_fraction_not_probability",
            "score_reasons": reasons,
            "required_data_present": True,
            "stale_data_used": bool(indicators.get("stale_data_used")),
            "direction_bias": side,
            "invalidation_level": stop_loss,
            "legacy_option_orb_high": indicators.get("orb_high"),
            "legacy_option_orb_low": indicators.get("orb_low"),
            "legacy_option_orb_ready": indicators.get("orb_ready"),
            "latest_bar_ts": snapshot["current_ts"].timestamp(),
            "setup_candle_timestamp": snapshot["current_ts"].timestamp(),
            "rejection_reasons": [],
        }
        LOGGER.info(
            "STRATEGY_VOTE strategy=ORBProV2 side=%s branch=%s score=%.2f source=%s underlying=%s",
            side,
            branch,
            strategy_score,
            source,
            snapshot["symbol"],
        )
        return EliteSignal(
            symbol=symbol,
            signal="BUY",
            confidence=max(0.1, min(0.9, strategy_score / 10.0)),
            entry_price=current_price,
            stop_loss=stop_loss,
            target=target,
            quantity=self._cfg.quantity or 1,
            strategy_name="ORBPro",
            metadata=metadata,
        )

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: dict[str, Any],
        current_price: float,
        position: Any | None = None,
    ) -> EliteSignal | None:
        del position
        self._no_vote("stale_or_invalid_data")
        if current_price <= 0 or bool(indicators.get("stale_data_used")):
            return None
        side = self._side_from_symbol(symbol)
        if not side:
            self._no_vote("invalid_option_contract_side")
            return None
        regime = str(indicators.get("regime") or "").upper()
        if regime == "CHOPPY":
            self._no_vote("choppy_regime")
            return None

        snapshot = self._underlying_snapshot(indicators)
        if snapshot is None:
            self._no_vote("underlying_orb_not_ready")
            return None
        if float(snapshot["minutes_after_range"]) > max(
            1.0, _env_float("ORB_MAX_ENTRY_MINUTES_AFTER_RANGE", 120.0)
        ):
            self._no_vote("orb_entry_window_expired")
            return None

        key = (str(snapshot["session_date"]), str(snapshot["symbol"]), side)
        current_ts = snapshot["current_ts"]
        if self._last_bar_by_key.get(key) == current_ts:
            self._no_vote("orb_bar_already_evaluated")
            return None
        self._last_bar_by_key[key] = current_ts

        direction = str(
            indicators.get("underlying_direction_bias")
            or indicators.get("direction_bias")
            or ""
        ).upper()
        if direction in {"CE", "PE"} and direction != side:
            self._no_vote("underlying_direction_conflict_fail_closed")
            return None

        orb_high = float(snapshot["orb_high"])
        orb_low = float(snapshot["orb_low"])
        underlying_atr = float(snapshot["atr"])
        buffer = max(0.0, _env_float("ORB_BREAKOUT_BUFFER_ATR", 0.05)) * underlying_atr
        tolerance = max(0.0, _env_float("ORB_RETEST_TOLERANCE_ATR", 0.15)) * underlying_atr
        previous_close = float(snapshot["previous"]["close"])
        current_bar = snapshot["current"]
        current_close = float(current_bar["close"])

        event = self._events.get(key)
        if event is not None and event.get("status") == "EMITTED":
            back_inside = current_close <= orb_high if side == "CE" else current_close >= orb_low
            if back_inside:
                self._events.pop(key, None)
                event = None
            else:
                self._no_vote("orb_event_already_emitted")
                return None
        if event is not None and event.get("status") in {"INVALIDATED", "EXPIRED"}:
            back_inside = current_close <= orb_high if side == "CE" else current_close >= orb_low
            if back_inside:
                self._events.pop(key, None)
                event = None
            else:
                self._no_vote("orb_event_inactive")
                return None

        if event is None:
            max_events = max(1, _env_int("ORB_MAX_EVENTS_PER_SIDE", 2))
            if self._event_count_by_key.get(key, 0) >= max_events:
                self._no_vote("orb_session_event_limit")
                return None
            if side == "CE":
                boundary = orb_high
                fresh_breakout = previous_close <= boundary + buffer and current_close > boundary + buffer
            else:
                boundary = orb_low
                fresh_breakout = previous_close >= boundary - buffer and current_close < boundary - buffer
            if not fresh_breakout:
                self._no_vote("no_fresh_underlying_breakout")
                return None

            body_ratio = float(snapshot["body_ratio"])
            if body_ratio < max(0.0, _env_float("ORB_MIN_BREAKOUT_BODY_PCT", 0.35)):
                self._no_vote("wick_only_underlying_breakout")
                return None
            penetration_atr = abs(current_close - boundary) / max(underlying_atr, 1e-9)
            event = {
                "status": "AWAITING_RETEST",
                "boundary": boundary,
                "breakout_timestamp": current_ts,
                "body_ratio": body_ratio,
                "penetration_atr": penetration_atr,
            }
            self._events[key] = event

            momentum_enabled = _env_bool("ORB_MOMENTUM_BRANCH_ENABLED", True)
            momentum_confirmed = bool(
                momentum_enabled
                and body_ratio >= _env_float("ORB_MOMENTUM_MIN_BODY_PCT", 0.60)
                and penetration_atr >= _env_float("ORB_MOMENTUM_MIN_PENETRATION_ATR", 0.20)
                and float(snapshot["volume_ratio"])
                >= _env_float("ORB_MOMENTUM_MIN_VOLUME_RATIO", 1.20)
            )
            if momentum_confirmed:
                event["status"] = "EMITTED"
                self._event_count_by_key[key] = self._event_count_by_key.get(key, 0) + 1
                return self._build_signal(
                    symbol=symbol,
                    side=side,
                    current_price=current_price,
                    indicators=indicators,
                    snapshot=snapshot,
                    event=event,
                    branch="momentum",
                    retest_timestamp=None,
                )
            self._no_vote("awaiting_orb_retest")
            return None

        breakout_ts = event["breakout_timestamp"]
        breakout_age = max(0.0, (current_ts - breakout_ts).total_seconds())
        if breakout_age > max(60.0, _env_float("ORB_BREAKOUT_MAX_AGE_SECONDS", 600.0)):
            event["status"] = "EXPIRED"
            self._no_vote("orb_breakout_expired")
            return None

        boundary = float(event["boundary"])
        if side == "CE":
            invalidated = current_close < boundary - tolerance
            retest = bool(
                current_ts > breakout_ts
                and float(current_bar["low"]) <= boundary + tolerance
                and current_close >= boundary
            )
        else:
            invalidated = current_close > boundary + tolerance
            retest = bool(
                current_ts > breakout_ts
                and float(current_bar["high"]) >= boundary - tolerance
                and current_close <= boundary
            )
        if invalidated:
            event["status"] = "INVALIDATED"
            self._no_vote("orb_breakout_invalidated")
            return None
        if not retest:
            self._no_vote("awaiting_orb_retest")
            return None

        event["status"] = "EMITTED"
        self._event_count_by_key[key] = self._event_count_by_key.get(key, 0) + 1
        return self._build_signal(
            symbol=symbol,
            side=side,
            current_price=current_price,
            indicators=indicators,
            snapshot=snapshot,
            event=event,
            branch="retest",
            retest_timestamp=current_ts,
        )


__all__ = ["ORBProStrategy"]
