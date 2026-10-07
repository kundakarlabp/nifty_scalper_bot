from __future__ import annotations

import os
from datetime import datetime, time, timedelta, timezone
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from nifty_scalper_bot.config.regime_ontology import MarketRegime, normalize_regime
from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    ORBProStrategyConfig,
)
from nifty_scalper_bot.strategies.setup_lifecycle import SetupStage, transition_setup
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
    symbol but derives opening-range structure and participation only from
    already-hydrated NIFTY futures history. Spot may still contribute to the
    external direction context, but it cannot authorize ORB structure because
    the NIFTY index itself has no traded volume. The strategy never selects or
    executes an underlying instrument.
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
            "futures_vwap_slope",
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
    def _volume_ratio_at(rows: list[dict[str, Any]], index: int) -> float:
        if not rows or index < 0 or index >= len(rows):
            return 0.0
        current = float(rows[index].get("volume") or 0.0)
        prior = [
            float(row.get("volume") or 0.0)
            for row in rows[max(0, index - 20) : index]
            if float(row.get("volume") or 0.0) > 0.0
        ]
        if current <= 0 or not prior:
            return 0.0
        return current / (sum(prior) / len(prior))

    @classmethod
    def _volume_ratio(cls, rows: list[dict[str, Any]]) -> float:
        return cls._volume_ratio_at(rows, len(rows) - 1)

    def _underlying_snapshot(
        self, indicators: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        orb_minutes = max(1, int(self._cfg.orb_minutes or 1))
        underlying_symbol = str(indicators.get("futures_symbol") or "").strip()
        if not underlying_symbol:
            return None
        rows = self._read_completed_bars(underlying_symbol)
        if len(rows) < 2:
            return None

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

        # Normalize ORB features strictly within the active session. Carrying
        # prior-session closing bars into opening participation or ATR makes
        # today's breakout depend on yesterday's close/volume and creates live
        # versus daily-reset research drift.
        session_rows = [
            row for row in rows if session_open <= row["timestamp"] <= latest_ts
        ]
        if len(session_rows) < 2:
            return None

        # Minute bars are timestamped by bar start, therefore a 15-minute OR
        # is [09:15, 09:30), not [09:15, 09:30].
        range_rows = [row for row in session_rows if row["timestamp"] < cutoff]
        # Elapsed time or row count cannot prove coverage: a missing minute
        # can hide the true range extreme, even when duplicates fill the count.
        expected_starts = {
            session_open + timedelta(minutes=minute) for minute in range(orb_minutes)
        }
        if {
            row["timestamp"] for row in range_rows
        } != expected_starts or latest_ts < cutoff:
            return None

        prior_rows = [row for row in session_rows if row["timestamp"] < latest_ts]
        if not prior_rows:
            return None
        previous = prior_rows[-1]
        orb_high = max(float(row["high"]) for row in range_rows)
        orb_low = min(float(row["low"]) for row in range_rows)
        if orb_high <= orb_low:
            return None

        atr = self._underlying_atr(session_rows)
        body_range = max(float(latest["high"]) - float(latest["low"]), 1e-9)
        body_ratio = abs(float(latest["close"]) - float(latest["open"])) / body_range
        return {
            "source": "futures",
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
            "volume_ratio": self._volume_ratio(session_rows),
            "rows": session_rows,
            "underlying_atr_basis": "same_session_completed_bars",
            "underlying_volume_ratio_basis": "same_session_prior_completed_bars",
            "minutes_after_range": max(
                0.0, (latest_ts - cutoff).total_seconds() / 60.0
            ),
        }

    def _recover_unconfirmed_breakout(
        self,
        snapshot: Mapping[str, Any],
        *,
        side: str,
    ) -> dict[str, Any] | None:
        """Recover only a breakout that could still be awaiting its first retest.

        The scan refuses momentum breakouts and any event already retested or
        invalidated before the current bar. This restores state lost between
        bars without replaying a signal that the previous process could have
        emitted.
        """
        rows = list(snapshot.get("rows") or ())
        if len(rows) < 3:
            return None
        current_ts = snapshot["current_ts"]
        cutoff = snapshot["cutoff"]
        boundary = float(snapshot["orb_high"] if side == "CE" else snapshot["orb_low"])
        max_age = max(60.0, _env_float("ORB_BREAKOUT_MAX_AGE_SECONDS", 600.0))

        for index in range(len(rows) - 2, 0, -1):
            breakout = rows[index]
            breakout_ts = breakout["timestamp"]
            if (
                breakout_ts < cutoff
                or (current_ts - breakout_ts).total_seconds() > max_age
            ):
                break
            previous_close = float(rows[index - 1]["close"])
            breakout_close = float(breakout["close"])
            breakout_atr = self._underlying_atr(rows[: index + 1])
            breakout_buffer = (
                max(0.0, _env_float("ORB_BREAKOUT_BUFFER_ATR", 0.05)) * breakout_atr
            )
            if side == "CE":
                fresh = (
                    previous_close <= boundary + breakout_buffer
                    and breakout_close > boundary + breakout_buffer
                )
            else:
                fresh = (
                    previous_close >= boundary - breakout_buffer
                    and breakout_close < boundary - breakout_buffer
                )
            if not fresh:
                continue

            bar_range = max(float(breakout["high"]) - float(breakout["low"]), 1e-9)
            body_ratio = (
                abs(float(breakout["close"]) - float(breakout["open"])) / bar_range
            )
            if body_ratio < max(0.0, _env_float("ORB_MIN_BREAKOUT_BODY_PCT", 0.35)):
                continue
            penetration_atr = abs(breakout_close - boundary) / max(breakout_atr, 1e-9)
            volume_ratio = self._volume_ratio_at(rows, index)
            breakout_tolerance = (
                max(0.0, _env_float("ORB_RETEST_TOLERANCE_ATR", 0.15)) * breakout_atr
            )
            momentum_emitted = bool(
                _env_bool("ORB_MOMENTUM_BRANCH_ENABLED", True)
                and body_ratio >= _env_float("ORB_MOMENTUM_MIN_BODY_PCT", 0.60)
                and penetration_atr
                >= _env_float("ORB_MOMENTUM_MIN_PENETRATION_ATR", 0.20)
                and volume_ratio >= _env_float("ORB_MOMENTUM_MIN_VOLUME_RATIO", 1.20)
            )
            if momentum_emitted:
                return None

            for following in rows[index + 1 : -1]:
                if (
                    self._classify_retest_bar(
                        following,
                        side=side,
                        boundary=boundary,
                        tolerance=breakout_tolerance,
                    )
                    is not None
                ):
                    return None

            return {
                "status": "AWAITING_RETEST",
                "boundary": boundary,
                "breakout_timestamp": breakout_ts,
                "body_ratio": body_ratio,
                "penetration_atr": penetration_atr,
                "volume_ratio": volume_ratio,
                "underlying_atr": breakout_atr,
                "recovered_from_history": True,
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

    @staticmethod
    def _classify_retest_bar(
        row: Mapping[str, Any],
        *,
        side: str,
        boundary: float,
        tolerance: float,
    ) -> str | None:
        """Classify one completed underlying bar against an active ORB event."""
        close = float(row["close"])
        if side == "CE":
            if close < boundary - tolerance:
                return "INVALIDATED"
            if float(row["low"]) <= boundary + tolerance and close >= boundary:
                return "RETESTED"
        else:
            if close > boundary + tolerance:
                return "INVALIDATED"
            if float(row["high"]) >= boundary - tolerance and close <= boundary:
                return "RETESTED"
        return None

    def _structural_evidence(
        self,
        *,
        side: str,
        direction: str,
        penetration_atr: float,
        volume_ratio: float,
        opening_range_atr: float,
        indicators: Mapping[str, Any],
    ) -> tuple[bool, list[str], bool, float, float]:
        """Return explicit ORB structural setup evidence."""
        reasons = [
            "underlying_opening_range_complete",
            "fresh_underlying_breakout",
            "breakout_event_confirmed",
        ]
        direction_aligned = direction == side
        if direction_aligned:
            reasons.append("underlying_direction_alignment")
        volume_confirmed = volume_ratio >= _env_float("ORB_VOLUME_CONFIRM_RATIO", 1.2)
        if volume_confirmed:
            reasons.append("underlying_volume_confirmation")
        penetration_confirmed = penetration_atr >= _env_float(
            "ORB_PENETRATION_CONFIRM_ATR", 0.2
        )
        if penetration_confirmed:
            reasons.append("normalized_breakout_penetration")
        slope = _safe_float(indicators.get("futures_vwap_slope"))
        slope_aligned = bool(
            slope is not None
            and ((side == "CE" and slope > 0) or (side == "PE" and slope < 0))
        )
        if slope_aligned:
            reasons.append("futures_vwap_slope_alignment")
        balanced_min = max(0.05, _env_float("ORB_BALANCED_RANGE_MIN_ATR", 0.25))
        balanced_max = max(balanced_min, _env_float("ORB_BALANCED_RANGE_MAX_ATR", 2.0))
        balanced_range = balanced_min <= opening_range_atr <= balanced_max
        if balanced_range:
            reasons.append("balanced_opening_range")
        passed = bool(
            direction_aligned
            and volume_confirmed
            and penetration_confirmed
            and balanced_range
            and slope_aligned
        )
        return passed, reasons, balanced_range, balanced_min, balanced_max

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
    ) -> EliteSignal | None:
        option_atr = max(float(indicators.get("atr") or 0.0), current_price * 0.01, 1.0)
        max_stop_pct = max(0.5, _env_float("ORB_PREMIUM_STOP_MAX_PCT", 8.0)) / 100.0
        atr_stop = max(0.5, _env_float("ORB_PREMIUM_STOP_ATR_MULT", 0.75) * option_atr)
        premium_stop_distance = min(atr_stop, max(0.5, current_price * max_stop_pct))
        stop_loss = max(0.05, current_price - premium_stop_distance)
        target_rr = max(1.0, _env_float("ORB_TARGET_RR", 1.8))
        target = current_price + premium_stop_distance * target_rr

        boundary = float(event["boundary"])
        current_underlying_atr = float(snapshot["atr"])
        event_underlying_atr = _safe_float(event.get("underlying_atr"))
        breakout_underlying_atr = max(
            1.0,
            (
                event_underlying_atr
                if event_underlying_atr is not None
                else current_underlying_atr
            ),
        )
        breakout_penetration_atr = float(event["penetration_atr"])
        event_volume_ratio = _safe_float(event.get("volume_ratio"))
        breakout_volume_ratio = (
            event_volume_ratio
            if event_volume_ratio is not None
            else float(snapshot["volume_ratio"])
        )
        invalidation_buffer = max(
            0.05 * breakout_underlying_atr,
            _env_float("ORB_RETEST_TOLERANCE_ATR", 0.15) * breakout_underlying_atr,
        )
        underlying_invalidation = (
            boundary - invalidation_buffer
            if side == "CE"
            else boundary + invalidation_buffer
        )
        current_underlying = float(snapshot["current"]["close"])
        current_penetration_atr = abs(current_underlying - boundary) / max(
            current_underlying_atr, 1e-9
        )
        opening_range_atr = (
            float(snapshot["orb_high"]) - float(snapshot["orb_low"])
        ) / max(breakout_underlying_atr, 1e-9)
        direction = str(
            indicators.get("underlying_direction_bias")
            or indicators.get("direction_bias")
            or ""
        ).upper()
        (
            setup_pass,
            reasons,
            balanced_range,
            balanced_min,
            balanced_max,
        ) = self._structural_evidence(
            side=side,
            direction=direction,
            penetration_atr=breakout_penetration_atr,
            volume_ratio=breakout_volume_ratio,
            opening_range_atr=opening_range_atr,
            indicators=indicators,
        )
        reasons.append("retest_hold" if branch == "retest" else "momentum_acceptance")
        context_fresh = indicators.get("context_fresh") is not False
        if not setup_pass or not context_fresh:
            self._no_vote(
                "orb_structural_contract_not_passed"
                if not setup_pass
                else "underlying_context_stale"
            )
            return None

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
            "signal_domain": "NIFTY_FUTURES",
            "source_domain": "underlying_orb",
            "opening_range_source": source,
            "underlying_symbol": snapshot["symbol"],
            "orb_window_minutes": int(snapshot["orb_minutes"]),
            "opening_range_high": float(snapshot["orb_high"]),
            "opening_range_low": float(snapshot["orb_low"]),
            "opening_range_width_atr": round(opening_range_atr, 4),
            "opening_range_balanced": balanced_range,
            "opening_range_balanced_min_atr": balanced_min,
            "opening_range_balanced_max_atr": balanced_max,
            "opening_range_complete": True,
            "breakout_side": side,
            "breakout_timestamp": breakout_ts.timestamp(),
            "breakout_recovered_from_history": bool(
                event.get("recovered_from_history")
            ),
            "breakout_age_seconds": max(
                0.0,
                (snapshot["current_ts"] - breakout_ts).total_seconds(),
            ),
            "entry_branch": branch,
            "retest_confirmed": branch == "retest",
            "retest_timestamp": (
                retest_timestamp.timestamp() if retest_timestamp else None
            ),
            "underlying_atr": breakout_underlying_atr,
            "underlying_breakout_body_pct": round(float(event["body_ratio"]), 4),
            "underlying_volume_ratio": round(breakout_volume_ratio, 4),
            "underlying_penetration_atr": round(breakout_penetration_atr, 4),
            "underlying_current_atr": current_underlying_atr,
            "underlying_current_volume_ratio": round(
                float(snapshot["volume_ratio"]), 4
            ),
            "underlying_current_penetration_atr": round(current_penetration_atr, 4),
            "underlying_atr_basis": snapshot["underlying_atr_basis"],
            "underlying_volume_ratio_basis": snapshot["underlying_volume_ratio_basis"],
            "underlying_entry": current_underlying,
            "underlying_invalidation": underlying_invalidation,
            "premium_stop_distance": premium_stop_distance,
            "premium_target_rr": target_rr,
            "setup_pass": True,
            "requires_runner_execution_validation": True,
            "setup_reasons": reasons,
            "context_fresh": context_fresh,
            "required_data_present": True,
            "stale_data_used": bool(indicators.get("stale_data_used")),
            "direction_bias": side,
            "invalidation_level": stop_loss,
            "latest_bar_ts": snapshot["current_ts"].timestamp(),
            "setup_candle_timestamp": snapshot["current_ts"].timestamp(),
            "rejection_reasons": [],
        }
        LOGGER.info(
            "STRATEGY_EVIDENCE strategy=ORBProV2 side=%s branch=%s "
            "source=%s underlying=%s opening_range_atr=%.3f",
            side,
            branch,
            source,
            snapshot["symbol"],
            opening_range_atr,
        )
        return EliteSignal(
            symbol=symbol,
            signal="BUY",
            confidence=1.0,
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
        regime = normalize_regime(indicators.get("regime"))
        if regime in {MarketRegime.RANGE, MarketRegime.LOW_ACTIVITY}:
            self._no_vote("non_breakout_regime")
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
        previous_close = float(snapshot["previous"]["close"])
        current_bar = snapshot["current"]
        current_close = float(current_bar["close"])

        event = self._events.get(key)
        if event is not None and event.get("status") == "EMITTED":
            back_inside = (
                current_close <= orb_high if side == "CE" else current_close >= orb_low
            )
            if back_inside:
                self._events.pop(key, None)
                event = None
            else:
                self._no_vote("orb_event_already_emitted")
                return None
        if event is not None and event.get("status") in {
            "INVALIDATED",
            "EXPIRED",
            "CONTRACT_REJECTED",
        }:
            back_inside = (
                current_close <= orb_high if side == "CE" else current_close >= orb_low
            )
            if back_inside:
                self._events.pop(key, None)
                event = None
            else:
                self._no_vote("orb_event_inactive")
                return None

        if event is None:
            event = self._recover_unconfirmed_breakout(
                snapshot,
                side=side,
            )
            if event is not None:
                self._events[key] = event

        if event is None:
            max_events = max(1, _env_int("ORB_MAX_EVENTS_PER_SIDE", 2))
            if self._event_count_by_key.get(key, 0) >= max_events:
                self._no_vote("orb_session_event_limit")
                return None
            if side == "CE":
                boundary = orb_high
                fresh_breakout = (
                    previous_close <= boundary + buffer
                    and current_close > boundary + buffer
                )
            else:
                boundary = orb_low
                fresh_breakout = (
                    previous_close >= boundary - buffer
                    and current_close < boundary - buffer
                )
            if not fresh_breakout:
                self._no_vote("no_fresh_underlying_breakout")
                return None

            body_ratio = float(snapshot["body_ratio"])
            if body_ratio < max(0.0, _env_float("ORB_MIN_BREAKOUT_BODY_PCT", 0.35)):
                self._no_vote("wick_only_underlying_breakout")
                return None
            penetration_atr = abs(current_close - boundary) / max(underlying_atr, 1e-9)
            volume_ratio = float(snapshot["volume_ratio"])
            event = {
                "status": "AWAITING_RETEST",
                "boundary": boundary,
                "breakout_timestamp": current_ts,
                "body_ratio": body_ratio,
                "penetration_atr": penetration_atr,
                "volume_ratio": volume_ratio,
                "underlying_atr": underlying_atr,
            }
            self._events[key] = event

            momentum_enabled = _env_bool("ORB_MOMENTUM_BRANCH_ENABLED", True)
            momentum_confirmed = bool(
                momentum_enabled
                and body_ratio >= _env_float("ORB_MOMENTUM_MIN_BODY_PCT", 0.60)
                and penetration_atr
                >= _env_float("ORB_MOMENTUM_MIN_PENETRATION_ATR", 0.20)
                and volume_ratio >= _env_float("ORB_MOMENTUM_MIN_VOLUME_RATIO", 1.20)
            )
            setup_id = (
                f"orbv2:{snapshot['session_date']}:{snapshot['symbol']}:{side}:"
                f"{current_ts.isoformat()}"
            )
            transition_setup(
                SetupStage.ARMED,
                strategy="ORBPro",
                setup_id=setup_id,
                symbol=symbol,
                side=side,
                reason="underlying_breakout",
            )
            if momentum_confirmed:
                transition_setup(
                    SetupStage.CONFIRMING,
                    strategy="ORBPro",
                    setup_id=setup_id,
                    symbol=symbol,
                    side=side,
                    reason="orb_momentum_confirmation",
                )
                signal = self._build_signal(
                    symbol=symbol,
                    side=side,
                    current_price=current_price,
                    indicators=indicators,
                    snapshot=snapshot,
                    event=event,
                    branch="momentum",
                    retest_timestamp=None,
                )
                if signal is None:
                    event["status"] = "CONTRACT_REJECTED"
                    transition_setup(
                        SetupStage.CONTRACT_REJECTED,
                        strategy="ORBPro",
                        setup_id=setup_id,
                        symbol=symbol,
                        side=side,
                        reason="orb_structural_contract_rejected",
                    )
                    return None
                event["status"] = "EMITTED"
                self._event_count_by_key[key] = self._event_count_by_key.get(key, 0) + 1
                return signal
            transition_setup(
                SetupStage.CONFIRMING,
                strategy="ORBPro",
                setup_id=setup_id,
                symbol=symbol,
                side=side,
                reason="awaiting_orb_retest",
            )
            self._no_vote("awaiting_orb_retest")
            return None

        breakout_ts = event["breakout_timestamp"]
        breakout_age = max(0.0, (current_ts - breakout_ts).total_seconds())
        if breakout_age > max(60.0, _env_float("ORB_BREAKOUT_MAX_AGE_SECONDS", 600.0)):
            event["status"] = "EXPIRED"
            transition_setup(
                SetupStage.EXPIRED,
                strategy="ORBPro",
                setup_id=(
                    f"orbv2:{snapshot['session_date']}:{snapshot['symbol']}:{side}:"
                    f"{breakout_ts.isoformat()}"
                ),
                symbol=symbol,
                side=side,
                reason="orb_breakout_expired",
            )
            self._no_vote("orb_breakout_expired")
            return None

        boundary = float(event["boundary"])
        event_underlying_atr = _safe_float(event.get("underlying_atr"))
        event_atr = max(
            1.0,
            (
                event_underlying_atr
                if event_underlying_atr is not None
                else underlying_atr
            ),
        )
        event_tolerance = (
            max(0.0, _env_float("ORB_RETEST_TOLERANCE_ATR", 0.15)) * event_atr
        )
        prior_resolution: str | None = None
        for following in snapshot["rows"]:
            following_ts = following["timestamp"]
            if not breakout_ts < following_ts < current_ts:
                continue
            prior_resolution = self._classify_retest_bar(
                following,
                side=side,
                boundary=boundary,
                tolerance=event_tolerance,
            )
            if prior_resolution is not None:
                break

        setup_id = (
            f"orbv2:{snapshot['session_date']}:{snapshot['symbol']}:{side}:"
            f"{breakout_ts.isoformat()}"
        )
        if prior_resolution == "INVALIDATED":
            event["status"] = "INVALIDATED"
            transition_setup(
                SetupStage.INVALIDATED,
                strategy="ORBPro",
                setup_id=setup_id,
                symbol=symbol,
                side=side,
                reason="orb_breakout_invalidated",
            )
            self._no_vote("orb_breakout_invalidated")
            return None
        if prior_resolution == "RETESTED":
            event["status"] = "EXPIRED"
            transition_setup(
                SetupStage.EXPIRED,
                strategy="ORBPro",
                setup_id=setup_id,
                symbol=symbol,
                side=side,
                reason="orb_retest_missed",
            )
            self._no_vote("orb_retest_missed")
            return None

        current_resolution = self._classify_retest_bar(
            current_bar,
            side=side,
            boundary=boundary,
            tolerance=event_tolerance,
        )
        invalidated = current_resolution == "INVALIDATED"
        retest = current_ts > breakout_ts and current_resolution == "RETESTED"
        if invalidated:
            event["status"] = "INVALIDATED"
            transition_setup(
                SetupStage.INVALIDATED,
                strategy="ORBPro",
                setup_id=setup_id,
                symbol=symbol,
                side=side,
                reason="orb_breakout_invalidated",
            )
            self._no_vote("orb_breakout_invalidated")
            return None
        if not retest:
            self._no_vote("awaiting_orb_retest")
            return None

        signal = self._build_signal(
            symbol=symbol,
            side=side,
            current_price=current_price,
            indicators=indicators,
            snapshot=snapshot,
            event=event,
            branch="retest",
            retest_timestamp=current_ts,
        )
        if signal is None:
            event["status"] = "CONTRACT_REJECTED"
            transition_setup(
                SetupStage.CONTRACT_REJECTED,
                strategy="ORBPro",
                setup_id=setup_id,
                symbol=symbol,
                side=side,
                reason="orb_structural_contract_rejected",
            )
            return None
        event["status"] = "EMITTED"
        self._event_count_by_key[key] = self._event_count_by_key.get(key, 0) + 1
        return signal


__all__ = ["ORBProStrategy"]
