# fmt: off
# ruff: noqa: E501,I001,F841
# mypy: ignore-errors
from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any, Mapping

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import VWAPProStrategyConfig
from nifty_scalper_bot.strategies.setup_lifecycle import SetupStage, transition_setup
from nifty_scalper_bot.strategies.entry_evidence import (
    canonical_max_spread_pct,
    resolve_signal_domain,
)
from nifty_scalper_bot.strategies.runtime_context_contract import resolve_context_age_seconds
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _fmt_optional(value: float | None, digits: int) -> str:
    """Format an optional metric for logging without crashing on None."""
    if value is None:
        return "unavailable"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "unavailable"


def _normalize_thesis_anchor(value: Any) -> str:
    """Use one UTC representation for live and recovered bar identities."""
    text = str(value).strip()
    try:
        if isinstance(value, datetime):
            timestamp = value
        elif len(text) >= 10 and text[4:5] == "-" and text[7:8] == "-":
            timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
        else:
            seconds = float(text)
            if seconds > 100_000_000_000:
                seconds /= 1000.0
            timestamp = datetime.fromtimestamp(seconds, tz=timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return str(timestamp.astimezone(timezone.utc))
    except (TypeError, ValueError, OverflowError, OSError):
        return text


def _resolve_session_token(indicators: dict[str, Any], bar_anchor: Any) -> str:
    """Resolve a stable trading-session identity without broker access."""
    explicit = indicators.get("session_date")
    if explicit not in (None, ""):
        return str(explicit).strip()

    if bar_anchor not in (None, ""):
        text = str(bar_anchor).strip()
        if len(text) >= 10 and text[4:5] == "-" and text[7:8] == "-":
            return text[:10]
        try:
            timestamp = float(text)
            if timestamp > 100_000_000_000:
                timestamp /= 1000.0
            if timestamp > 0:
                return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()
        except (TypeError, ValueError, OverflowError, OSError):
            pass

    # No session proof means no previously armed, dated thesis can match this
    # scope. This is deliberately fail-closed rather than reusing another day.
    return "unknown"


class VWAPProStrategy(EliteStrategy):
    """VWAP continuation/pullback strategy emitting structural entry evidence."""

    MIN_BARS_REQUIRED = 10
    ROLE = "trigger"
    TRIGGER_KEY = "vwap_pro"

    def __init__(self, config: VWAPProStrategyConfig, indicator_engine: Any) -> None:
        """Args: config, indicator_engine. Returns: None. Raises: Exception."""
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config
        # The exposed EMA period has one native role: warm-up/history sufficiency.
        # Direction remains owned by the underlying context engine; there is no
        # second EMA direction calculation inside VWAPPro.
        self.MIN_BARS_REQUIRED = max(10, int(self._cfg.ema_period or 10))
        self._allow_pullback = str(os.getenv("VWAP_ALLOW_PULLBACK_ENTRY", "1")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._allow_penetration_only = str(
            os.getenv("VWAP_ALLOW_PENETRATION_ONLY_ENTRY", "0")
        ).lower() in {"1", "true", "yes", "on"}

        # Keep the historical option-distance environment value as a fraction
        # (0.18 == 18%) to avoid silently tightening live option-premium
        # geometry.  The new name makes the unit explicit; the old name remains
        # a compatibility fallback.
        max_distance_raw = os.getenv("VWAP_MAX_OPTION_DISTANCE_FRACTION")
        if max_distance_raw in (None, ""):
            max_distance_raw = os.getenv("VWAP_MAX_OPTION_DISTANCE_PCT", "0.18")
        self._max_distance_fraction = max(0.0, float(max_distance_raw or 0.18))
        self._max_distance_pct = self._max_distance_fraction  # legacy attribute

        self._max_atr_distance_mult = float(
            os.getenv("VWAP_MAX_ATR_DISTANCE_MULT", "1.5") or 1.5
        )
        self._quality_max_distance_atr = max(
            0.5, float(os.getenv("VWAP_QUALITY_MAX_DISTANCE_ATR", "2.0") or 2.0)
        )
        self._trend_quality_max_distance_atr = max(
            self._quality_max_distance_atr,
            float(os.getenv("VWAP_TREND_QUALITY_MAX_DISTANCE_ATR", "3.0") or 3.0),
        )
        self._min_penetration_atr = max(
            0.0,
            float(os.getenv("VWAP_MIN_PENETRATION_ATR_MULT", "0.15") or 0.15),
        )

        # Express reclaim depth directly in ATR units.  Preserve the legacy
        # VWAP_SLACK_ATR_MULT contract as a fallback (1.5 * 0.2 == 0.30 ATR).
        reclaim_depth_raw = os.getenv("VWAP_RECLAIM_MIN_DEPTH_ATR")
        if reclaim_depth_raw in (None, ""):
            legacy_slack = float(os.getenv("VWAP_SLACK_ATR_MULT", "1.5") or 1.5)
            reclaim_depth_raw = str(legacy_slack * 0.2)
        self._reclaim_min_depth_atr = max(0.0, float(reclaim_depth_raw))
        self._slack_atr_mult = self._reclaim_min_depth_atr / 0.2

        self._option_volume_min_ratio = max(
            0.0, float(os.getenv("VWAP_OPTION_VOLUME_MIN_RATIO", "0.6") or 0.6)
        )
        self._futures_volume_min_ratio = max(
            0.0, float(os.getenv("VWAP_FUTURES_VOLUME_MIN_RATIO", "1.0") or 1.0)
        )

        # get_session_vwap_slope() returns percentage change over finalized
        # bars. Convert percentage points to basis points for a scale-readable
        # threshold. Preserve the legacy epsilon if explicitly configured.
        slope_bps_raw = os.getenv("VWAP_FUTURES_SLOPE_MIN_BPS")
        if slope_bps_raw in (None, ""):
            legacy_slope_eps = os.getenv("VWAP_FUTURES_SLOPE_NEUTRAL_EPS")
            slope_bps_raw = (
                str(float(legacy_slope_eps) * 100.0)
                if legacy_slope_eps not in (None, "")
                else "1.0"
            )
        self._futures_slope_min_bps = max(0.0, float(slope_bps_raw))
        self._futures_slope_neutral_eps = self._futures_slope_min_bps / 100.0

        # A reclaim thesis belongs to one concrete option contract in one
        # trading session. Keying only by CE/PE allowed ATM rotations to inherit
        # another contract's structural state.
        self._thesis_anchor_by_scope: dict[tuple[str, str], str] = {}
        self._early_trend_min_context_conf = float(
            os.getenv("VWAP_EARLY_TREND_MIN_CONTEXT_CONF", "0.90") or 0.90
        )
        self._early_trend_max_context_age = float(
            os.getenv("VWAP_EARLY_TREND_MAX_CONTEXT_AGE", "2.0") or 2.0
        )

    def get_required_indicators(self) -> set[str]:
        """Args: none. Returns: indicators set. Raises: Exception."""
        return {
            "vwap",
            "exchange_vwap",
            "session_vwap",
            "vwap_std",
            "vwap_stddev",
            "atr",
            "close",
            "open",
            "high",
            "low",
            "volume",
            "avg_volume",
            "direction_bias",
            "underlying_direction_bias",
            "underlying_direction_confidence",
            "context_age_seconds",
            "futures_vwap_slope",
            "futures_volume_ratio",
            "spread_pct",
            "quote_depth_valid",
            "tradable_quote",
            "spot_context",
            "futures_context",
            "stale_data_used",
            "data_age_seconds",
            "days_to_expiry",
            "strike_distance_from_atm",
            "minutes_since_open",
        }

    def _recover_thesis_anchor_from_history(
        self,
        symbol: str,
        *,
        session_scope: str,
        vwap: float,
    ) -> str | None:
        """Reconstruct same-contract VWAP state after restart/ATM rotation.

        Only completed bars belonging to the executable option contract and the
        current trading session are considered. This restores state that was
        already observable before process restart; it does not create a new
        trigger, relax a structural prerequisite, or borrow state from another strike.
        """
        if session_scope == "unknown" or vwap <= 0:
            return None
        engine = self._indicator_engine
        getter = getattr(engine, "get_history", None)
        if not callable(getter):
            return None
        try:
            rows = getter(symbol, field="bars")
        except TypeError:
            try:
                rows = getter(symbol)
            except Exception:
                return None
        except Exception:
            return None
        if not rows:
            return None
        raw_lookback = os.getenv("VWAP_THESIS_RECOVERY_LOOKBACK_BARS")
        lookback: int | None = None
        if raw_lookback not in (None, ""):
            try:
                lookback = int(float(raw_lookback))
            except (TypeError, ValueError):
                lookback = None
            if lookback is not None:
                lookback = max(5, min(120, lookback))
        session_rows: list[tuple[Any, float, float, float, float]] = []
        cumulative_turnover = 0.0
        cumulative_volume = 0.0
        for raw in rows:
            if not isinstance(raw, Mapping):
                continue
            if raw.get("is_provisional") is True or raw.get("is_complete") is False:
                continue
            timestamp = raw.get("timestamp") or raw.get("date") or raw.get("time")
            if timestamp in (None, ""):
                continue
            if _resolve_session_token({}, timestamp) != session_scope:
                continue
            try:
                row_open = float(raw.get("open") or 0.0)
                row_high = float(raw.get("high") or raw.get("close") or 0.0)
                row_low = float(raw.get("low") or 0.0)
                row_close = float(raw.get("close") or 0.0)
                row_volume = float(raw.get("volume") or 0.0)
            except (TypeError, ValueError):
                continue
            if min(row_open, row_high, row_low, row_close) <= 0:
                continue
            if row_volume > 0:
                typical_price = (row_high + row_low + row_close) / 3.0
                cumulative_turnover += typical_price * row_volume
                cumulative_volume += row_volume
            if cumulative_volume <= 0:
                continue
            session_rows.append(
                (
                    timestamp,
                    row_open,
                    row_low,
                    row_close,
                    cumulative_turnover / cumulative_volume,
                )
            )
        recovery_rows = session_rows if lookback is None else session_rows[-lookback:]
        for timestamp, row_open, row_low, row_close, row_vwap in reversed(
            recovery_rows
        ):
            if (
                row_close < row_vwap
                or row_open < row_vwap
                or row_low < row_vwap
            ):
                return _normalize_thesis_anchor(timestamp)
        return None

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: dict[str, Any],
        current_price: float,
        position: Any | None = None,
    ) -> EliteSignal | None:
        """Args: symbol, indicators, current_price, position. Returns: EliteSignal|None. Raises: Exception."""
        del position
        try:
            self._no_vote("stale_or_invalid_data")
            vwap = float(
                indicators.get("exchange_vwap")
                or indicators.get("session_vwap")
                or indicators.get("vwap")
                or 0.0
            )
            atr = float(indicators.get("atr") or 0.0)
            close = float(indicators.get("close") or current_price)
            open_price = float(indicators.get("open") or current_price)
            high = float(indicators.get("high") or current_price)
            low = float(indicators.get("low") or current_price)
            vol = float(indicators.get("volume") or 0.0)
            avg_vol = float(indicators.get("avg_volume") or 0.0)
            spread_pct = float(indicators.get("spread_pct") or 0.0)
            spot_ctx = (
                indicators.get("spot_context")
                if isinstance(indicators.get("spot_context"), dict)
                else {}
            )
            fut_ctx = (
                indicators.get("futures_context")
                if isinstance(indicators.get("futures_context"), dict)
                else {}
            )
            direction = str(
                indicators.get("direction_bias")
                or indicators.get("underlying_direction_bias")
                or spot_ctx.get("direction_bias")
                or fut_ctx.get("direction_bias")
                or ""
            ).upper()
            underlying_direction = str(
                indicators.get("underlying_direction_bias")
                or indicators.get("direction_bias")
                or spot_ctx.get("underlying_direction_bias")
                or spot_ctx.get("direction_bias")
                or fut_ctx.get("underlying_direction_bias")
                or fut_ctx.get("direction_bias")
                or ""
            ).upper()
            context_age_seconds = resolve_context_age_seconds(indicators)
            try:
                underlying_direction_confidence = float(
                    indicators.get("underlying_direction_confidence")
                    or spot_ctx.get("underlying_direction_confidence")
                    or fut_ctx.get("underlying_direction_confidence")
                    or 0.0
                )
            except (TypeError, ValueError):
                underlying_direction_confidence = 0.0

            def _optional_float(value: Any) -> float | None:
                try:
                    return None if value is None else float(value)
                except (TypeError, ValueError):
                    return None

            futures_vwap_slope = _optional_float(indicators.get("futures_vwap_slope"))
            futures_volume_ratio = _optional_float(indicators.get("futures_volume_ratio"))
            vwap_stddev = _optional_float(indicators.get("vwap_stddev"))
            if vwap_stddev is None:
                vwap_stddev = _optional_float(indicators.get("vwap_std"))
            futures_slope_bps = (
                None if futures_vwap_slope is None else futures_vwap_slope * 100.0
            )
            option_volume_ratio = vol / avg_vol if avg_vol > 0 else None
            if current_price <= 0 or vwap <= 0:
                self._no_vote("missing_vwap")
                LOGGER.debug("STRATEGY_NO_VOTE strategy=VWAPPro reason=missing_vwap")
                return None

            required_data_present = bool(vwap > 0 and atr >= 0)
            max_data_age = (
                45.0
                if str(os.getenv("EXECUTION_MODE", "SHADOW")).strip().upper()
                == "LIVE"
                else 120.0
            )
            stale_data = bool(indicators.get("stale_data_used")) or float(
                indicators.get("data_age_seconds") or 0.0
            ) > max_data_age
            atr_safe = max(atr, current_price * 0.01, 1.0)
            distance_points = abs(current_price - vwap)
            distance_pct = distance_points / max(vwap, 1e-9)
            distance_atr = distance_points / atr_safe
            configured_proximity_pct = max(0.0, float(self._cfg.proximity_pct))
            configured_proximity_fraction = configured_proximity_pct / 100.0
            near_configured_vwap = distance_pct <= configured_proximity_fraction
            allowed_distance = max(
                self._max_distance_fraction,
                self._max_atr_distance_mult * atr_safe / max(vwap, 1e-9),
            )
            vwap_distance_sigma = (
                distance_points / vwap_stddev
                if vwap_stddev is not None and vwap_stddev > 0
                else None
            )
            symbol_upper = str(symbol or "").upper()
            preliminary_side = (
                "CE"
                if symbol_upper.endswith("CE")
                else "PE"
                if symbol_upper.endswith("PE")
                else ""
            )
            strong_fresh_trend_context = bool(
                preliminary_side in {"CE", "PE"}
                and underlying_direction == preliminary_side
                and underlying_direction_confidence
                >= self._early_trend_min_context_conf
                and context_age_seconds <= self._early_trend_max_context_age
            )
            effective_quality_max_distance_atr = (
                self._trend_quality_max_distance_atr
                if strong_fresh_trend_context
                else self._quality_max_distance_atr
            )
            bar_anchor = next(
                (
                    indicators.get(key)
                    for key in (
                        "latest_bar_ts",
                        "bar_timestamp",
                        "setup_candle_timestamp",
                    )
                    if indicators.get(key) not in (None, "")
                ),
                None,
            )
            symbol_scope = str(symbol or "").strip().upper()
            session_scope = _resolve_session_token(indicators, bar_anchor)
            thesis_scope = (symbol_scope, session_scope)
            thesis_recovered_from_history = False
            thesis_anchor = self._thesis_anchor_by_scope.get(thesis_scope)
            if not thesis_anchor:
                recovered_anchor = self._recover_thesis_anchor_from_history(
                    symbol,
                    session_scope=session_scope,
                    vwap=vwap,
                )
                if recovered_anchor:
                    thesis_anchor = recovered_anchor
                    thesis_recovered_from_history = True
                    self._thesis_anchor_by_scope[thesis_scope] = recovered_anchor
                    LOGGER.info(
                        "VWAP_THESIS_RECOVERED symbol=%s session=%s anchor=%s source=own_completed_history",
                        symbol_scope,
                        session_scope,
                        recovered_anchor,
                        extra={
                            "event": "VWAP_THESIS_RECOVERED",
                            "symbol": symbol_scope,
                            "session": session_scope,
                            "anchor": recovered_anchor,
                            "source": "own_completed_history",
                        },
                    )
            if close < vwap:
                if bar_anchor is not None:
                    self._thesis_anchor_by_scope[thesis_scope] = (
                        _normalize_thesis_anchor(bar_anchor)
                    )
                    thesis_anchor = self._thesis_anchor_by_scope[thesis_scope]
            elif (open_price < vwap or low < vwap) and bar_anchor is not None:
                self._thesis_anchor_by_scope[thesis_scope] = (
                    _normalize_thesis_anchor(bar_anchor)
                )
                thesis_anchor = self._thesis_anchor_by_scope[thesis_scope]

            overextended = bool(
                distance_pct > allowed_distance
                or distance_atr > effective_quality_max_distance_atr
            )
            if overextended:
                self._no_vote("distance_outside_band")
                LOGGER.debug("STRATEGY_NO_VOTE strategy=VWAPPro reason=overextended")
                return None
            if stale_data:
                self._no_vote("stale_data")
                LOGGER.debug("STRATEGY_NO_VOTE strategy=VWAPPro reason=stale_data")
                return None
            max_spread_pct = canonical_max_spread_pct()
            if spread_pct > max_spread_pct:
                self._no_vote("wide_spread")
                LOGGER.debug("STRATEGY_NO_VOTE strategy=VWAPPro reason=wide_spread")
                return None

            reasons: list[str] = []
            contract_side, option_premium_domain, _ = resolve_signal_domain(
                symbol, indicators
            )
            trend_alignment = False
            pullback_flag = False
            continuation_confirmed = False

            if contract_side not in {"CE", "PE"}:
                fallback_side = str(indicators.get("direction_bias") or "").upper()
                if fallback_side in {"CE", "PE"}:
                    contract_side = fallback_side
                elif symbol.upper().endswith("CE"):
                    contract_side = "CE"
                    reasons.append("symbol_side_fallback")
                elif symbol.upper().endswith("PE"):
                    contract_side = "PE"
                    reasons.append("symbol_side_fallback")
                else:
                    self._no_vote("unknown_contract_side")
                    LOGGER.debug(
                        "STRATEGY_NO_VOTE strategy=VWAPPro reason=unknown_contract_side"
                    )
                    return None
            if not option_premium_domain:
                self._no_vote("invalid_price_domain")
                return None

            if close < vwap:
                self._no_vote("vwap_thesis_reset")
                return None
            thesis_anchor = self._thesis_anchor_by_scope.get(thesis_scope) or thesis_anchor
            if not thesis_anchor:
                self._no_vote("vwap_thesis_not_armed")
                return None

            premium_above_vwap = close >= vwap
            candle_body = abs(close - open_price)
            momentum_continuation_confirmed = bool(
                close > open_price and candle_body >= (0.35 * atr_safe)
            )
            reclaim_from_below = bool(
                low <= (vwap - (atr_safe * self._reclaim_min_depth_atr))
                and close >= vwap
            )
            if self._allow_pullback and reclaim_from_below:
                pullback_flag = True

            penetration_atr = max(0.0, close - vwap) / atr_safe
            penetration_confirmed = bool(
                premium_above_vwap
                and penetration_atr >= self._min_penetration_atr
            )
            continuation_hold_confirmed = bool(
                open_price >= vwap
                and low >= vwap
                and close >= vwap
                and penetration_confirmed
            )
            continuation_confirmed = bool(
                momentum_continuation_confirmed or continuation_hold_confirmed
            )

            vol_support = bool(
                option_volume_ratio is not None
                and option_volume_ratio >= self._option_volume_min_ratio
            )
            fut_vol_support = bool(
                futures_volume_ratio is not None
                and futures_volume_ratio >= self._futures_volume_min_ratio
            )
            if vol_support and fut_vol_support:
                volume_confirmation_source = "option_and_futures"
            elif vol_support:
                volume_confirmation_source = "option"
            elif fut_vol_support:
                volume_confirmation_source = "futures"
            else:
                volume_confirmation_source = None

            # Explicit underlying direction owns alignment; generic bias is a
            # fallback only when no usable underlying direction is available.
            bias = (
                underlying_direction
                if underlying_direction in {"CE", "PE"}
                else direction
            )
            if bias in {"CE", "PE"}:
                trend_alignment = bias == contract_side
                if not trend_alignment:
                    reasons.append("direction_conflict")

            if futures_slope_bps is None:
                slope_support = False
                reasons.append("futures_slope_unavailable")
            elif abs(futures_slope_bps) < self._futures_slope_min_bps:
                slope_support = False
                reasons.append("futures_slope_below_floor")
            else:
                slope_support = (
                    (contract_side == "CE" and futures_slope_bps > 0)
                    or (contract_side == "PE" and futures_slope_bps < 0)
                )
                if not slope_support:
                    reasons.append("futures_slope_conflict")

            execution_mode = str(
                os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW"
            ).strip().upper()
            is_live = execution_mode == "LIVE"
            require_alignment_live = str(
                os.getenv("VWAP_PRO_REQUIRE_UNDERLYING_ALIGNMENT_LIVE", "true")
            ).lower() in {"1", "true", "yes", "on"}
            require_alignment_shadow = str(
                os.getenv("VWAP_PRO_REQUIRE_UNDERLYING_ALIGNMENT_SHADOW", "false")
            ).lower() in {"1", "true", "yes", "on"}
            context_fresh = context_age_seconds <= float(
                os.getenv("VWAP_CONTEXT_MAX_AGE_SECONDS", "120") or "120"
            )
            hard_conflict = bool(
                (
                    (is_live and require_alignment_live)
                    or ((not is_live) and require_alignment_shadow)
                )
                and not trend_alignment
                and context_fresh
            )
            if hard_conflict:
                self._no_vote("underlying_direction_conflict")
                return None

            setup_type: str | None = None
            if pullback_flag and continuation_confirmed:
                setup_type = "vwap_reclaim_momentum"
            elif pullback_flag:
                setup_type = "vwap_reclaim"
            elif continuation_confirmed:
                setup_type = "vwap_continuation"
            elif penetration_confirmed:
                if not self._allow_penetration_only:
                    self._no_vote("vwap_penetration_only_disabled")
                    return None
                setup_type = "vwap_penetration"

            event_confirmed = setup_type is not None
            setup_lifecycle_id = (
                f"vwap:{contract_side}:{thesis_anchor}:{session_scope}:{symbol_scope}"
            )
            if is_live:
                transition_setup(
                    SetupStage.ARMED,
                    strategy="VWAPPro",
                    setup_id=setup_lifecycle_id,
                    symbol=symbol,
                    side=contract_side,
                    reason="vwap_thesis_armed",
                )
            if not event_confirmed:
                if is_live:
                    transition_setup(
                        SetupStage.CONFIRMING,
                        strategy="VWAPPro",
                        setup_id=setup_lifecycle_id,
                        symbol=symbol,
                        side=contract_side,
                        reason="vwap_event_unconfirmed",
                    )
                self._no_vote("vwap_event_unconfirmed")
                return None

            # Structural contract: a VWAP continuation/pullback entry must have
            # a premium event, fresh underlying alignment, futures slope support
            # and observed activity.  No downstream component may compensate for a
            # missing prerequisite.
            structural_failures: list[str] = []
            if not premium_above_vwap:
                structural_failures.append("premium_below_vwap")
            if not trend_alignment:
                structural_failures.append("underlying_direction_conflict")
            if not context_fresh:
                structural_failures.append("underlying_context_stale")
            if not slope_support:
                structural_failures.append("futures_slope_not_aligned")
            if not (vol_support or fut_vol_support):
                structural_failures.append("volume_confirmation_missing")
            if structural_failures:
                self._no_vote(structural_failures[0])
                transition_setup(
                    SetupStage.CONTRACT_REJECTED,
                    strategy="VWAPPro",
                    setup_id=setup_lifecycle_id,
                    symbol=symbol,
                    side=contract_side,
                    reason=structural_failures[0],
                )
                LOGGER.info(
                    "VWAP_STRUCTURAL_SETUP_REJECTED symbol=%s side=%s failures=%s",
                    symbol,
                    contract_side,
                    structural_failures,
                    extra={
                        "event": "VWAP_STRUCTURAL_SETUP_REJECTED",
                        "symbol": symbol,
                        "side": contract_side,
                        "failures": structural_failures,
                    },
                )
                return None

            if setup_type == "vwap_reclaim_momentum":
                reasons.extend(["premium_reclaim_vwap", "premium_continuation"])
            elif setup_type == "vwap_reclaim":
                reasons.append("premium_reclaim_vwap")
            elif setup_type == "vwap_continuation":
                reasons.append("premium_continuation")
            else:
                reasons.append("premium_vwap_penetration")
            reasons.extend(
                [
                    "premium_above_vwap",
                    "underlying_direction_alignment",
                    "futures_slope_alignment",
                    "volume_confirmation",
                ]
            )
            metadata = {
                "strategy": "VWAPPro",
                "strategy_name": "VWAPPro",
                "role": "trigger",
                "source_domain": "option_premium",
                "signal_family": "reclaim_structure",
                "trade_side": contract_side,
                "side": contract_side,
                "contract_side": contract_side,
                "setup_id": setup_lifecycle_id,
                # Stop rearm is structural, not merely a later evaluation bar.
                # Keep the exact VWAP reclaim/reset anchor so the same thesis
                # cannot re-enter after a stop just because one minute elapsed.
                "setup_candle_timestamp": thesis_anchor,
                "premium_above_vwap": premium_above_vwap,
                "direction_bias": direction if direction in {"CE", "PE"} else None,
                "underlying_direction_bias": (
                    underlying_direction
                    if underlying_direction in {"CE", "PE"}
                    else None
                ),
                "underlying_direction_confidence": underlying_direction_confidence,
                "context_age_seconds": context_age_seconds,
                "context_fresh": context_fresh,
                "futures_slope_alignment": slope_support,
                "volume_confirmation": bool(vol_support or fut_vol_support),
                "context_direction_used": bias if bias in {"CE", "PE"} else None,
                "requires_runner_execution_validation": True,
                "setup_pass": True,
                "execution_required": True,
                "regime_required": True,
                "strategy_family": "vwap_continuation_pullback",
                "context_required": False,
                "setup_reasons": reasons,
                "setup_type": setup_type,
                "setup_name": setup_type,
                "vwap_event_subtype": setup_type,
                "required_data_present": required_data_present,
                "stale_data_used": stale_data,
                "candidate_symbol": symbol,
                "rejection_reasons": [],
                "vwap": vwap,
                "distance_pct": round(distance_pct, 4),
                "vwap_distance_atr": round(distance_atr, 4),
                "vwap_stddev": vwap_stddev,
                "vwap_distance_sigma": (
                    round(vwap_distance_sigma, 4)
                    if vwap_distance_sigma is not None
                    else None
                ),
                "allowed_distance_pct": round(allowed_distance, 4),
                "allowed_distance_fraction": round(allowed_distance, 4),
                "vwap_max_option_distance_fraction": self._max_distance_fraction,
                "vwap_quality_max_distance_atr": effective_quality_max_distance_atr,
                "vwap_base_quality_max_distance_atr": self._quality_max_distance_atr,
                "vwap_trend_quality_max_distance_atr": self._trend_quality_max_distance_atr,
                "vwap_strong_fresh_trend_context": strong_fresh_trend_context,
                "vwap_configured_proximity_pct": configured_proximity_pct,
                "vwap_configured_proximity_pass": near_configured_vwap,
                "vwap_configured_proximity_role": "telemetry_only",
                "atr": atr_safe,
                "pullback_flag": pullback_flag,
                "trend_alignment": trend_alignment,
                "underlying_context_used": bool(
                    indicators.get("spot_context")
                    or indicators.get("underlying_direction_bias")
                ),
                "futures_context_used": bool(
                    indicators.get("futures_context")
                    or indicators.get("futures_vwap") is not None
                ),
                "futures_vwap_slope": futures_vwap_slope,
                "futures_vwap_slope_bps": futures_slope_bps,
                "futures_vwap_slope_min_bps": self._futures_slope_min_bps,
                "vwap_domain": "option_premium",
                "underlying_alignment": trend_alignment,
                "futures_alignment": slope_support,
                "futures_volume_ratio": futures_volume_ratio,
                "futures_volume_min_ratio": self._futures_volume_min_ratio,
                "option_volume_ratio": option_volume_ratio,
                "option_volume_min_ratio": self._option_volume_min_ratio,
                "volume_confirmation_source": volume_confirmation_source,
                "trigger_block_reason": "",
                "continuation_confirmed": continuation_confirmed,
                "momentum_continuation_confirmed": momentum_continuation_confirmed,
                "continuation_hold_confirmed": continuation_hold_confirmed,
                "reclaim_confirmed": pullback_flag,
                "reclaim_min_depth_atr": self._reclaim_min_depth_atr,
                "penetration_confirmed": penetration_confirmed,
                "penetration_only_enabled": self._allow_penetration_only,
                "vwap_penetration_atr": round(penetration_atr, 4),
                "vwap_event_confirmed": event_confirmed,
                "days_to_expiry": indicators.get("days_to_expiry"),
                "strike_distance_from_atm": indicators.get("strike_distance_from_atm"),
                "minutes_since_open": indicators.get("minutes_since_open"),
                "quality_calibrated": False,
                "quality_probability": None,
                "quality_model": "structural_evidence_only",
                "thesis_scope_symbol": symbol_scope,
                "thesis_scope_session": session_scope,
                "thesis_recovered_from_history": thesis_recovered_from_history,
                "setup_invalidation_premium": current_price - atr_safe,
                "invalidation_level_domain": "option_premium",
                "premium_stop_distance": atr_safe,
                "premium_target_rr": 2.0,
                "direction_conflict_mode": "hard" if hard_conflict else "none",
            }
            LOGGER.info(
                "STRATEGY_EVIDENCE strategy=VWAPPro side=%s setup=%s",
                contract_side,
                metadata.get("setup_type"),
            )
            return EliteSignal(
                symbol=symbol,
                signal="BUY",
                confidence=1.0,
                entry_price=current_price,
                stop_loss=None,
                target=None,
                quantity=self._cfg.quantity or 1,
                strategy_name="VWAPPro",
                metadata=metadata,
            )
        except Exception as e:
            LOGGER.error(
                "Failure in VWAPProStrategy._evaluate_signal: %s", e, exc_info=e
            )
            return None


__all__ = ["VWAPProStrategy"]
