# Path: src/strategies/runner.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd

from src.config import settings
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.execution.order_executor import OrderExecutor

# Broker SDK
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:
    KiteConnect = None  # type: ignore

# Data source
try:
    from src.data.source import LiveKiteSource  # type: ignore
except Exception:
    LiveKiteSource = None  # type: ignore


@dataclass
class RiskState:
    trading_day: datetime
    trades_today: int = 0
    consecutive_losses: int = 0
    day_realized_loss: float = 0.0
    day_realized_pnl: float = 0.0


class StrategyRunner:
    """
    Core trading loop: data → signal → risk → sizing → execution.
    TelegramController is attached by main.py.
    """

    def __init__(self, kite: Optional[KiteConnect] = None, telegram_controller: Any = None) -> None:
        self.log = logging.getLogger(self.__class__.__name__)
        self.kite = kite
        self.telegram = telegram_controller or None
        self.telegram_controller = telegram_controller or None

        # Core modules
        self.strategy = EnhancedScalpingStrategy()
        self.executor = OrderExecutor(kite=self.kite, telegram_controller=self.telegram)

        # Data source
        self.data_source = None
        self._last_fetch_ts: float = 0.0
        if LiveKiteSource is not None:
            try:
                self.data_source = LiveKiteSource(kite=self.kite)
                self.data_source.connect()
                self.log.info("Data source initialized: LiveKiteSource")
            except Exception as e:
                self.log.warning(f"Data source init failed; continuing without: {e}")

        # Risk state
        self.risk = RiskState(trading_day=self._today_ist())
        self._equity_last_refresh_ts: float = 0.0
        self._equity_cached_value: float = float(settings.risk.default_equity)
        self._max_daily_loss_rupees: float = self._equity_cached_value * float(
            settings.risk.max_daily_drawdown_pct
        )

        # Trading window
        self._start_time = self._parse_hhmm(settings.data.time_filter_start)
        self._end_time = self._parse_hhmm(settings.data.time_filter_end)

        # Debug state
        self._paused: bool = False
        self._last_signal_debug: Dict[str, Any] = {"note": "no_eval"}
        self._last_flow_debug: Dict[str, Any] = {"note": "no_flow"}
        self._last_error: Optional[str] = None
        self._last_signal_at: float = 0.0

        self.log.info(
            "StrategyRunner ready (live_trading=%s, use_live_equity=%s)",
            settings.enable_live_trading, settings.risk.use_live_equity
        )

    # ---------------------------------------------------------
    # Main loop entry
    # ---------------------------------------------------------
    def process_tick(self, tick: Optional[Dict[str, Any]]) -> None:
        flow: Dict[str, Any] = {
            "within_window": False, "paused": self._paused, "data_ok": False, "bars": 0,
            "signal_ok": False, "rr_ok": True, "risk_gates": {}, "sizing": {}, "qty": 0,
            "executed": False, "reason_block": None,
        }

        try:
            # Window
            if not self._within_trading_window() and not settings.allow_offhours_testing:
                flow["reason_block"] = "off_hours"
                self._last_flow_debug = flow
                return
            flow["within_window"] = True

            # Pause
            if self._paused:
                flow["reason_block"] = "paused"
                self._last_flow_debug = flow
                return

            # Reset day + refresh equity
            self._ensure_day_state()
            self._refresh_equity_if_due()

            # Data
            df = self._fetch_spot_ohlc()
            flow["bars"] = int(len(df)) if isinstance(df, pd.DataFrame) else 0
            if df is None or len(df) < int(settings.strategy.min_bars_for_signal):
                flow["reason_block"] = "insufficient_data"
                self._last_flow_debug = flow
                return
            flow["data_ok"] = True

            # Signal
            signal = self.strategy.generate_signal(df, current_tick=tick)
            self._last_signal_debug = getattr(self.strategy, "get_debug", lambda: {})()
            if not signal:
                flow["reason_block"] = self._last_signal_debug.get("reason_block", "no_signal")
                self._last_flow_debug = flow
                return
            flow["signal_ok"] = True

            # RR check
            rr_min = float(settings.strategy.rr_min)
            rr_val = float(signal.get("rr", 0.0))
            if rr_val and rr_val < rr_min:
                flow["rr_ok"] = False
                flow["reason_block"] = f"rr<{rr_min}"
                flow["signal"] = {"rr": rr_val, "rr_min": rr_min}
                self._last_flow_debug = flow
                return

            # Risk gates
            gates = self._risk_gates_for(signal)
            flow["risk_gates"] = gates
            if not all(gates.values()):
                flow["reason_block"] = "risk_gate_block"
                self._last_flow_debug = flow
                return

            # Sizing
            qty, diag = self._calculate_quantity_diag(
                entry=float(signal["entry_price"]),
                stop=float(signal["stop_loss"]),
                lot_size=int(settings.instruments.nifty_lot_size),
                equity=self._active_equity(),
            )
            flow["sizing"] = diag
            flow["qty"] = int(qty)
            if qty <= 0:
                flow["reason_block"] = "qty_zero"
                self._last_flow_debug = flow
                return

            # Execution
            placed_ok = False
            if hasattr(self.executor, "place_order"):
                placed_ok = bool(self.executor.place_order({
                    "action": signal["action"],
                    "quantity": int(qty),
                    "entry_price": float(signal["entry_price"]),
                    "stop_loss": float(signal["stop_loss"]),
                    "take_profit": float(signal["take_profit"]),
                    "strike": float(signal["strike"]),
                    "option_type": signal["option_type"],
                }))
            elif hasattr(self.executor, "place_entry_order"):
                side = "BUY" if str(signal["action"]).upper() == "BUY" else "SELL"
                token = int(getattr(settings.instruments, "instrument_token", 0))
                oid = self.executor.place_entry_order(
                    token=token,
                    symbol=settings.instruments.trade_symbol,
                    side=side,
                    quantity=int(qty),
                    price=float(signal["entry_price"]),
                )
                placed_ok = bool(oid)
                if placed_ok and hasattr(self.executor, "setup_gtt_orders"):
                    try:
                        self.executor.setup_gtt_orders(
                            record_id=oid,
                            sl_price=float(signal["stop_loss"]),
                            tp_price=float(signal["take_profit"]),
                        )
                    except Exception as e:
                        self.log.warning("setup_gtt_orders failed: %s", e)

            flow["executed"] = placed_ok
            if not placed_ok:
                flow["reason_block"] = getattr(self.executor, "last_error", "exec_fail")

            if placed_ok:
                self.risk.trades_today += 1
                self._last_signal_at = time.time()
                self._notify(
                    f"✅ Placed: {signal['action']} {qty} {signal['option_type']} {int(signal['strike'])} "
                    f"@ {float(signal['entry_price']):.2f} (SL {float(signal['stop_loss']):.2f}, "
                    f"TP {float(signal['take_profit']):.2f})"
                )

            self._last_flow_debug = flow

        except Exception as e:
            flow["reason_block"] = f"exception:{e.__class__.__name__}"
            self._last_error = str(e)
            self._last_flow_debug = flow
            self.log.exception("process_tick error: %s", e)

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _refresh_equity_if_due(self, silent: bool = False) -> None:
        now = time.time()
        if not settings.risk.use_live_equity:
            self._max_daily_loss_rupees = self._equity_cached_value * settings.risk.max_daily_drawdown_pct
            return
        if (now - self._equity_last_refresh_ts) < settings.risk.equity_refresh_seconds:
            return

        new_eq = None
        if self.kite:
            try:
                margins = self.kite.margins()
                if isinstance(margins, dict):
                    for k in ("equity", "available", "net", "final", "cash"):
                        v = margins.get(k)
                        if isinstance(v, (int, float)):
                            new_eq = float(v)
                            break
                if new_eq is None:
                    new_eq = settings.risk.default_equity
            except Exception as e:
                if not silent:
                    self.log.warning("Equity refresh failed; fallback: %s", e)

        self._equity_cached_value = float(new_eq) if (isinstance(new_eq, (int, float)) and new_eq > 0) \
            else settings.risk.default_equity
        self._max_daily_loss_rupees = self._equity_cached_value * settings.risk.max_daily_drawdown_pct
        self._equity_last_refresh_ts = now

        if not silent:
            self.log.info(
                "Equity snapshot: ₹%s | Max daily loss: ₹%s",
                f"{self._equity_cached_value:,.0f}", f"{self._max_daily_loss_rupees:,.0f}"
            )

    def _active_equity(self) -> float:
        return self._equity_cached_value if settings.risk.use_live_equity else settings.risk.default_equity

    def _risk_gates_for(self, signal: Dict[str, Any]) -> Dict[str, bool]:
        gates = {
            "equity_floor": True,
            "daily_drawdown": True,
            "loss_streak": True,
            "trades_per_day": True,
            "sl_valid": True,
        }
        if settings.risk.use_live_equity and self._active_equity() < settings.risk.min_equity_floor:
            gates["equity_floor"] = False
        if self.risk.day_realized_loss >= self._max_daily_loss_rupees:
            gates["daily_drawdown"] = False
        if self.risk.consecutive_losses >= settings.risk.consecutive_loss_limit:
            gates["loss_streak"] = False
        if self.risk.trades_today >= settings.risk.max_trades_per_day:
            gates["trades_per_day"] = False
        if abs(signal["entry_price"] - signal["stop_loss"]) <= 0:
            gates["sl_valid"] = False
        return gates

    def _calculate_quantity_diag(self, *, entry: float, stop: float, lot_size: int, equity: float) -> Tuple[int, Dict]:
        risk_rupees = equity * settings.risk.risk_per_trade
        sl_points = abs(entry - stop)
        rupee_risk_per_lot = sl_points * lot_size
        if rupee_risk_per_lot <= 0:
            return 0, {}

        lots_raw = int(risk_rupees // rupee_risk_per_lot)
        lots = max(lots_raw, settings.instruments.min_lots)
        lots = min(lots, settings.instruments.max_lots)

        notional = entry * lot_size * lots
        max_notional = equity * settings.risk.max_position_size_pct
        if max_notional > 0 and notional > max_notional:
            denom = entry * lot_size
            lots_cap = int(max_notional // denom) if denom > 0 else 0
            lots = max(min(lots_cap, lots), 0)

        qty = lots * lot_size
        diag = {
            "entry": entry, "stop": stop, "equity": equity,
            "sl_points": sl_points, "lots_raw": lots_raw,
            "lots_final": lots, "exposure": notional,
        }
        return qty, diag

    # ---------------------------------------------------------
    # Data helpers
    # ---------------------------------------------------------
    def _fetch_spot_ohlc(self) -> Optional[pd.DataFrame]:
        if not self.data_source:
            return None
        try:
            lookback = settings.data.lookback_minutes
            end = self._now_ist().replace(second=0, microsecond=0)
            start = end - timedelta(minutes=lookback)

            token = getattr(settings.instruments, "instrument_token", 0) or \
                    getattr(settings.instruments, "spot_token", 0)

            self._last_fetch_ts = time.time()
            if token and token > 0:
                df = self.data_source.fetch_ohlc(token=token, start=start, end=end, timeframe=settings.data.timeframe)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df.sort_index()

            # fallback: last price synthetic
            sym = settings.instruments.trade_symbol
            ltp = self.data_source.get_last_price(sym if sym else token)
            if isinstance(ltp, (int, float)) and ltp > 0:
                ts = end
                return pd.DataFrame(
                    {"open": [ltp], "high": [ltp], "low": [ltp], "close": [ltp], "volume": [0]},
                    index=[ts],
                )
            return None
        except Exception as e:
            self.log.warning("OHLC fetch failed: %s", e)
            return None

    # ---------------------------------------------------------
    # Session / diagnostics
    # ---------------------------------------------------------
    def _ensure_day_state(self) -> None:
        today = self._today_ist()
        if today.date() != self.risk.trading_day.date():
            self.risk = RiskState(trading_day=today)
            self._notify("🔁 New trading day — risk counters reset")

    def _within_trading_window(self) -> bool:
        now = self._now_ist().time()
        return self._start_time <= now <= self._end_time

    @staticmethod
    def _parse_hhmm(text: str):
        return datetime.strptime(text, "%H:%M").time()

    @staticmethod
    def _now_ist():
        return datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)

    @staticmethod
    def _today_ist():
        now = StrategyRunner._now_ist()
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    # ---------------------------------------------------------
    # Telegram helpers
    # ---------------------------------------------------------
    def get_last_signal_debug(self) -> Dict[str, Any]:
        return dict(self._last_signal_debug)

    def build_diag(self) -> Dict[str, Any]:
        return self._build_diag_bundle()

    def get_last_flow_debug(self) -> Dict[str, Any]:
        return self._build_diag_bundle()

    def _build_diag_bundle(self) -> Dict[str, Any]:
        checks: List[Dict[str, Any]] = []

        checks.append({
            "name": "Telegram wiring",
            "ok": bool(self.telegram),
            "detail": "attached" if self.telegram else "missing",
        })

        live = bool(settings.enable_live_trading)
        checks.append({
            "name": "Broker session",
            "ok": bool(self.kite) if live else True,
            "detail": "live+ok" if (live and self.kite) else ("dry" if not live else "live but kite=None"),
        })

        age_s = (time.time() - self._last_fetch_ts) if self._last_fetch_ts else 1e9
        checks.append({
            "name": "Data feed",
            "ok": age_s < 120,
            "detail": "fresh" if age_s < 120 else "stale",
        })

        ready = self._last_flow_debug.get("bars", 0) >= settings.strategy.min_bars_for_signal
        checks.append({"name": "Strategy readiness", "ok": ready, "detail": f"bars={self._last_flow_debug.get('bars', 0)}"})

        gates = self._last_flow_debug.get("risk_gates", {})
        gates_ok = bool(gates) and all(gates.values())
        checks.append({"name": "Risk gates", "ok": gates_ok, "detail": str(gates)})

        checks.append({"name": "Errors", "ok": self._last_error is None, "detail": self._last_error or "none"})

        ok = all(c.get("ok", False) for c in checks)
        last_sig = (time.time() - self._last_signal_at) < 900 if self._last_signal_at else False

        return {"ok": ok, "checks": checks, "last_signal": last_sig, "last_flow": dict(self._last_flow_debug)}

    def get_compact_diag_summary(self) -> Dict[str, Any]:
        bundle = self._build_diag_bundle()
        flow = bundle.get("last_flow", {})
        return {
            "ok": bundle.get("ok", False),
            "status_messages": {
                "telegram": "ok" if self.telegram else "missing",
                "broker": "ok" if self.kite else "missing",
                "data": "ok" if (time.time() - self._last_fetch_ts) < 120 else "stale",
                "strategy": "ok" if flow.get("bars", 0) >= settings.strategy.min_bars_for_signal else "not_ready",
                "risk": "ok" if all(flow.get("risk_gates", {}).values()) else "blocked",
                "errors": "ok" if not self._last_error else "present",
            },
        }

    def get_status_snapshot(self) -> Dict[str, Any]:
        return {
            "time_ist": self._now_ist().strftime("%Y-%m-%d %H:%M:%S"),
            "live_trading": settings.enable_live