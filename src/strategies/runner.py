from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.config import settings

# Optional/soft imports – keep the runner tolerant
try:
    from src.data.source import LiveKiteSource  # expected production source
except Exception:  # pragma: no cover
    LiveKiteSource = None  # type: ignore

try:
    from src.utils.strike_selector import get_instrument_tokens
except Exception:  # pragma: no cover
    get_instrument_tokens = None  # type: ignore

try:
    from src.signals.regime_detector import detect_market_regime  # optional
except Exception:  # pragma: no cover
    detect_market_regime = None  # type: ignore

try:
    from src.strategies.scalping_strategy import EnhancedScalpingStrategy
except Exception:  # pragma: no cover
    EnhancedScalpingStrategy = None  # type: ignore

try:
    from src.risk.position_sizing import PositionSizer, SizingInputs, DailyRiskState
except Exception:  # pragma: no cover
    PositionSizer = None  # type: ignore
    SizingInputs = None  # type: ignore
    DailyRiskState = None  # type: ignore

try:
    from src.execution.order_executor import OrderExecutor
except Exception:  # pragma: no cover
    OrderExecutor = None  # type: ignore


log = logging.getLogger(__name__)


def _now_ist() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)


def _within_trading_window(dt_ist: datetime) -> bool:
    if settings.allow_offhours_testing:
        return True
    start = getattr(settings, "TIME_FILTER_START", None)
    end = getattr(settings, "TIME_FILTER_END", None)
    if not start or not end:
        return True
    try:
        hh, mm = [int(x) for x in str(start).split(":")]
        start_t = dtime(hh, mm)
        hh, mm = [int(x) for x in str(end).split(":")]
        end_t = dtime(hh, mm)
        cur_t = dt_ist.time()
        return start_t <= cur_t <= end_t
    except Exception:
        return True


@dataclass
class SignalDecision:
    ok: bool
    side: Optional[str] = None
    score: float = 0.0
    confidence: float = 0.0
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    reasons: Optional[List[str]] = None


class StrategyRunner:
    """
    Orchestrates market data → signals → sizing → (optional) execution.
    Produces rich diagnostics for Telegram (/diag, /flow).
    """

    def __init__(
        self,
        *,
        live: bool = False,
        notifier: Optional[Any] = None,
    ) -> None:
        self.live = bool(live)
        self.notifier = notifier

        self.source = LiveKiteSource() if LiveKiteSource else None
        self.strategy = EnhancedScalpingStrategy() if EnhancedScalpingStrategy else None
        self.sizer = PositionSizer() if PositionSizer else None
        self.executor = OrderExecutor(live=live) if OrderExecutor else None

        self.paused: bool = False
        self.last_diag: Dict[str, Any] = {}
        self.last_signal: Optional[SignalDecision] = None
        self.last_place_error: Optional[str] = None
        self.trades_placed_today: int = 0
        self.loss_streak: int = 0
        self.realized_pnl: float = 0.0
        self.peak_equity: Optional[float] = None

    # ---------- telegram providers ----------

    def status_provider(self) -> Dict[str, Any]:
        dt = _now_ist().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "time_ist": dt,
            "live_trading": self.live,
            "paused": self.paused,
            "active_orders": self._active_order_count(),
            "broker": "Kite" if self.source else "None",
            "quality": getattr(self.strategy, "quality_mode", "auto"),
            "regime_mode": getattr(self.strategy, "regime_mode", "auto"),
        }

    def positions_provider(self) -> Dict[str, Any]:
        try:
            if self.executor and hasattr(self.executor, "get_day_positions"):
                return self.executor.get_day_positions() or {}
        except Exception as e:
            log.warning("positions_provider error: %s", e)
        return {}

    def actives_provider(self) -> List[Any]:
        try:
            if self.executor and hasattr(self.executor, "list_active_orders"):
                return self.executor.list_active_orders() or []
        except Exception as e:
            log.warning("actives_provider error: %s", e)
        return []

    # ---------- internal utils ----------

    def _active_order_count(self) -> int:
        acts = self.actives_provider()
        try:
            return len(acts)
        except Exception:
            return 0

    def _daily_state(self) -> Optional[DailyRiskState]:
        if DailyRiskState is None:
            return None
        return DailyRiskState(
            trades_placed=self.trades_placed_today,
            realized_pnl=self.realized_pnl,
            peak_equity=self.peak_equity,
            loss_streak=self.loss_streak,
        )

    # ---------- external toggles from Telegram ----------

    def pause(self, minutes: Optional[int] = None) -> None:
        self.paused = True
        if self.notifier:
            self.notifier.notify_text("⏸️ Entries paused.")
        if minutes:
            import threading
            def _timer():
                time.sleep(max(1, int(minutes * 60)))
                self.resume()
            threading.Thread(target=_timer, daemon=True).start()

    def resume(self) -> None:
        self.paused = False
        if self.notifier:
            self.notifier.notify_text("▶️ Entries resumed.")

    def diagnostics(self) -> Dict[str, Any]:
        return self.last_diag or {"ok": False, "error": "no run yet"}

    # ---------- core loop step ----------

    def run_once(self) -> None:
        diag: Dict[str, Any] = {"ok": True, "checks": []}
        t0 = _now_ist()

        if self.source is None:
            diag["ok"] = False
            diag["error"] = "market data source unavailable"
            self.last_diag = diag
            return
        if self.strategy is None:
            diag["ok"] = False
            diag["error"] = "strategy unavailable"
            self.last_diag = diag
            return

        is_open = _within_trading_window(t0)
        diag["checks"].append({"name": "market_open", "ok": bool(is_open)})
        if not is_open:
            diag["ok"] = False
            self.last_diag = diag
            return

        # Spot LTP
        spot_token = settings.instruments.instrument_token
        spot_price = None
        try:
            if hasattr(self.source, "get_spot_ltp"):
                spot_price = float(self.source.get_spot_ltp(spot_token))
        except Exception as e:
            logging.warning("spot LTP error: %s", e)
        diag["checks"].append({"name": "spot_ltp", "ok": spot_price is not None, "value": spot_price})
        if spot_price is None:
            diag["ok"] = False
            self.last_diag = diag
            return

        # Strike selection – POSitional arg only
        tokens = None
        try:
            if get_instrument_tokens:
                tokens = get_instrument_tokens(spot_price)
        except TypeError as e:
            try:
                tokens = get_instrument_tokens()  # type: ignore
            except Exception:
                tokens = None
            logging.warning("strike selector signature mismatch: %s", e)
        except Exception as e:
            logging.warning("strike selection error: %s", e)
            tokens = None

        diag_tokens = None
        if tokens and isinstance(tokens, dict):
            diag_tokens = {
                "spot_token": spot_token,
                "spot_price": spot_price,
                "atm_strike": tokens.get("atm_strike"),
                "target_strike": tokens.get("target_strike", tokens.get("atm_strike")),
                "expiry": tokens.get("expiry"),
                "tokens": tokens.get("tokens") or tokens.get("token_map") or {},
            }
        diag["checks"].append({"name": "strike_selection", "ok": tokens is not None, "result": diag_tokens})
        if tokens is None:
            diag["ok"] = False
            self.last_diag = diag
            return

        # OHLC
        spot_ohlc = None
        opt_ohlc = None
        try:
            if hasattr(self.source, "get_ohlc"):
                spot_ohlc = self.source.get_ohlc(spot_token, minutes=settings.data.lookback_minutes)
        except Exception as e:
            logging.warning("spot OHLC error: %s", e)

        ce_token = (tokens.get("tokens") or {}).get("ce")
        pe_token = (tokens.get("tokens") or {}).get("pe")
        chosen_token = ce_token if ce_token else pe_token

        try:
            if hasattr(self.source, "get_ohlc") and chosen_token:
                opt_ohlc = self.source.get_ohlc(chosen_token, minutes=settings.data.lookback_minutes)
        except Exception as e:
            logging.warning("option OHLC error: %s", e)

        diag["checks"].append({"name": "spot_ohlc", "ok": bool(getattr(spot_ohlc, "__len__", lambda: 0)() > 0),
                               "rows": (len(spot_ohlc) if spot_ohlc is not None else 0)})
        diag["checks"].append({"name": "option_ohlc", "ok": bool(getattr(opt_ohlc, "__len__", lambda: 0)() > 0),
                               "rows": (len(opt_ohlc) if opt_ohlc is not None else 0)})
        if not spot_ohlc or not getattr(spot_ohlc, "__len__", lambda: 0)():
            diag["ok"] = False
            self.last_diag = diag
            return
        if not opt_ohlc or not getattr(opt_ohlc, "__len__", lambda: 0)():
            diag["ok"] = False
            self.last_diag = diag
            return

        # Indicators / regime
        try:
            regime = detect_market_regime(spot_ohlc) if detect_market_regime else {"regime": "auto"}
            diag["checks"].append({"name": "indicators", "ok": True, "regime": regime})
        except Exception as e:
            logging.warning("indicator/regime error: %s", e)
            diag["checks"].append({"name": "indicators", "ok": False, "error": str(e)})
            diag["ok"] = False
            self.last_diag = diag
            return

        # Signal
        signal = None
        try:
            if hasattr(self.strategy, "generate_signal"):
                signal = self.strategy.generate_signal(spot_ohlc, opt_ohlc, spot_price=spot_price, regime=regime)
        except Exception as e:
            logging.warning("signal error: %s", e)

        sig_ok = bool(signal and getattr(signal, "ok", False))
        if not sig_ok and isinstance(signal, dict):
            sig_ok = bool(signal.get("ok"))

        if sig_ok:
            sd = SignalDecision(
                ok=True,
                side=getattr(signal, "side", None) or (signal.get("side") if isinstance(signal, dict) else None),
                score=float(getattr(signal, "score", 0.0) or (signal.get("score", 0.0) if isinstance(signal, dict) else 0.0)),
                confidence=float(getattr(signal, "confidence", 0.0) or (signal.get("confidence", 0.0) if isinstance(signal, dict) else 0.0)),
                entry=getattr(signal, "entry_price", None) or (signal.get("entry_price") if isinstance(signal, dict) else None),
                stop=getattr(signal, "stop_loss", None) or (signal.get("stop_loss") if isinstance(signal, dict) else None),
                target=getattr(signal, "target", None) or (signal.get("target") if isinstance(signal, dict) else None),
                reasons=getattr(signal, "reasons", None) or (signal.get("reasons") if isinstance(signal, dict) else None),
            )
            self.last_signal = sd
            diag["checks"].append({"name": "signal", "ok": True, "side": sd.side, "score": sd.score,
                                   "conf": sd.confidence, "entry": sd.entry, "sl": sd.stop, "tp": sd.target})
        else:
            self.last_signal = None
            diag["checks"].append({"name": "signal", "ok": False, "error": "no signal or thresholds not met"})
            diag["ok"] = False
            self.last_diag = diag
            return

        # Sizing
        sizing_ok = False
        sizing_note: Optional[str] = None
        qty = lots = per_order_qty = num_orders = 0
        stop_points = 0.0

        try:
            if self.sizer and SizingInputs and self.last_signal and self.last_signal.entry and self.last_signal.stop:
                equity = getattr(settings.risk, "default_equity", 30000.0)
                if hasattr(self.source, "get_equity"):
                    try:
                        equity = float(self.source.get_equity() or equity)
                    except Exception:
                        pass

                inputs = SizingInputs(
                    equity=equity,
                    risk_per_trade=settings.risk.risk_per_trade,
                    entry_price=float(self.last_signal.entry),
                    stop_price=float(self.last_signal.stop),
                    lot_size=settings.instruments.nifty_lot_size,
                    tick_size=settings.executor.tick_size,
                    freeze_qty=settings.executor.exchange_freeze_qty,
                    min_lots=settings.instruments.min_lots,
                    max_lots=settings.instruments.max_lots,
                    max_trades_per_day=settings.risk.max_trades_per_day,
                    max_daily_drawdown_pct=settings.risk.max_daily_drawdown_pct,
                    max_consecutive_losses=settings.risk.consecutive_loss_limit,
                )
                decision = self.sizer.size(inputs, daily=self._daily_state())
                sizing_ok = bool(decision.ok)
                if decision.ok:
                    qty = decision.qty
                    lots = decision.lots
                    per_order_qty = decision.per_order_qty
                    num_orders = decision.num_orders
                    stop_points = decision.stop_points
                    sizing_note = "; ".join(decision.reasons)
                else:
                    sizing_note = "; ".join(decision.reasons)
        except Exception as e:
            sizing_note = f"exception: {e}"
            logging.warning("sizing error: %s", e)

        diag["checks"].append({
            "name": "sizing",
            "ok": sizing_ok,
            "qty": qty,
            "lots": lots,
            "per_order_qty": per_order_qty,
            "num_orders": num_orders,
            "stop_points": stop_points,
            "note": sizing_note,
        })
        if not sizing_ok:
            diag["ok"] = False
            self.last_diag = diag
            return

        # Execution ready
        exec_ready = bool(self.executor is not None)
        diag["checks"].append({"name": "execution_ready", "ok": exec_ready,
                               "live": self.live, "broker": bool(self.source), "executor": bool(self.executor)})
        if not exec_ready:
            diag["ok"] = False
            self.last_diag = diag
            return

        # Place entry
        if self.live and not self.paused:
            try:
                place_ok = False
                place_msg = None

                if hasattr(self.executor, "place_entry"):
                    place_ok, place_msg = self.executor.place_entry(
                        side=self.last_signal.side,
                        qty=qty,
                        price=self.last_signal.entry,
                        token=chosen_token,
                        per_order_qty=per_order_qty,
                    )
                elif hasattr(self.executor, "enter"):
                    place_ok, place_msg = self.executor.enter(
                        side=self.last_signal.side,
                        qty=qty,
                        price=self.last_signal.entry,
                        token=chosen_token,
                        per_order_qty=per_order_qty,
                    )
                else:
                    place_msg = "executor has no place_entry/enter"
                    place_ok = False

                diag["checks"].append({"name": "place_entry", "ok": bool(place_ok), "msg": place_msg})
                if not place_ok:
                    self.last_place_error = str(place_msg)
                    diag["ok"] = False
                else:
                    self.trades_placed_today += 1
                    if self.notifier:
                        self.notifier.notify_entry(
                            symbol=str(chosen_token),
                            side=str(self.last_signal.side),
                            qty=int(qty),
                            price=float(self.last_signal.entry or 0.0),
                            record_id=str(place_msg or ""),
                        )
            except Exception as e:
                logging.exception("entry placement failed: %s", e)
                self.last_place_error = str(e)
                diag["checks"].append({"name": "place_entry", "ok": False, "error": str(e)})
                diag["ok"] = False
        else:
            diag["checks"].append({"name": "place_entry", "ok": False,
                                   "msg": "skipped (not live or paused)"})

        # Open orders snapshot
        try:
            cnt = self._active_order_count()
            diag["checks"].append({"name": "open_orders", "ok": True, "count": cnt})
        except Exception:
            diag["checks"].append({"name": "open_orders", "ok": False})

        diag["tokens"] = diag_tokens
        self.last_diag = diag

    def loop_forever(self, interval_sec: int = 30) -> None:
        while True:
            try:
                if not self.paused:
                    self.run_once()
            except Exception as e:
                logging.exception("Unexpected error in run loop: %s", e)
            time.sleep(max(5, int(interval_sec)))