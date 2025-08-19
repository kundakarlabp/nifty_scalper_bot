# src/strategies/runner.py
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from src.config import settings
from src.utils import strike_selector as sel
from src.execution.order_executor import OrderExecutor
from src.data.source import LiveKiteSource
from src.risk.position_sizing import PositionSizer
from src.risk.session import TradingSession
from src.notifications.telegram_controller import TelegramController

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    side: Optional[str]
    score: float
    atr: float
    confidence: float
    reason: str = ""


class StrategyRunner:
    def __init__(
        self,
        data_source: LiveKiteSource,
        strategy: Any,
        order_executor: OrderExecutor,
        trading_session: TradingSession,
        position_sizer: PositionSizer,
        telegram_controller: Optional[TelegramController] = None,
    ):
        self.ds = data_source
        self.strategy = strategy
        self.exec = order_executor
        self.session = trading_session
        self.sizer = position_sizer
        self.tg = telegram_controller

        self._running: bool = False
        self._sleep_sec: int = int(os.getenv("PEAK_POLL_SEC", "5"))

        self._cache: Dict[str, Any] = {"NFO": [], "NSE": []}
        self._cache_ts: float = 0.0
        self._cache_ttl_sec: int = 300

        self._opt_sl_pct: float = float(os.getenv("OPTION_SL_PERCENT", "0.06"))
        self._opt_tp_pct: float = float(os.getenv("OPTION_TP_PERCENT", "0.20"))

        self._min_score: int = int(settings.strategy.min_signal_score)
        self._conf_threshold: float = float(settings.strategy.confidence_threshold)

        if self.tg:
            self.tg._config_getter = self._config_get  # type: ignore[attr-defined]
            self.tg._config_setter = self._config_set  # type: ignore[attr-defined]

    def start(self) -> None:
        self._running = True
        try:
            self.ds.connect()
        except Exception as e:
            logger.error("DataSource connect failed: %s", e, exc_info=True)
            return

        kite = getattr(self.ds, "kite", None)
        if kite is None:
            logger.error("Runner requires LiveKiteSource with kite client.")
            return

        logger.info("Runner started. Live:%s TZ=%s", settings.enable_live_trading, os.getenv("TZ", "unset"))

        while self._running:
            try:
                if not self._is_ok_to_trade_now():
                    time.sleep(self._sleep_sec); continue

                self._maybe_refresh_instruments(kite)

                strike_range = getattr(settings.strategy, "strike_selection_range", 3)
                tokens = sel.get_instrument_tokens(
                    symbol=settings.executor.trade_symbol,
                    kite_instance=kite,
                    cached_nfo_instruments=self._cache["NFO"],
                    cached_nse_instruments=self._cache["NSE"],
                    offset=0,
                    strike_range=int(strike_range),
                )
                if not tokens or not tokens.get("spot_token"):
                    time.sleep(self._sleep_sec); continue

                spot_token = int(tokens["spot_token"])
                end = datetime.now()
                start = end - timedelta(minutes=max(5, settings.executor.data_lookback_minutes))
                df = self.ds.get_historical_ohlc(spot_token, start, end, interval="minute")
                if df.empty or len(df) < max(20, settings.strategy.min_bars_for_signal):
                    time.sleep(self._sleep_sec); continue

                sig = self._get_signal(df)
                if not sig or sig.side is None:
                    self._housekeeping(); time.sleep(self._sleep_sec); continue

                if int(sig.score) < self._min_score or sig.confidence < self._conf_threshold:
                    self._housekeeping(); time.sleep(self._sleep_sec); continue

                if sig.side == "UP":
                    opt_sym, leg = tokens.get("ce_symbol"), "CE"
                else:
                    opt_sym, leg = tokens.get("pe_symbol"), "PE"
                if not opt_sym:
                    self._housekeeping(); time.sleep(self._sleep_sec); continue

                lot_size = int(settings.executor.nifty_lot_size)  # set 75 in env/config
                qty = self.sizer.size_for_account(self.session, lot_size)
                qty = max(lot_size, (qty // lot_size) * lot_size)
                if qty <= 0:
                    self._housekeeping(); time.sleep(self._sleep_sec); continue

                opt_ltp = self.exec.get_last_price(opt_sym, exchange=settings.executor.trade_exchange)
                if opt_ltp <= 0.0:
                    self._housekeeping(); time.sleep(self._sleep_sec); continue

                order_id = self.exec.place_entry_order(
                    symbol=opt_sym,
                    exchange=settings.executor.trade_exchange,
                    quantity=qty,
                    transaction_type="BUY",
                    order_type="MARKET",
                )

                sl_px, tp_px = self._compute_option_levels(entry_price=opt_ltp)

                self.exec.setup_gtt_orders(
                    entry_order_id=order_id,
                    entry_price=opt_ltp,
                    stop_loss_price=sl_px,
                    target_price=tp_px,
                    symbol=opt_sym,
                    exchange=settings.executor.trade_exchange,
                    quantity=qty,
                    transaction_type="BUY",
                )

                if self.tg:
                    self.tg.send_message(
                        f"✅ Entry BUY <b>{opt_sym}</b> x{qty} @ <code>{opt_ltp:.2f}</code>\n"
                        f"SL <code>{sl_px:.2f}</code> | TP <code>{tp_px:.2f}</code> "
                        f"(leg={leg}, score={sig.score}, conf={sig.confidence})"
                    )

                if sig.atr > 0:
                    self.exec.update_trailing_stop(order_id, current_price=opt_ltp, atr=max(0.5, sig.atr * 0.2))

                self._housekeeping()

            except Exception as e:
                logger.error("Runner loop error: %s", e, exc_info=True)
                time.sleep(self._sleep_sec)

        logger.info("Runner stopped.")

    def stop(self) -> None:
        self._running = False

    # ----- internals -----

    def _is_ok_to_trade_now(self) -> bool:
        if settings.allow_offhours_testing:
            return True
        return sel.is_trading_hours(settings.executor.market_open, settings.executor.market_close, tz_name=os.getenv("TZ"))

    def _maybe_refresh_instruments(self, kite) -> None:
        now = time.time()
        if now - self._cache_ts >= self._cache_ttl_sec or not self._cache["NFO"]:
            self._cache = sel.fetch_cached_instruments(kite)
            self._cache_ts = now

    def _get_signal(self, df) -> Optional[Signal]:
        try:
            if hasattr(self.strategy, "evaluate"):
                res = self.strategy.evaluate(df)
            elif hasattr(self.strategy, "generate_signal"):
                res = self.strategy.generate_signal(df)
            elif hasattr(self.strategy, "compute_signal"):
                res = self.strategy.compute_signal(df)
            else:
                return None
            side = res.get("side") or res.get("direction")
            if isinstance(side, str):
                s = side.upper()
                side = "UP" if s in ("BUY", "LONG", "UP") else ("DOWN" if s in ("SELL", "SHORT", "DOWN") else None)
            score = float(res.get("score", 0.0))
            confidence = float(res.get("confidence", 0.0))
            atr = float(res.get("atr", 0.0))
            reason = str(res.get("reason", ""))
            return Signal(side=side, score=score, atr=atr, confidence=confidence, reason=reason)
        except Exception:
            return None

    def _compute_option_levels(self, entry_price: float) -> tuple[float, float]:
        sl = max(1.0, entry_price * (1.0 - self._opt_sl_pct))
        tp = max(sl + self.exec.get_tick_size(), entry_price * (1.0 + self._opt_tp_pct))
        return (round(sl, 2), round(tp, 2))

    def _housekeeping(self) -> None:
        for rec in self.exec.get_active_orders():
            ltp = self.exec.get_last_price(rec.symbol, exchange=rec.exchange)
            if ltp > 0:
                atr_proxy = max(0.5, ltp * 0.02)
                self.exec.update_trailing_stop(rec.order_id, current_price=ltp, atr=atr_proxy, trail_mult=0.6)
        self.exec.manage_partials()

    # ----- Telegram /config -----

    def _config_get(self) -> Dict[str, Any]:
        ex = self.exec
        return {
            "poll.sec": self._sleep_sec,
            "opt.sl_pct": self._opt_sl_pct,
            "opt.tp_pct": self._opt_tp_pct,
            "strategy.min_score": self._min_score,
            "strategy.conf_threshold": self._conf_threshold,
            "tp.enable": ex.partial_tp_enable,
            "tp.ratio": ex.partial_tp_ratio,
            "tp2.rmult": ex.partial_tp2_r_mult,
            "be.after_tp1": ex.breakeven_after_tp1,
            "be.offset_ticks": ex.breakeven_offset_ticks,
            "gtt.trail": ex.allow_gtt_trailing,
            "gtt.min_step": ex.gtt_trail_min_step,
            "gtt.cooldown": ex.gtt_trail_cooldown_s,
        }

    def _config_set(self, key: str, value: str) -> str:
        key = key.strip().lower(); v = value.strip(); ex = self.exec
        def as_bool(s: str) -> bool: return s.lower() in ("1", "true", "yes", "y", "on")
        try:
            if key == "poll.sec":               self._sleep_sec = max(1, int(float(v)))
            elif key == "opt.sl_pct":           self._opt_sl_pct = max(0.0, float(v))
            elif key == "opt.tp_pct":           self._opt_tp_pct = max(0.0, float(v))
            elif key == "strategy.min_score":   self._min_score = max(0, int(float(v)))
            elif key == "strategy.conf_threshold": self._conf_threshold = max(0.0, float(v))
            elif key == "tp.enable":            ex.partial_tp_enable = as_bool(v)
            elif key == "tp.ratio":
                r = float(v)
                if not (0.05 <= r <= 0.95): return "tp.ratio must be between 0.05 and 0.95"
                ex.partial_tp_ratio = r
            elif key == "tp2.rmult":            ex.partial_tp2_r_mult = max(0.5, float(v))
            elif key == "be.after_tp1":         ex.breakeven_after_tp1 = as_bool(v)
            elif key == "be.offset_ticks":      ex.breakeven_offset_ticks = max(0, int(float(v)))
            elif key == "gtt.trail":            ex.allow_gtt_trailing = as_bool(v)
            elif key == "gtt.min_step":         ex.gtt_trail_min_step = max(0.0, float(v))
            elif key == "gtt.cooldown":         ex.gtt_trail_cooldown_s = max(1, int(float(v)))
            else:                                return f"unknown key: {key}"
        except Exception as e:
            return f"error: {e}"
        return f"ok: {key} = {v}"
