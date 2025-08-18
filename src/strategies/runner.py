# src/strategies/runner.py
"""
The StrategyRunner orchestrates data, strategy, execution, and session.
Upgrades:
- Applies playbook (range/trend) trail_mult into trailing updates.
- Greeks-aware follow-up (optional): if greeks resolvable, adapt SL/TP.
- Hooks for Telegram /config get|set.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

from src.config import settings
from src.risk.position_sizing import PositionSizer
from src.risk.session import TradingSession, Trade
from src.signals.regime_detector import detect_market_regime
from src.utils.strike_selector import get_instrument_tokens, is_market_open, fetch_cached_instruments
from src.utils.greeks import estimate_delta, estimate_iv

if TYPE_CHECKING:
    from src.data.source import DataSource
    from src.strategies.scalping_strategy import EnhancedScalpingStrategy
    from src.execution.order_executor import OrderExecutor
    from src.notifications.telegram_controller import TelegramController

logger = logging.getLogger(__name__)


class StrategyRunner:
    def __init__(
        self,
        data_source: "DataSource",
        strategy: "EnhancedScalpingStrategy",
        order_executor: "OrderExecutor",
        trading_session: TradingSession,
        position_sizer: PositionSizer,
        telegram_controller: Optional["TelegramController"] = None,
    ) -> None:
        self.ds = data_source
        self.strategy = strategy
        self.executor = order_executor
        self.session = trading_session
        self.sizer = position_sizer
        self.telegram = telegram_controller

        # expose /config live overrides
        if self.telegram:
            self.telegram._config_getter = self._config_get
            self.telegram._config_setter = self._config_set

        self._running = False

    # ---------- telegram config bridge ----------
    def _config_get(self) -> Dict[str, Any]:
        out = {"strategy": self.strategy.get_params()}
        out["executor"] = {
            "allow_gtt_trailing": self.executor.allow_gtt_trailing,
            "gtt_trail_min_step": self.executor.gtt_trail_min_step,
            "gtt_trail_cooldown_s": self.executor.gtt_trail_cooldown_s,
        }
        return out

    def _config_set(self, key: str, val: str) -> str:
        # route to strategy first
        msg = self.strategy.set_param(key, val)
        if msg != "Key not permitted":
            return f"strategy.{msg}"
        # executor keys
        k = key.strip().lower()
        if k == "allow_gtt_trailing":
            self.executor.allow_gtt_trailing = val.lower() in ("1","true","yes","on")
            return f"executor.allow_gtt_trailing={self.executor.allow_gtt_trailing}"
        if k == "gtt_trail_min_step":
            try:
                self.executor.gtt_trail_min_step = float(val); return f"executor.gtt_trail_min_step={self.executor.gtt_trail_min_step}"
            except ValueError:
                return "Invalid float"
        if k == "gtt_trail_cooldown_s":
            try:
                self.executor.gtt_trail_cooldown_s = int(val); return f"executor.gtt_trail_cooldown_s={self.executor.gtt_trail_cooldown_s}"
            except ValueError:
                return "Invalid int"
        return "Key not permitted"

    # ---------- main loop ----------

    def start(self) -> None:
        self._running = True
        logger.info("StrategyRunner started.")
        self.ds.connect()

        while self._running:
            try:
                if settings.enable_live_trading and not is_market_open(settings.executor.market_open, settings.executor.market_close):
                    time.sleep(5); continue

                # TODO: fetch latest df window from your data source
                df = self._fetch_df_window()
                if df is None or df.empty:
                    time.sleep(0.5); continue

                signal = self.strategy.generate_signal(df, current_price=float(df["close"].iloc[-1]))
                if signal:
                    # risk/session gates
                    deny = self.sizer.check_risk_gates(self.session)
                    if deny:
                        logger.info("Denied by risk gate: %s", deny)
                    else:
                        self._execute_signal(signal)

                # manage open trades (trailing & greeks-aware adaptation)
                self._manage_open_positions(df)

                time.sleep(0.5)
            except Exception as e:
                logger.error("Main loop error: %s", e, exc_info=True)
                time.sleep(1.0)

    def stop(self) -> None:
        self._running = False
        logger.info("StrategyRunner stopped.")

    # ---------- internals ----------

    def _fetch_df_window(self) -> Optional[Any]:
        """
        Implement: fetch a minute dataframe slice for the underlier.
        This is a placeholder (depends on your data source design).
        """
        return None  # wire to your LiveKiteSource / loader

    def _execute_signal(self, signal: Dict[str, Any]) -> None:
        try:
            # strike selection example (ATM ± 0)
            instr = fetch_cached_instruments()
            tokens = get_instrument_tokens(instr, underlying="NIFTY", offset=0)
            option_symbol = tokens["CE"] if signal["signal"] == "BUY" else tokens["PE"]

            qty = self.sizer.calculate_lot_quantity(settings.executor.nifty_lot_size)
            order_id = self.executor.place_entry_order(
                symbol=option_symbol,
                exchange=settings.executor.trade_exchange,
                quantity=qty,
                transaction_type=signal["signal"],
                order_type="MARKETABLE_LIMIT",
            )
            self.executor.setup_gtt_orders(
                entry_order_id=order_id,
                entry_price=signal["entry_price"],
                stop_loss_price=signal["stop_loss"],
                target_price=signal["target"],
                symbol=option_symbol,
                exchange=settings.executor.trade_exchange,
                quantity=qty,
                transaction_type=signal["signal"],
            )
            if self.telegram:
                self.telegram.send_message(f"✅ Trade: {signal['signal']} {option_symbol} @ {signal['entry_price']:.2f} | SL {signal['stop_loss']:.2f} TP {signal['target']:.2f} | regime={signal.get('regime')}")
        except Exception as e:
            logger.error("Execute signal failed: %s", e, exc_info=True)

    def _manage_open_positions(self, df_underlier: Any) -> None:
        """
        - Trailing stop with playbook's trail multiplier.
        - Greeks-aware tightening/extension if option & underlier quotes available.
        """
        for rec in self.executor.get_active_orders():
            if not rec.is_open:
                continue

            # current underlier price from df
            try:
                cur_px = float(df_underlier["close"].iloc[-1])
            except Exception:
                continue

            # ATR from latest signal context if present; else approximate with rolling std proxy
            atr_proxy = float(max(0.1, df_underlier["close"].rolling(14).std().iloc[-1] * 0.8))

            # use trail multiplier from last signal (fallback 1.0)
            trail_mult = 1.0
            # if you persist trail_mult per order, plug it here; else use 1.0
            new_sl = self.executor.update_trailing_stop(rec.order_id, cur_px, atr_proxy, trail_mult=trail_mult)

            # Greeks-aware adjustment (best-effort)
            try:
                opt_ltp = self.executor.get_last_price(rec.symbol)
                und_ltp = cur_px
                # rough IV from last N bars
                iv_guess = float(max(0.01, estimate_iv(spot=und_ltp, option_price=opt_ltp, strike=und_ltp, days_to_expiry=5)))
                delta = float(estimate_delta(spot=und_ltp, strike=und_ltp, days_to_expiry=5, iv=iv_guess, call=(rec.transaction_type=="BUY")))
                # If delta improves in our favor, gently ratchet SL tighter; if worsens, leave SL or consider early exit logic
                if new_sl is not None and ((rec.transaction_type=="BUY" and delta > 0.35) or (rec.transaction_type=="SELL" and delta < -0.35)):
                    self.executor.update_trailing_stop(rec.order_id, cur_px, atr_proxy, trail_mult=trail_mult * 1.15)
            except Exception:
                pass
