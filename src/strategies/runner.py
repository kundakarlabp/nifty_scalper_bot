# src/strategies/runner.py
"""
StrategyRunner: orchestrates the live loop

Responsibilities
- Respect trading hours (timezone-aware)
- Pull OHLC window for the spot token
- Ask strategy for signal (direction, score, ATR/levels)
- Resolve CE/PE instruments around ATM using strike_selector
- Size trade, place entry, set up TP/SL (GTT or fallback)
- Maintain trailing & partials (TP1/TP2), and basic risk halts
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from src.config import settings
from src.utils.strike_selector import (
    fetch_cached_instruments,
    get_instrument_tokens,
    is_trading_hours,
)
from src.execution.order_executor import OrderExecutor
from src.data.source import LiveKiteSource
from src.risk.position_sizing import PositionSizer
from src.risk.session import TradingSession
from src.notifications.telegram_controller import TelegramController

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    side: Optional[str]          # "UP" or "DOWN" (spot direction) or None
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

        # loop control
        self._running: bool = False
        self._sleep_sec: int = int(os.getenv("PEAK_POLL_SEC", "5"))  # default 5s

        # cache for instruments
        self._cache: Dict[str, Any] = {"NFO": [], "NSE": []}
        self._cache_ts: float = 0.0
        self._cache_ttl_sec: int = 300  # refresh every 5 minutes

        # env-tunable basic TP/SL for options (percent of option entry)
        self._opt_sl_pct: float = float(os.getenv("OPTION_SL_PERCENT", "0.06"))   # 6%
        self._opt_tp_pct: float = float(os.getenv("OPTION_TP_PERCENT", "0.20"))   # 20%

    # ---------------- public ----------------

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

        logger.info("Runner started. Live: %s TZ=%s", settings.enable_live_trading, os.getenv("TZ", "unset"))

        while self._running:
            try:
                if not self._is_ok_to_trade_now():
                    time.sleep(self._sleep_sec)
                    continue

                self._maybe_refresh_instruments(kite)

                # --- Resolve instruments (ATM ± offset) ---
                tokens = get_instrument_tokens(
                    symbol=settings.executor.trade_symbol,
                    kite_instance=kite,
                    cached_nfo_instruments=self._cache["NFO"],
                    cached_nse_instruments=self._cache["NSE"],
                    offset=0,
                    strike_range=settings.strategy.strike_selection_range,
                )
                if not tokens or not tokens.get("spot_token"):
                    time.sleep(self._sleep_sec)
                    continue

                spot_token = int(tokens["spot_token"])
                # --- Fetch spot OHLC window ---
                end = datetime.now()
                start = end - timedelta(minutes=max(5, settings.executor.data_lookback_minutes))
                df = self.ds.get_historical_ohlc(spot_token, start, end, interval="minute")
                if df.empty or len(df) < max(20, settings.strategy.min_bars_for_signal):
                    time.sleep(self._sleep_sec)
                    continue

                # --- Strategy signal ---
                sig = self._get_signal(df)
                if not sig or sig.side is None:
                    self._housekeeping()
                    time.sleep(self._sleep_sec)
                    continue

                # --- Check min score/confidence ---
                if int(sig.score) < int(settings.strategy.min_signal_score) or sig.confidence < float(settings.strategy.confidence_threshold):
                    self._housekeeping()
                    time.sleep(self._sleep_sec)
                    continue

                # --- Choose option leg by spot direction ---
                if sig.side == "UP":
                    opt_sym = tokens.get("ce_symbol")
                    leg = "CE"
                else:
                    opt_sym = tokens.get("pe_symbol")
                    leg = "PE"
                if not opt_sym:
                    logger.debug("No option symbol for leg=%s; skipping", leg)
                    self._housekeeping()
                    time.sleep(self._sleep_sec)
                    continue

                # --- Sizing ---
                lot_size = int(settings.executor.nifty_lot_size)
                qty = self.sizer.size_for_account(self.session, lot_size)  # strategy-independent sizing
                if qty <= 0:
                    self._housekeeping()
                    time.sleep(self._sleep_sec)
                    continue

                # --- Entry ---
                opt_ltp = self.exec.get_last_price(opt_sym, exchange=settings.executor.trade_exchange)
                if opt_ltp <= 0.0:
                    self._housekeeping()
                    time.sleep(self._sleep_sec)
                    continue

                entry_side = "BUY"  # options buying strategy
                order_id = self.exec.place_entry_order(
                    symbol=opt_sym,
                    exchange=settings.executor.trade_exchange,
                    quantity=qty,
                    transaction_type=entry_side,
                    order_type="MARKET",  # safer for fills; change to MARKETABLE_LIMIT if preferred
                )

                # --- Initial SL/TP on option price (percentage-based; ATR could refine later) ---
                stop_loss_price, target_price = self._compute_option_levels(entry_price=opt_ltp, side=entry_side)

                self.exec.setup_gtt_orders(
                    entry_order_id=order_id,
                    entry_price=opt_ltp,
                    stop_loss_price=stop_loss_price,
                    target_price=target_price,
                    symbol=opt_sym,
                    exchange=settings.executor.trade_exchange,
                    quantity=qty,
                    transaction_type=entry_side,
                )

                # Optional notify
                if self.tg:
                    self.tg.send_message(
                        f"✅ Entry {entry_side} <b>{opt_sym}</b> x{qty} @ <code>{opt_ltp:.2f}</code>\n"
                        f"SL <code>{stop_loss_price:.2f}</code> | TP <code>{target_price:.2f}</code> "
                        f"(leg={leg}, score={sig.score}, conf={sig.confidence})"
                    )

                # Post-entry trailing baseline
                if sig.atr > 0:
                    self.exec.update_trailing_stop(order_id, current_price=opt_ltp, atr=max(0.5, sig.atr * 0.2))

                # Housekeeping after possible entry
                self._housekeeping()

            except Exception as e:
                logger.error("Runner loop error: %s", e, exc_info=True)
                time.sleep(self._sleep_sec)

        logger.info("Runner stopped.")

    def stop(self) -> None:
        self._running = False

    # ---------------- internals ----------------

    def _is_ok_to_trade_now(self) -> bool:
        if settings.allow_offhours_testing:
            return True
        return is_trading_hours(settings.executor.market_open, settings.executor.market_close, tz_name=os.getenv("TZ"))

    def _maybe_refresh_instruments(self, kite) -> None:
        now = time.time()
        if now - self._cache_ts >= self._cache_ttl_sec or not self._cache["NFO"]:
            self._cache = fetch_cached_instruments(kite)
            self._cache_ts = now

    def _get_signal(self, df) -> Optional[Signal]:
        """
        Adapter for your strategy; expected to return direction, score, confidence, atr.
        Implementations may vary; we adapt generously.
        """
        try:
            # Common patterns:
            # - strategy.evaluate(df) -> dict with keys
            # - strategy.generate_signal(df) / compute_signal(df)
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
                side = side.upper()
                if side in ("BUY", "LONG", "UP"):
                    side = "UP"
                elif side in ("SELL", "SHORT", "DOWN"):
                    side = "DOWN"
                else:
                    side = None

            score = float(res.get("score", 0.0))
            confidence = float(res.get("confidence", 0.0))
            atr = float(res.get("atr", 0.0))
            reason = str(res.get("reason", ""))

            return Signal(side=side, score=score, atr=atr, confidence=confidence, reason=reason)
        except Exception as e:
            logger.debug("signal adapter failed: %s", e)
            return None

    def _compute_option_levels(self, entry_price: float, side: str) -> tuple[float, float]:
        """
        Set SL/TP on the option price using percentages.
        If you later expose OPTION_SL_PERCENT/OPTION_TP_PERCENT via Settings, wire them here.
        """
        sl = max(1.0, entry_price * (1.0 - self._opt_sl_pct))
        tp = max(sl + self.exec.get_tick_size(), entry_price * (1.0 + self._opt_tp_pct))
        return (round(sl, 2), round(tp, 2))

    def _housekeeping(self) -> None:
        """Trailing & partials management for all open orders; extend with risk halts if needed."""
        # Opportunistic trailing: use a small ATR stand-in derived from price
        for rec in self.exec.get_active_orders():
            ltp = self.exec.get_last_price(rec.symbol, exchange=rec.exchange)
            if ltp > 0:
                # crude ATR proxy if the strategy didn’t provide one — use ~2% of price
                atr_proxy = max(0.5, ltp * 0.02)
                self.exec.update_trailing_stop(rec.order_id, current_price=ltp, atr=atr_proxy, trail_mult=0.6)

        # manage TP1/TP2 partials
        self.exec.manage_partials()
