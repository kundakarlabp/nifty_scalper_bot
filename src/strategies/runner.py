# src/strategies/runner.py
from __future__ import annotations

import logging
import time
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, List, Callable

import pandas as pd

from src.config import settings
from src.risk.position_sizing import PositionSizing
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.utils.account_info import get_equity_estimate
from src.utils.atr_helper import compute_atr
from src.data.source import DataSource, LiveKiteSource
from src.execution.order_executor import OrderExecutor

# Optional broker SDK
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import NetworkException, TokenException, InputException  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception

log = logging.getLogger(__name__)


def _now_ist_naive() -> datetime:
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return ist.replace(tzinfo=None)


def _ensure_adx_di(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    if df is None or df.empty or not {"high", "low", "close"}.issubset(df.columns):
        return df
    try:
        from ta.trend import ADXIndicator  # type: ignore
        adxi = ADXIndicator(df["high"], df["low"], df["close"], window=window)
        df[f"adx_{window}"] = adxi.adx()
        df[f"di_plus_{window}"] = adxi.adx_pos()
        df[f"di_minus_{window}"] = adxi.adx_neg()
        return df
    except Exception:
        pass

    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = (df["high"] - df["low"]).abs()
    atr = tr.ewm(alpha=1 / window, adjust=False).mean().replace(0, 1e-9)
    plus_di = (plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr) * 100.0
    minus_di = (minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr) * 100.0
    dx = (plus_di.subtract(minus_di).abs() / (plus_di.add(minus_di).abs() + 1e-9)) * 100.0
    adx = dx.ewm(alpha=1 / window, adjust=False).mean()
    df[f"adx_{window}"] = adx
    df[f"di_plus_{window}"] = plus_di
    df[f"di_minus_{window}"] = minus_di
    return df


def _fetch_and_prepare_df(
    data_source: Optional[DataSource],
    token: Optional[int],
    lookback: timedelta,
    timeframe: str,
) -> pd.DataFrame:
    if data_source is None or token is None:
        return pd.DataFrame()
    end_date = _now_ist_naive()
    start_date = end_date - lookback
    df = data_source.fetch_ohlc(token, start_date, end_date, timeframe)
    if df.empty:
        return pd.DataFrame()
    req = {"open", "high", "low", "close"}
    return df if req.issubset(df.columns) else pd.DataFrame()


class StrategyRunner:
    """
    End-to-end orchestrator:
      - Gets spot & option OHLC
      - Strategy → signal
      - Sizing → lots→units
      - Executor → entry/exits/trailing/OCO
      - Emits events for Telegram
    """

    def __init__(
        self,
        strategy: Optional[EnhancedScalpingStrategy] = None,
        data_source: Optional[DataSource] = None,
        spot_source: Optional[DataSource] = None,
        kite: Optional["KiteConnect"] = None,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.strategy = strategy or EnhancedScalpingStrategy()
        self._kite = kite or self._build_kite()
        self.data_source = data_source or self._build_live_source(self._kite)
        self.spot_source = spot_source or self.data_source
        self._live = bool(getattr(settings, "enable_live_trading", False))
        self._paused = False
        self._event_sink = event_sink

        # executor
        self.executor = OrderExecutor(
            config=getattr(settings, "executor", None),
            kite=self._kite,
            data_source=self.data_source,
        )

        # caches
        self._symbol_map: Dict[int, str] = {}

    def _emit(self, evt_type: str, **payload: Any) -> None:
        if self._event_sink:
            try:
                self._event_sink({"type": evt_type, **payload})
            except Exception:
                pass

    def _build_kite(self) -> Optional["KiteConnect"]:
        if KiteConnect is None:
            log.warning("KiteConnect not installed; StrategyRunner in shadow mode.")
            return None
        api_key = getattr(getattr(settings, "zerodha", object()), "api_key", None)
        access_token = getattr(getattr(settings, "zerodha", object()), "access_token", None)
        if not api_key:
            return None
        kc = KiteConnect(api_key=api_key)
        if access_token:
            try:
                kc.set_access_token(access_token)
            except Exception:
                pass
        return kc

    def _build_live_source(self, kite: Optional["KiteConnect"]) -> Optional[DataSource]:
        if kite is None:
            return None
        try:
            ds = LiveKiteSource(kite)
            ds.connect()
            return ds
        except Exception:
            return None

    def _token_to_symbol(self, token: int) -> Optional[str]:
        if token in self._symbol_map:
            return self._symbol_map[token]
        if not self._kite:
            return None
        try:
            for seg in ("NFO", "NSE"):
                for row in self._kite.instruments(seg):
                    if int(row.get("instrument_token", -1)) == int(token):
                        sym = str(row.get("tradingsymbol"))
                        self._symbol_map[token] = sym
                        return sym
        except Exception:
            pass
        return None

    def to_status_dict(self) -> Dict[str, Any]:
        active = self.executor.get_active_orders()
        return {
            "time_ist": _now_ist_naive().isoformat(sep=" ", timespec="seconds"),
            "broker": "Kite" if self._kite else "none",
            "data_source": type(self.data_source).__name__ if self.data_source else None,
            "live_trading": self._live,
            "paused": self._paused,
            "active_orders": len(active),
        }

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def run_once(self, stop_event: threading.Event) -> Optional[Dict[str, Any]]:
        # market hours gate (keep servicing open trades)
        now_open = self._ist_market_open()
        if not now_open and not self._paused:
            # still service open trades so trailing/OCO keep working
            self._service_open_trades(None)
            return None

        if stop_event.is_set():
            return None

        try:
            # Resolve instruments:
            inst = getattr(settings, "instruments", object())
            spot_token = int(getattr(inst, "instrument_token", 256265))  # NIFTY spot default
            timeframe = str(getattr(getattr(settings, "data", object()), "timeframe", "minute"))
            lookback_minutes = int(getattr(getattr(settings, "data", object()), "lookback_minutes", 60))
            lookback = timedelta(minutes=lookback_minutes)

            # Get spot & option data:
            spot_df = _fetch_and_prepare_df(self.spot_source, spot_token, lookback, timeframe)
            if spot_df.empty:
                self._service_open_trades(None)
                return None

            # Spot LTP for strike selection
            spot_symbol = str(getattr(inst, "spot_symbol", "NSE:NIFTY 50"))
            spot_ltp = None
            try:
                if hasattr(self.spot_source, "get_last_price"):
                    spot_ltp = self.spot_source.get_last_price(spot_symbol)
            except Exception:
                spot_ltp = None
            if not spot_ltp or spot_ltp <= 0:
                self._service_open_trades(spot_df)
                return None

            # Select CE/PE token by spot (simple ATM+offset logic lives elsewhere in your project; here we keep CE path)
            from src.utils.strike_selector import get_instrument_tokens  # local import to avoid cycles
            token_info = get_instrument_tokens(kite=self._kite, spot_price=spot_ltp)
            if not token_info:
                self._service_open_trades(spot_df)
                return None
            ce_token = token_info.get("tokens", {}).get("ce")
            if not ce_token:
                self._service_open_trades(spot_df)
                return None

            opt_df = _fetch_and_prepare_df(self.data_source, ce_token, lookback, timeframe)
            if opt_df.empty:
                self._service_open_trades(spot_df)
                return None

            # Indicators (ADX/DI on spot)
            adx_window = int(getattr(getattr(settings, "strategy", object()), "atr_period", 14))
            spot_df = _ensure_adx_di(spot_df, window=adx_window)
            current_price = float(opt_df["close"].iloc[-1])

            # Pause gate: allow trailing/OCO but skip new entries
            if self._paused:
                self._service_open_trades(opt_df)
                return None

            # Strategy
            signal = self.strategy.generate_signal(opt_df, spot_df, current_price)
            if not signal:
                self._service_open_trades(opt_df)
                return None

            # Sizing
            equity = float(get_equity_estimate(self._kite))
            sl_points = float(signal.get("sl_points", 0.0) or 0.0)
            lots = int(PositionSizing.lots_from_equity(equity=equity, sl_points=sl_points))
            if equity <= 0 or sl_points <= 0 or lots <= 0:
                self._service_open_trades(opt_df)
                return None
            lot_size = int(getattr(inst, "nifty_lot_size", 75))
            quantity_units = lots * lot_size

            # Place orders if live
            option_symbol = self._token_to_symbol(ce_token) if self._kite else None
            enriched = {
                **signal,
                "equity": equity,
                "lots": lots,
                "quantity_units": quantity_units,
                "instrument": {
                    "symbol_ce": option_symbol,
                    "token_ce": ce_token,
                    "atm_strike": token_info.get("atm_strike"),
                    "target_strike": token_info.get("target_strike"),
                    "expiry": token_info.get("expiry"),
                },
            }

            if self._live and self._kite and option_symbol:
                side = str(signal.get("side", "BUY")).upper()
                entry_price = float(signal.get("entry_price", current_price))
                sl_price = float(signal.get("stop_loss", 0.0) or 0.0)
                tp_price = float(signal.get("target", 0.0) or 0.0)

                rec_id = self.executor.place_entry_order(
                    token=ce_token,
                    symbol=option_symbol,
                    side=side,
                    quantity=quantity_units,
                    price=entry_price,
                )
                if rec_id:
                    enriched["order_record_id"] = rec_id
                    # Arm exits
                    if sl_price > 0 and tp_price > 0:
                        self.executor.setup_gtt_orders(rec_id, sl_price=sl_price, tp_price=tp_price)
                    # Emit event for Telegram
                    self._emit("ENTRY_PLACED", symbol=option_symbol, side=side, qty=quantity_units, price=entry_price, record_id=rec_id)

            # Service open trades (trail & OCO)
            self._service_open_trades(opt_df)

            return enriched

        except (NetworkException, TokenException, InputException) as e:
            log.error("Transient broker error: %s", e)
        except Exception as e:
            log.exception("Unexpected error in run_once: %s", e)

        return None

    def _service_open_trades(self, opt_df: Optional[pd.DataFrame]) -> None:
        try:
            active = self.executor.get_active_orders()
            if not active:
                return
            atr_val = None
            if opt_df is not None and not opt_df.empty:
                atr_period = int(getattr(getattr(settings, "strategy", object()), "atr_period", 14))
                atr_series = compute_atr(opt_df, period=atr_period)
                if atr_series is not None and len(atr_series):
                    atr_val = float(atr_series.iloc[-1])

            for rec in active:
                if not rec.is_open or atr_val is None or atr_val <= 0:
                    continue
                cur = float(opt_df["close"].iloc[-1]) if opt_df is not None and not opt_df.empty else None
                if cur is None:
                    continue
                try:
                    self.executor.update_trailing_stop(rec.record_id, current_price=cur, atr=atr_val, atr_multiplier=None)
                except Exception:
                    pass

            fills = self.executor.sync_and_enforce_oco()
            if fills:
                self._emit("FILLS", fills=fills)

        except Exception:
            pass

    @staticmethod
    def _ist_market_open() -> bool:
        now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
        if now.weekday() > 4:
            return False
        start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return start <= now < end
