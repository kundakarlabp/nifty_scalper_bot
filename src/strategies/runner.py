from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional

import pandas as pd

from src.config import settings
from src.risk.position_sizing import PositionSizing
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.utils.atr_helper import compute_atr

# tokens / market open
try:
    from src.utils.strike_selector import get_instrument_tokens, is_market_open  # type: ignore
except Exception:  # pragma: no cover
    def is_market_open() -> bool:  # fallback gate
        return True
    def get_instrument_tokens(*args, **kwargs) -> Optional[Dict[str, Any]]:
        return None

# Broker SDK / executor
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import NetworkException, TokenException, InputException  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception

try:
    from src.execution.order_executor import OrderExecutor  # type: ignore
except Exception:  # pragma: no cover
    OrderExecutor = None  # type: ignore

# Data source
try:
    from src.data.source import DataSource, LiveKiteSource  # type: ignore
except Exception:  # pragma: no cover
    DataSource = object  # type: ignore
    LiveKiteSource = None  # type: ignore

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
        # fallback manual
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
    Orchestrates:
      1) market-hours gating
      2) strike tokens from spot LTP
      3) OHLC fetch (spot + option)
      4) indicators on spot (ADX/DI)
      5) strategy signal (df, spot_df, current_price)
      6) position sizing
      7) (optional) live execution
      8) emits events for Telegram (ENTRY_PLACED, FILLS)
      9) /diag snapshot & /tick
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
        self._tick_lock = threading.Lock()

        # optional executor
        self.executor = None
        if OrderExecutor is not None and getattr(settings, "executor", None) is not None:
            try:
                self.executor = OrderExecutor(
                    config=settings.executor,
                    kite=self._kite,
                    data_source=self.data_source,
                )
            except Exception as e:
                log.warning("OrderExecutor not initialized: %s", e)

        self._symbol_cache: dict[int, str] = {}

    # ---- infra / wiring ----
    def set_event_sink(self, sink: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        self._event_sink = sink

    def _emit(self, evt_type: str, **payload: Any) -> None:
        if self._event_sink:
            try:
                self._event_sink({"type": evt_type, **payload})
            except Exception:
                pass

    def _build_kite(self) -> Optional["KiteConnect"]:
        if KiteConnect is None:
            log.info("KiteConnect not installed; shadow mode.")
            return None
        zk = getattr(settings, "zerodha", object())
        api_key = getattr(zk, "api_key", None)
        access_token = getattr(zk, "access_token", None)
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
        if kite is None or LiveKiteSource is None:
            return None
        try:
            ds = LiveKiteSource(kite)
            ds.connect()
            return ds
        except Exception:
            return None

    def _token_to_symbol(self, token: int) -> Optional[str]:
        if token in self._symbol_cache or not self._kite:
            return self._symbol_cache.get(token)
        try:
            for seg in ("NFO", "NSE"):
                for row in self._kite.instruments(seg):
                    if int(row.get("instrument_token", -1)) == int(token):
                        sym = str(row.get("tradingsymbol"))
                        self._symbol_cache[token] = sym
                        return sym
        except Exception:
            pass
        return None

    def _equity_estimate(self) -> float:
        # robust fallback (no import dependency)
        try:
            if self._kite:
                # Optional: margins snapshot (if allowed)
                # Some setups restrict this; keep it guarded.
                pass
        except Exception:
            pass
        return float(getattr(getattr(settings, "risk", object()), "default_equity", 30000.0))

    # ---- public ----
    def to_status_dict(self) -> Dict[str, Any]:
        active = self.executor.get_active_orders() if self.executor else []
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

    def diag_snapshot(self) -> Dict[str, Any]:
        """Used by /diag to show pipeline health."""
        inst = getattr(settings, "instruments", object())
        timeframe = str(getattr(getattr(settings, "data", object()), "timeframe", "minute"))
        lookback_minutes = int(getattr(getattr(settings, "data", object()), "lookback_minutes", 60))
        lookback = timedelta(minutes=lookback_minutes)

        checks: list[dict] = []
        tokens: Optional[Dict[str, Any]] = None

        # 1) market gate
        mo = is_market_open()
        checks.append({"name": "market_open", "ok": bool(mo)})

        # 2) spot LTP
        spot_symbol = str(getattr(inst, "spot_symbol", "NSE:NIFTY 50"))
        spot_ltp = None
        try:
            if hasattr(self.spot_source, "get_last_price"):
                spot_ltp = self.spot_source.get_last_price(spot_symbol)  # type: ignore[attr-defined]
        except Exception:
            spot_ltp = None
        checks.append({"name": "spot_ltp", "ok": spot_ltp is not None, "value": spot_ltp})

        # 3) spot OHLC
        spot_token = int(getattr(inst, "instrument_token", 256265))
        spot_df = _fetch_and_prepare_df(self.spot_source, spot_token, lookback, timeframe)
        checks.append({"name": "spot_ohlc", "ok": not spot_df.empty, "rows": int(len(spot_df))})

        # 4) strikes
        tokens = None
        try:
            # be tolerant of both function signatures
            try:
                tokens = get_instrument_tokens(kite_instance=self._kite)  # new signature
            except TypeError:
                tokens = get_instrument_tokens(kite=self._kite, spot_price=spot_ltp)  # legacy
        except Exception:
            tokens = None
        checks.append({"name": "strike_selection", "ok": bool(tokens), "result": tokens})

        # 5) option OHLC
        opt_rows = 0
        ok_opt = False
        if tokens and tokens.get("tokens", {}).get("ce"):
            ce_token = int(tokens["tokens"]["ce"])
            opt_df = _fetch_and_prepare_df(self.data_source, ce_token, lookback, timeframe)
            opt_rows = int(len(opt_df))
            ok_opt = not opt_df.empty
        checks.append({"name": "option_ohlc", "ok": ok_opt, "rows": opt_rows})

        # 6–8) indicators/signal/sizing quick check
        if not spot_df.empty:
            spot_df = _ensure_adx_di(spot_df, window=int(getattr(getattr(settings, "strategy", object()), "adx_period", 14)))
            checks.append({"name": "indicators", "ok": True})
        else:
            checks.append({"name": "indicators", "ok": False, "error": "spot OHLC empty"})

        sig_ok = False
        size_ok = False
        if ok_opt:
            current_price = float(opt_df["close"].iloc[-1])  # type: ignore[name-defined]
            sig = self.strategy.generate_signal(opt_df, spot_df, current_price)  # type: ignore[name-defined]
            sig_ok = bool(sig)
            if sig:
                sl_points = float(sig.get("sl_points", 0.0) or 0.0)
                equity = self._equity_estimate()
                lots = int(PositionSizing.lots_from_equity(equity=equity, sl_points=sl_points))
                size_ok = lots > 0
            checks.append({"name": "signal", "ok": sig_ok})
            checks.append({"name": "sizing", "ok": size_ok})
        else:
            checks.append({"name": "signal", "ok": False, "error": "option OHLC empty"})
            checks.append({"name": "sizing", "ok": False, "error": "no signal"})

        checks.append({"name": "execution_ready", "ok": bool(self._live and self._kite and self.executor),
                       "live": self._live, "broker": bool(self._kite), "executor": bool(self.executor)})
        n_open = len(self.executor.get_active_orders()) if self.executor else 0
        checks.append({"name": "open_orders", "ok": True, "count": n_open})

        ok_all = all(c.get("ok") for c in checks if c["name"] not in ("market_open",))
        return {"ok": ok_all, "checks": checks, "tokens": tokens}

    def tick_once(self) -> None:
        """Force a single tick (used by /tick)."""
        with self._tick_lock:
            dummy_event = threading.Event()
            self.run_once(stop_event=dummy_event)

    # ---- main tick ----
    def run_once(self, stop_event: threading.Event) -> Optional[Dict[str, Any]]:
        service_only = not is_market_open()
        if stop_event.is_set():
            return None

        try:
            inst = getattr(settings, "instruments", object())
            timeframe = str(getattr(getattr(settings, "data", object()), "timeframe", "minute"))
            lookback_minutes = int(getattr(getattr(settings, "data", object()), "lookback_minutes", 60))
            lookback = timedelta(minutes=lookback_minutes)

            # spot history
            spot_token = int(getattr(inst, "instrument_token", 256265))
            spot_df = _fetch_and_prepare_df(self.spot_source, spot_token, lookback, timeframe)

            # spot LTP (for info only)
            try:
                spot_symbol = str(getattr(inst, "spot_symbol", "NSE:NIFTY 50"))
                spot_ltp = self.spot_source.get_last_price(spot_symbol) if hasattr(self.spot_source, "get_last_price") else None  # type: ignore[attr-defined]
            except Exception:
                spot_ltp = None

            # service open trades always
            self._service_open_trades(spot_df if not spot_df.empty else None)

            if service_only or self._paused:
                return None

            if not spot_df.empty:
                adx_window = int(getattr(getattr(settings, "strategy", object()), "adx_period", 14))
                spot_df = _ensure_adx_di(spot_df, window=adx_window)

            # tolerant strike selection (new vs legacy signature)
            try:
                token_info = get_instrument_tokens(kite_instance=self._kite)  # new
            except TypeError:
                token_info = get_instrument_tokens(kite=self._kite, spot_price=spot_ltp)  # legacy
            if not token_info:
                return None
            ce_token = token_info.get("tokens", {}).get("ce")
            if not ce_token:
                return None

            # option OHLC
            opt_df = _fetch_and_prepare_df(self.data_source, ce_token, lookback, timeframe)
            if opt_df.empty:
                return None
            current_price = float(opt_df["close"].iloc[-1])

            # strategy (NOTE: order = df, spot_df, current_price)
            signal = self.strategy.generate_signal(opt_df, spot_df, current_price)
            if not signal:
                return None

            # sizing
            equity = self._equity_estimate()
            sl_points = float(signal.get("sl_points", 0.0) or 0.0)
            if equity <= 0 or sl_points <= 0:
                return None
            lot_size = int(getattr(inst, "nifty_lot_size", 75))
            lots = int(PositionSizing.lots_from_equity(equity=equity, sl_points=sl_points, lot_size=lot_size))
            if lots <= 0:
                return None
            quantity_units = lots * lot_size

            option_symbol = self._token_to_symbol(int(ce_token)) if self._kite else None

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

            # live execution
            if self._live and self._kite and self.executor and option_symbol:
                side = str(signal.get("side", "BUY")).upper()
                entry_price = float(signal.get("entry_price", current_price))
                sl_price = float(signal.get("stop_loss", 0.0) or 0.0)
                tp_price = float(signal.get("target", 0.0) or 0.0)

                rec_id = self.executor.place_entry_order(
                    token=int(ce_token),
                    symbol=option_symbol,
                    side=side,
                    quantity=quantity_units,
                    price=entry_price,
                )
                if rec_id:
                    enriched["order_record_id"] = rec_id
                    if sl_price > 0 and tp_price > 0:
                        try:
                            self.executor.setup_gtt_orders(rec_id, sl_price=sl_price, tp_price=tp_price)
                        except Exception:
                            pass
                    self._emit("ENTRY_PLACED", symbol=option_symbol, side=side, qty=quantity_units, price=entry_price, record_id=rec_id)
                    # service immediately after placement
                    self._service_open_trades(opt_df)

            return enriched

        except (NetworkException, TokenException, InputException) as e:
            log.error("Transient broker error: %s", e)
        except Exception as e:
            log.exception("Unexpected error in run_once: %s", e)

        return None

    def _service_open_trades(self, opt_or_spot_df: Optional[pd.DataFrame]) -> None:
        if not self.executor:
            return
        try:
            active = self.executor.get_active_orders()
            if not active:
                return

            atr_val = None
            if opt_or_spot_df is not None and not opt_or_spot_df.empty:
                atr_period = int(getattr(getattr(settings, "strategy", object()), "atr_period", 14))
                atr_series = compute_atr(opt_or_spot_df, period=atr_period)
                if atr_series is not None and len(atr_series):
                    atr_val = float(atr_series.iloc[-1])

            if atr_val and atr_val > 0:
                try:
                    cur = float(opt_or_spot_df["close"].iloc[-1])  # last close proxy
                except Exception:
                    cur = None
                if cur is not None:
                    for rec in active:
                        if getattr(rec, "is_open", False):
                            try:
                                self.executor.update_trailing_stop(
                                    rec.order_id, current_price=cur, atr=atr_val, atr_multiplier=None
                                )
                            except Exception:
                                pass

            fills = self.executor.sync_and_enforce_oco()
            if fills:
                self._emit("FILLS", fills=fills)
        except Exception:
            pass