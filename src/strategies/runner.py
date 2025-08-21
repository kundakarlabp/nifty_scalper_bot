from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional, List

import pandas as pd

from src.config import settings
from src.risk.position_sizing import PositionSizing
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.utils.account_info import get_equity_estimate
from src.utils.atr_helper import compute_atr

# strike + hours
from src.utils.strike_selector import get_instrument_tokens, is_market_open

# broker + executor
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

# data source
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
    # soft dependency; safe manual fallback
    try:
        from ta.trend import ADXIndicator  # type: ignore
        adxi = ADXIndicator(df["high"], df["low"], df["close"], window=window)
        df[f"adx_{window}"] = adxi.adx()
        df[f"di_plus_{window}"] = adxi.adx_pos()
        df[f"di_minus_{window}"] = adxi.adx_neg()
        return df
    except Exception:
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


def _fetch_df(
    data_source: Optional[DataSource],
    token: Optional[int],
    lookback: timedelta,
    timeframe: str,
) -> pd.DataFrame:
    if data_source is None or token is None:
        return pd.DataFrame()
    end = _now_ist_naive()
    start = end - lookback
    df = data_source.fetch_ohlc(token, start, end, timeframe)
    req = {"open", "high", "low", "close"}
    if df is None or df.empty or not req.issubset(df.columns):
        return pd.DataFrame()
    return df


class StrategyRunner:
    """
    End-to-end orchestrator:
      - hours gate (but always services exits)
      - resolves tokens
      - fetches OHLC (spot + option)
      - computes indicators (spot)
      - strategy -> signal
      - position sizing
      - live execution via OrderExecutor
      - emits events for Telegram (ENTRY_PLACED, FILLS)
      - diagnostics /diag
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

        self._live = bool(settings.enable_live_trading)
        self._paused = False
        self._event_sink = event_sink

        self.executor = None
        if OrderExecutor is not None and getattr(settings, "executor", None) is not None and self._kite:
            try:
                self.executor = OrderExecutor(
                    config=settings.executor,
                    kite=self._kite,
                    data_source=self.data_source,
                )
            except Exception as e:
                log.warning("OrderExecutor not initialized: %s", e)

        self._symbol_cache: dict[int, str] = {}

    # -------- infra --------
    def _emit(self, evt_type: str, **payload: Any) -> None:
        if self._event_sink:
            try: self._event_sink({"type": evt_type, **payload})
            except Exception: pass

    def _build_kite(self) -> Optional["KiteConnect"]:
        if KiteConnect is None:
            return None
        zk = settings.zerodha
        if not zk.api_key:
            return None
        kc = KiteConnect(api_key=zk.api_key)
        if zk.access_token:
            try: kc.set_access_token(zk.access_token)
            except Exception: pass
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

    # -------- external controls (wired to Telegram) --------
    def set_live_mode(self, val: bool) -> None:
        self._live = bool(val)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    # -------- status / heartbeat --------
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

    # -------- diagnostics for /diag --------
    def diagnose(self) -> Dict[str, Any]:
        checks: List[Dict[str, Any]] = []

        # market hours
        mo = is_market_open()
        checks.append({"name": "market_open", "ok": bool(mo)})

        inst = settings.instruments
        timeframe = settings.data.timeframe
        lookback = timedelta(minutes=int(settings.data.lookback_minutes))

        # spot ltp
        spot_price = None
        try:
            if self.spot_source and hasattr(self.spot_source, "get_last_price"):
                spot_price = self.spot_source.get_last_price(inst.spot_symbol)  # type: ignore[attr-defined]
        except Exception:
            spot_price = None
        checks.append({"name": "spot_ltp", "ok": spot_price is not None and spot_price > 0, "value": spot_price})

        # spot ohlc
        spot_df = _fetch_df(self.spot_source, inst.instrument_token, lookback, timeframe)
        checks.append({"name": "spot_ohlc", "ok": not spot_df.empty, "rows": int(len(spot_df))})

        # token selection
        token_info = get_instrument_tokens(kite_instance=self._kite)
        checks.append({"name": "strike_selection", "ok": bool(token_info), "result": token_info})

        # option ohlc
        opt_rows = 0
        if token_info and token_info.get("tokens", {}).get("ce"):
            opt_df = _fetch_df(self.data_source, token_info["tokens"]["ce"], lookback, timeframe)
            opt_rows = len(opt_df)
            checks.append({"name": "option_ohlc", "ok": not opt_df.empty, "rows": int(opt_rows)})
        else:
            checks.append({"name": "option_ohlc", "ok": False, "rows": 0})

        # indicators precheck
        if not spot_df.empty:
            adx_win = int(settings.strategy.adx_period)
            sd2 = _ensure_adx_di(spot_df.copy(), adx_win)
            ok_ind = sd2.columns.str.startswith(("adx_", "di_plus_", "di_minus_")).any()
            checks.append({"name": "indicators", "ok": bool(ok_ind)})
        else:
            checks.append({"name": "indicators", "ok": False, "error": "spot OHLC empty"})

        # signal precheck
        if opt_rows == 0:
            checks.append({"name": "signal", "ok": False, "error": "option OHLC empty"})
            checks.append({"name": "sizing", "ok": False, "error": "no signal"})
        else:
            checks.append({"name": "signal", "ok": True})
            checks.append({"name": "sizing", "ok": True})

        # execution ready
        checks.append({
            "name": "execution_ready",
            "ok": bool(self._live and self._kite and self.executor),
            "live": bool(self._live),
            "broker": bool(self._kite),
            "executor": bool(self.executor),
        })

        # open orders
        a = self.executor.get_active_orders() if self.executor else []
        checks.append({"name": "open_orders", "ok": True, "count": len(a)})

        return {"ok": all(c.get("ok") for c in checks), "checks": checks, "tokens": token_info}

    # -------- main cycle --------
    def run_once(self, stop_event: threading.Event) -> Optional[Dict[str, Any]]:
        # always service exits
        self._service_open_trades()
        if stop_event.is_set():
            return None

        # gate new entries by hours or pause
        if not is_market_open() and not settings.allow_offhours_testing:
            return None
        if self._paused:
            return None

        try:
            inst = settings.instruments
            timeframe = settings.data.timeframe
            lookback = timedelta(minutes=int(settings.data.lookback_minutes))

            # spot + opt OHLC
            spot_df = _fetch_df(self.spot_source, inst.instrument_token, lookback, timeframe)
            if not spot_df.empty:
                spot_df = _ensure_adx_di(spot_df, window=int(settings.strategy.adx_period))

            token_info = get_instrument_tokens(kite_instance=self._kite)
            if not token_info or not token_info.get("tokens", {}).get("ce"):
                return None
            ce_token = int(token_info["tokens"]["ce"])
            opt_df = _fetch_df(self.data_source, ce_token, lookback, timeframe)
            if opt_df.empty:
                return None

            current_price = float(opt_df["close"].iloc[-1])

            # strategy (opt_df, spot_df, current_price)
            signal = self.strategy.generate_signal(opt_df, spot_df, current_price)
            if not signal:
                return None

            # sizing
            try:
                equity = float(get_equity_estimate(self._kite))
            except TypeError:
                equity = float(get_equity_estimate())

            sl_points = float(signal.get("sl_points", 0.0) or 0.0)
            if equity <= 0 or sl_points <= 0:
                return None

            lots = int(PositionSizing.lots_from_equity(equity=equity, sl_points=sl_points))
            if lots <= 0:
                return None

            lot_size = int(inst.nifty_lot_size)
            qty_units = lots * lot_size

            # symbol for execution
            symbol = self._token_to_symbol(ce_token) if self._kite else None

            enriched = {
                **signal,
                "equity": equity,
                "lots": lots,
                "quantity_units": qty_units,
                "instrument": {
                    "symbol_ce": symbol,
                    "token_ce": ce_token,
                    "atm_strike": token_info.get("atm_strike"),
                    "target_strike": token_info.get("target_strike"),
                    "expiry": token_info.get("expiry"),
                },
            }

            # live execution
            if self._live and self._kite and self.executor and symbol:
                side = str(signal["side"]).upper()
                entry_price = float(signal.get("entry_price", current_price))
                sl_price = float(signal.get("stop_loss", 0.0) or 0.0)
                tp_price = float(signal.get("target", 0.0) or 0.0)

                rec_id = self.executor.place_entry_order(
                    token=ce_token, symbol=symbol, side=side, quantity=qty_units, price=entry_price
                )
                if rec_id:
                    enriched["order_record_id"] = rec_id

                    # setup exits: SL GTT + TP1/TP2 regular
                    if sl_price > 0 and tp_price > 0:
                        try:
                            self.executor.setup_gtt_orders(rec_id, sl_price=sl_price, tp_price=tp_price)
                        except Exception:
                            pass

                    self._emit("ENTRY_PLACED", symbol=symbol, side=side, qty=qty_units, price=entry_price, record_id=rec_id)

                # after entry, service open trades once
                self._service_open_trades()

            return enriched

        except (NetworkException, TokenException, InputException) as e:
            log.error("Transient broker error: %s", e)
        except Exception as e:
            log.exception("Unexpected error in run_once: %s", e)

        return None

    def _service_open_trades(self) -> None:
        if not self.executor:
            return
        try:
            active = self.executor.get_active_orders()
            if not active:
                return
            # generic ATR from last seen instrument (tighten-only trailing)
            inst = settings.instruments
            timeframe = settings.data.timeframe
            lookback = timedelta(minutes=int(settings.data.lookback_minutes))
            df = _fetch_df(self.data_source, active[0].instrument_token if hasattr(active[0], "instrument_token") else inst.instrument_token, lookback, timeframe)
            atr_val = None
            if df is not None and not df.empty:
                atr = compute_atr(df, period=int(settings.strategy.atr_period))
                if atr is not None and len(atr):
                    atr_val = float(atr.iloc[-1])
            fills = self.executor.sync_and_enforce_oco()
            if fills:
                self._emit("FILLS", fills=fills)
            # trailing on all
            if atr_val and atr_val > 0:
                last_px = float(df["close"].iloc[-1]) if df is not None and not df.empty else None
                if last_px is not None:
                    for rec in active:
                        try:
                            self.executor.update_trailing_stop(rec.order_id, current_price=last_px, atr=atr_val, atr_multiplier=None)
                        except Exception:
                            pass
        except Exception:
            pass