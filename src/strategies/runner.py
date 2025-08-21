from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Deque, Dict, Literal, Optional, Tuple

import pandas as pd

from src.config import settings
from src.risk.position_sizing import PositionSizing
from src.utils.account_info import get_equity_estimate
from src.utils.atr_helper import compute_atr

# Optional utils (tokens / market open)
try:
    from src.utils.strike_selector import get_instrument_tokens, is_market_open  # type: ignore
except Exception:  # pragma: no cover
    def is_market_open() -> bool:  # fallback gate: always open
        return True
    def get_instrument_tokens(*args, **kwargs) -> Optional[Dict[str, Any]]:
        return None

# Optional broker SDK and executor
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

# Strategy
try:
    from src.strategies.scalping_strategy import EnhancedScalpingStrategy  # type: ignore
except Exception:  # pragma: no cover
    EnhancedScalpingStrategy = object  # type: ignore

# Data source
try:
    from src.data.source import DataSource, LiveKiteSource  # type: ignore
except Exception:  # pragma: no cover
    DataSource = object  # type: ignore
    LiveKiteSource = None  # type: ignore

# TA (optional)
try:
    from ta.trend import ADXIndicator  # type: ignore
    _TA_OK = True
except Exception:  # pragma: no cover
    ADXIndicator = None  # type: ignore
    _TA_OK = False


log = logging.getLogger(__name__)


# ---------------- Helpers ----------------
def _now_ist_naive() -> datetime:
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return ist.replace(tzinfo=None)


def _ensure_adx_di(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    if df is None or df.empty or not {"high", "low", "close"}.issubset(df.columns):
        return df
    if _TA_OK:
        try:
            adxi = ADXIndicator(df["high"], df["low"], df["close"], window=window)
            df[f"adx_{window}"] = adxi.adx()
            df[f"di_plus_{window}"] = adxi.adx_pos()
            df[f"di_minus_{window}"] = adxi.adx_neg()
            return df
        except Exception:
            pass
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
    data_source: Optional["DataSource"],
    token: Optional[int],
    lookback: timedelta,
    timeframe: str,
) -> pd.DataFrame:
    if data_source is None or token is None:
        return pd.DataFrame()
    end_date = _now_ist_naive()
    start_date = end_date - lookback
    try:
        df = data_source.fetch_ohlc(token, start_date, end_date, timeframe)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    req = {"open", "high", "low", "close"}
    return df if req.issubset(df.columns) else pd.DataFrame()


# ---------------- internal structs ----------------
@dataclass
class LastSignal:
    time: datetime
    side: str
    score: float
    conf: float
    sl_pts: float
    tp_pts: float
    entry: float
    reasons: Tuple[str, ...]


# ---------------- Runner ----------------
class StrategyRunner:
    """
    Orchestrates:
      1) market-hours gating
      2) strike tokens from spot LTP
      3) OHLC fetch (spot + option)
      4) indicators on spot (ADX/DI)
      5) strategy signal (df, spot_df, current_price)
      6) dynamic SL/TP (regime/quality/confidence)
      7) position sizing
      8) (optional) live execution via OrderExecutor
      9) emits events for Telegram (ENTRY_PLACED, FILLS)
      10) on-demand diagnostics (/flow, /diag, /tick)
    """

    def __init__(
        self,
        strategy: Optional[EnhancedScalpingStrategy] = None,
        data_source: Optional["DataSource"] = None,
        spot_source: Optional["DataSource"] = None,
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

        # runtime toggles (telegram)
        self._quality_mode: Literal["auto", "on", "off"] = "auto"
        self._regime_mode: Literal["auto", "trend", "range", "off"] = "auto"

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
        self._signals: Deque[LastSignal] = deque(maxlen=60)  # recent signals for /summary

    # ---- infra ----
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

    def _build_live_source(self, kite: Optional["KiteConnect"]) -> Optional["DataSource"]:
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

    # ---- external controls (telegram hooks) ----
    def set_live(self, v: bool) -> None:
        self._live = bool(v)

    def pause(self, minutes: Optional[int] = None) -> None:
        self._paused = True
        if minutes and minutes > 0:
            # unpause via timer thread
            def _t():
                import time
                time.sleep(float(minutes) * 60.0)
                self._paused = False
            th = threading.Thread(target=_t, name="pause-timer", daemon=True)
            th.start()

    def resume(self) -> None:
        self._paused = False

    def set_quality_mode(self, mode: Literal["auto", "on", "off"]) -> None:
        self._quality_mode = mode

    def set_regime_mode(self, mode: Literal["auto", "trend", "range", "off"]) -> None:
        self._regime_mode = mode

    # ---- status / summaries ----
    def to_status_dict(self) -> Dict[str, Any]:
        active = self.executor.get_active_orders() if self.executor else []
        return {
            "time_ist": _now_ist_naive().isoformat(sep=" ", timespec="seconds"),
            "broker": "Kite" if self._kite else "none",
            "data_source": type(self.data_source).__name__ if self.data_source else None,
            "live_trading": self._live,
            "paused": self._paused,
            "quality_mode": self._quality_mode,
            "regime_mode": self._regime_mode,
            "active_orders": len(active),
        }

    def last_signals(self, n: int = 10) -> list[dict]:
        out = []
        for s in list(self._signals)[-n:]:
            out.append({
                "time": s.time.isoformat(sep=" ", timespec="seconds"),
                "side": s.side, "score": s.score, "confidence": s.conf,
                "sl_points": s.sl_pts, "tp_points": s.tp_pts, "entry": s.entry,
                "reasons": list(s.reasons),
            })
        return out

    # ---- core helpers ----
    def _regime_from_spot(self, spot_df: pd.DataFrame) -> Literal["trend", "range"]:
        if spot_df is None or spot_df.empty:
            return "range"
        adx_w = int(getattr(getattr(settings, "strategy", object()), "adx_period", 14))
        spot_df = _ensure_adx_di(spot_df, window=adx_w)
        adx = float(spot_df[f"adx_{adx_w}"].iloc[-1]) if f"adx_{adx_w}" in spot_df else 0.0
        di_p = float(spot_df[f"di_plus_{adx_w}"].iloc[-1]) if f"di_plus_{adx_w}" in spot_df else 0.0
        di_m = float(spot_df[f"di_minus_{adx_w}"].iloc[-1]) if f"di_minus_{adx_w}" in spot_df else 0.0
        strong = float(getattr(getattr(settings, "strategy", object()), "adx_trend_strength", 20))
        diff_th = float(getattr(getattr(settings, "strategy", object()), "di_diff_threshold", 10.0))
        if adx >= strong and abs(di_p - di_m) >= diff_th:
            return "trend"
        return "range"

    def _dynamic_exits(self, signal: Dict[str, Any], spot_df: pd.DataFrame) -> Dict[str, Any]:
        """Adjust SL/TP based on quality/regime/confidence."""
        if not signal:
            return signal
        stg = getattr(settings, "strategy", object())
        conf = float(signal.get("confidence", 0.0) or 0.0)

        # base ATR multipliers
        atr_sl = float(getattr(stg, "atr_sl_multiplier", 1.5))
        atr_tp = float(getattr(stg, "atr_tp_multiplier", 3.0))

        # regime
        regime = self._regime_from_spot(spot_df)
        if self._regime_mode in ("trend", "range"):
            regime = self._regime_mode
        if self._regime_mode == "off":
            regime = "range"

        if regime == "trend":
            atr_tp += float(getattr(stg, "trend_tp_boost", 0.6))
            atr_sl += float(getattr(stg, "trend_sl_relax", 0.2))
        else:  # range
            atr_tp += float(getattr(stg, "range_tp_tighten", -0.4))
            atr_sl += float(getattr(stg, "range_sl_tighten", -0.2))

        # quality gate
        if self._quality_mode == "on":
            atr_sl *= 0.9
            atr_tp *= 1.1
        elif self._quality_mode == "off":
            pass
        # else auto: do nothing extra

        # confidence nudges
        atr_sl -= float(getattr(stg, "sl_confidence_adj", 0.12)) * max(0.0, min(conf, 1.0))
        atr_tp += float(getattr(stg, "tp_confidence_adj", 0.35)) * max(0.0, min(conf, 1.0))

        # materialize points if ATR is known on option df
        sl_pts = float(signal.get("sl_points", 0.0) or 0.0)
        tp_pts = float(signal.get("tp_points", 0.0) or 0.0)
        if sl_pts == 0.0 or tp_pts == 0.0:
            # If strategy didn't compute points, derive from option ATR
            try:
                opt_atr = float(signal.get("opt_atr", 0.0) or 0.0)
            except Exception:
                opt_atr = 0.0
            if opt_atr <= 0:
                return signal
            sl_pts = atr_sl * opt_atr
            tp_pts = atr_tp * opt_atr

        out = dict(signal)
        out["sl_points"] = max(0.01, sl_pts)
        out["tp_points"] = max(out["sl_points"] * 1.2, tp_pts)  # TP must exceed SL meaningfully
        out["regime"] = regime
        out["quality_mode"] = self._quality_mode
        return out

    # ---- main tick ----
    def run_once(self, stop_event: threading.Event) -> Optional[Dict[str, Any]]:
        # Always service open trades (trailing/OCO), even off-hours
        service_only = not is_market_open() and not getattr(settings, "allow_offhours_testing", False)
        if stop_event.is_set():
            return None

        try:
            inst = getattr(settings, "instruments", object())
            timeframe = str(getattr(getattr(settings, "data", object()), "timeframe", "minute"))
            lookback_minutes = int(getattr(getattr(settings, "data", object()), "lookback_minutes", 60))
            lookback = timedelta(minutes=lookback_minutes)

            # fetch spot history
            spot_token = int(getattr(inst, "instrument_token", 256265))  # NIFTY spot default
            spot_df = _fetch_and_prepare_df(self.spot_source, spot_token, lookback, timeframe)

            # spot LTP to pick strikes
            spot_symbol = str(getattr(inst, "spot_symbol", "NSE:NIFTY 50"))
            spot_ltp = None
            try:
                if hasattr(self.spot_source, "get_last_price"):
                    spot_ltp = self.spot_source.get_last_price(spot_symbol)  # type: ignore[attr-defined]
            except Exception:
                spot_ltp = None

            # service open trades regardless
            self._service_open_trades(spot_df if not spot_df.empty else None)

            # no entries when market closed or paused
            if service_only or self._paused:
                return None

            if not spot_df.empty:
                adx_window = int(getattr(getattr(settings, "strategy", object()), "adx_period", 14))
                spot_df = _ensure_adx_di(spot_df, window=adx_window)

            if spot_ltp is None or spot_ltp <= 0:
                return None

            # resolve CE token from selector
            token_info = get_instrument_tokens(spot_price=spot_ltp)
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

            # Attach ATR for dynamic exits if needed
            try:
                atr_series = compute_atr(opt_df, period=int(getattr(getattr(settings, "strategy", object()), "atr_period", 14)))
                opt_atr = float(atr_series.iloc[-1]) if atr_series is not None and len(atr_series) else 0.0
            except Exception:
                opt_atr = 0.0

            # --- Strategy: NOTE signature (df, spot_df, current_price) ---
            try:
                signal = self.strategy.generate_signal(opt_df, spot_df, current_price)  # type: ignore[attr-defined]
            except Exception:
                signal = {}

            if signal:
                signal["opt_atr"] = opt_atr
                signal = self._dynamic_exits(signal, spot_df)
                # remember last signal (for /summary)
                try:
                    self._signals.append(LastSignal(
                        time=_now_ist_naive(),
                        side=str(signal.get("side", "?")),
                        score=float(signal.get("score", 0.0) or 0.0),
                        conf=float(signal.get("confidence", 0.0) or 0.0),
                        sl_pts=float(signal.get("sl_points", 0.0) or 0.0),
                        tp_pts=float(signal.get("tp_points", 0.0) or 0.0),
                        entry=float(signal.get("entry_price", current_price)),
                        reasons=tuple(map(str, signal.get("reasons", []) or [])),
                    ))
                except Exception:
                    pass
            else:
                return None

            # --- Sizing ---
            try:
                equity = float(get_equity_estimate(self._kite))  # prefer broker margins when available
            except TypeError:
                equity = float(get_equity_estimate())  # fallback to default-equity variant
            sl_points = float(signal.get("sl_points", 0.0) or 0.0)
            if equity <= 0 or sl_points <= 0:
                return None
            lots = int(PositionSizing.lots_from_equity(equity=equity, sl_points=sl_points))
            lot_size = int(getattr(inst, "nifty_lot_size", 75))
            min_l = int(getattr(inst, "min_lots", 1))
            max_l = int(getattr(inst, "max_lots", 10))
            lots = max(min_l, min(max_l, lots))
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

            # --- Live execution (if available) ---
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
                    # Arm exits if available
                    if sl_price > 0 and tp_price > 0:
                        try:
                            self.executor.setup_gtt_orders(rec_id, sl_price=sl_price, tp_price=tp_price)
                        except Exception:
                            pass
                    # Notify via event
                    self._emit(
                        "ENTRY_PLACED",
                        symbol=option_symbol,
                        side=side,
                        qty=quantity_units,
                        price=entry_price,
                        record_id=rec_id,
                    )

                # Service open trades on the fresh opt_df
                self._service_open_trades(opt_df)

            return enriched

        except (NetworkException, TokenException, InputException) as e:
            log.error("Transient broker error: %s", e)
        except Exception as e:
            log.exception("Unexpected error in run_once: %s", e)

        return None

    # ---- diagnostics (/flow, /diag) ----
    def diagnose(self) -> Dict[str, Any]:
        """Step-by-step health/flow checks without placing orders."""
        checks: list[dict] = []
        ok_all = True

        market_ok = bool(is_market_open() or getattr(settings, "allow_offhours_testing", False))
        checks.append({"name": "market_open", "ok": market_ok})
        ok_all &= market_ok

        inst = getattr(settings, "instruments", object())
        timeframe = str(getattr(getattr(settings, "data", object()), "timeframe", "minute"))
        lookback = timedelta(minutes=int(getattr(getattr(settings, "data", object()), "lookback_minutes", 60)))

        # spot ltp
        spot_symbol = str(getattr(inst, "spot_symbol", "NSE:NIFTY 50"))
        spot_token = int(getattr(inst, "instrument_token", 256265))
        spot_ltp = None
        if hasattr(self.spot_source, "get_last_price"):
            try:
                spot_ltp = self.spot_source.get_last_price(spot_symbol)  # type: ignore[attr-defined]
            except Exception:
                pass
        checks.append({"name": "spot_ltp", "ok": bool(spot_ltp), "value": spot_ltp})

        # spot ohlc
        spot_df = _fetch_and_prepare_df(self.spot_source, spot_token, lookback, timeframe)
        checks.append({"name": "spot_ohlc", "ok": not spot_df.empty, "rows": 0 if spot_df is None else len(spot_df)})

        # strikes
        tokens = None
        if spot_ltp:
            tokens = get_instrument_tokens(spot_price=spot_ltp)
        checks.append({"name": "strike_selection", "ok": bool(tokens), "result": tokens or {}})

        # option ohlc
        opt_df = pd.DataFrame()
        if tokens:
            ce = tokens.get("tokens", {}).get("ce")
            if ce:
                opt_df = _fetch_and_prepare_df(self.data_source, ce, lookback, timeframe)
        checks.append({"name": "option_ohlc", "ok": not opt_df.empty, "rows": 0 if opt_df is None else len(opt_df)})

        # indicators
        ind_ok = False
        if not spot_df.empty:
            spot_df2 = _ensure_adx_di(spot_df.copy(), window=int(getattr(getattr(settings, "strategy", object()), "adx_period", 14)))
            ind_ok = f"adx_{getattr(getattr(settings, 'strategy'), 'adx_period', 14)}" in spot_df2.columns
        checks.append({"name": "indicators", "ok": ind_ok, "error": None if ind_ok else "spot OHLC empty"})

        # signal probe
        sig_ok = False
        if not opt_df.empty and not spot_df.empty:
            try:
                sig = self.strategy.generate_signal(opt_df, spot_df, float(opt_df["close"].iloc[-1]))  # type: ignore[attr-defined]
                sig_ok = bool(sig)
            except Exception:
                sig_ok = False
        checks.append({"name": "signal", "ok": sig_ok, "error": None if sig_ok else "no/weak signal"})

        # sizing
        siz_ok = False
        if sig_ok:
            try:
                equity = float(get_equity_estimate(self._kite))
                # need sl points to size — probe dynamic exits
                sig = self._dynamic_exits({"sl_points": 1.0, "tp_points": 2.0, "confidence": 0.5}, spot_df)
                lots = int(PositionSizing.lots_from_equity(equity=equity, sl_points=float(sig.get("sl_points", 1.0))))
                siz_ok = lots > 0
            except Exception:
                siz_ok = False
        checks.append({"name": "sizing", "ok": siz_ok, "error": None if siz_ok else "no signal"})

        # execution ready?
        checks.append({
            "name": "execution_ready",
            "ok": bool(self._live and self._kite and self.executor),
            "live": self._live, "broker": bool(self._kite), "executor": bool(self.executor)
        })

        # open orders?
        open_ct = len(self.executor.get_active_orders()) if self.executor else 0
        checks.append({"name": "open_orders", "ok": True, "count": open_ct})

        for c in checks:
            ok_all &= bool(c.get("ok"))

        out = {"ok": ok_all, "checks": checks}
        if tokens:
            out["tokens"] = tokens
        return out

    # ---- service open trades ----
    def _service_open_trades(self, opt_or_spot_df: Optional[pd.DataFrame]) -> None:
        """Trailing / OCO sync; emits FILLS events if any."""
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
                # apply trailing to each open order if current price is available
                try:
                    cur = float(opt_or_spot_df["close"].iloc[-1])  # last bar close as proxy
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