# src/strategies/runner.py
from __future__ import annotations
import logging, threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional, List
import pandas as pd

from src.config import settings
from src.risk.position_sizing import PositionSizing
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.utils.account_info import get_equity_estimate
from src.utils.atr_helper import compute_atr

try:
    from src.utils.strike_selector import get_instrument_tokens, is_market_open
except Exception:
    def is_market_open() -> bool: return True
    def get_instrument_tokens(*args, **kwargs) -> Optional[Dict[str, Any]]: return None

try:
    from kiteconnect import KiteConnect
    from kiteconnect.exceptions import NetworkException, TokenException, InputException
except Exception:
    KiteConnect=None
    NetworkException=TokenException=InputException=Exception

try:
    from src.execution.orderexecutor import OrderExecutor
except Exception:
    OrderExecutor=None

try:
    from src.data.source import DataSource, LiveKiteSource
except Exception:
    DataSource=object; LiveKiteSource=None

log=logging.getLogger(__name__)

# ---------- helpers ----------
def _now_ist_naive() -> datetime:
    return datetime.now(timezone(timedelta(hours=5,minutes=30))).replace(tzinfo=None)

def _ensure_adx_di(df: pd.DataFrame, window: int=14) -> pd.DataFrame:
    if df is None or df.empty: return df
    if not {"high","low","close"}.issubset(df.columns): return df
    try:
        from ta.trend import ADXIndicator
        adxi=ADXIndicator(df["high"], df["low"], df["close"], window=window)
        df[f"adx_{window}"]=adxi.adx()
        df[f"di_plus_{window}"]=adxi.adx_pos()
        df[f"di_minus_{window}"]=adxi.adx_neg()
    except Exception:
        # fallback simplified calc
        up=df["high"].diff(); dn=-df["low"].diff()
        plus_dm=up.where((up>dn)&(up>0),0.0); minus_dm=dn.where((dn>up)&(dn>0),0.0)
        tr=(df["high"]-df["low"]).abs()
        atr=tr.ewm(alpha=1/window,adjust=False).mean().replace(0,1e-9)
        plus_di=(plus_dm.ewm(alpha=1/window,adjust=False).mean()/atr)*100.0
        minus_di=(minus_dm.ewm(alpha=1/window,adjust=False).mean()/atr)*100.0
        dx=(plus_di.minus(minus_di).abs()/(plus_di.add(minus_di).abs()+1e-9))*100.0
        df[f"adx_{window}"]=dx.ewm(alpha=1/window,adjust=False).mean()
        df[f"di_plus_{window}"]=plus_di; df[f"di_minus_{window}"]=minus_di
    return df

def _fetch_and_prepare_df(ds: Optional[DataSource], token: Optional[int], lookback: timedelta, timeframe: str) -> pd.DataFrame:
    if not ds or not token: return pd.DataFrame()
    end=_now_ist_naive(); start=end-lookback
    df=ds.fetch_ohlc(token, start, end, timeframe)
    if df is None or df.empty: return pd.DataFrame()
    return df if {"open","high","low","close"}.issubset(df.columns) else pd.DataFrame()

# ---------- runner ----------
class StrategyRunner:
    def __init__(self, strategy: Optional[EnhancedScalpingStrategy]=None,
                 data_source: Optional[DataSource]=None,
                 spot_source: Optional[DataSource]=None,
                 kite: Optional["KiteConnect"]=None,
                 event_sink: Optional[Callable[[Dict[str,Any]],None]]=None) -> None:
        self.strategy=strategy or EnhancedScalpingStrategy()
        self._kite=kite or self._build_kite()
        self.data_source=data_source or self._build_live_source(self._kite)
        self.spot_source=spot_source or self.data_source
        self._live=bool(getattr(settings,"enable_live_trading",False))
        self._paused=False
        self._event_sink=event_sink

        self.executor=None
        if OrderExecutor and getattr(settings,"executor",None):
            try:
                self.executor=OrderExecutor(settings.executor, self._kite, self.data_source)
            except Exception as e:
                log.warning("OrderExecutor init failed: %s",e)
        self._symbol_cache: dict[int,str]={}

    # ----- infra -----
    def _emit(self, evt:str, **payload:Any)->None:
        if self._event_sink:
            try: self._event_sink({"type":evt,**payload})
            except Exception: pass

    def _build_kite(self)->Optional["KiteConnect"]:
        if not KiteConnect: return None
        zk=getattr(settings,"zerodha",object())
        api_key=getattr(zk,"api_key",None); access_token=getattr(zk,"access_token",None)
        if not api_key: return None
        kc=KiteConnect(api_key=api_key)
        if access_token:
            try: kc.set_access_token(access_token)
            except Exception: pass
        return kc

    def _build_live_source(self,kite)->Optional[DataSource]:
        if not kite or not LiveKiteSource: return None
        try: ds=LiveKiteSource(kite); ds.connect(); return ds
        except Exception: return None

    def _token_to_symbol(self, token:int)->Optional[str]:
        if token in self._symbol_cache: return self._symbol_cache[token]
        if not self._kite: return None
        try:
            for seg in ("NFO","NSE"):
                for row in self._kite.instruments(seg):
                    if int(row.get("instrument_token",-1))==int(token):
                        sym=str(row.get("tradingsymbol")); self._symbol_cache[token]=sym; return sym
        except Exception: pass
        return None

    # ----- public -----
    def pause(self)->None: self._paused=True
    def resume(self)->None: self._paused=False

    def to_status_dict(self)->Dict[str,Any]:
        active=self.executor.get_active_orders() if self.executor else []
        return {"time_ist":_now_ist_naive().isoformat(" ","seconds"),
                "broker":"Kite" if self._kite else "none",
                "data_source":type(self.data_source).__name__ if self.data_source else None,
                "live_trading":self._live,"paused":self._paused,
                "active_orders":len(active)}

    # ----- diagnostics -----
    def diagnose(self)->Dict[str,Any]:
        checks=[]; ok_all=True
        try: mkt_open=is_market_open()
        except Exception: mkt_open=True
        checks.append({"name":"market_open","ok":bool(mkt_open)})

        inst=getattr(settings,"instruments",object())
        tf=str(getattr(getattr(settings,"data",object()),"timeframe","minute"))
        lookback=timedelta(minutes=int(getattr(getattr(settings,"data",object()),"lookback_minutes",60)))
        spot_token=int(getattr(inst,"instrument_token",256265))
        spot_symbol=str(getattr(inst,"spot_symbol","NSE:NIFTY 50"))

        # spot ltp
        spot_ltp=None
        if hasattr(self.spot_source,"get_last_price"):
            try: spot_ltp=self.spot_source.get_last_price(spot_symbol)
            except Exception: spot_ltp=None
        checks.append({"name":"spot_ltp","ok":spot_ltp is not None,"value":spot_ltp})

        # spot ohlc
        try:
            spot_df=_fetch_and_prepare_df(self.spot_source,spot_token,lookback,tf)
            checks.append({"name":"spot_ohlc","ok":not spot_df.empty,"rows":len(spot_df)})
        except Exception as e:
            checks.append({"name":"spot_ohlc","ok":False,"error":str(e)}); spot_df=pd.DataFrame()

        # strike
        token_info=None
        try:
            token_info=get_instrument_tokens(kite_instance=self._kite)
            checks.append({"name":"strike_selection","ok":bool(token_info),"result":token_info})
        except Exception as e:
            checks.append({"name":"strike_selection","ok":False,"error":str(e)})

        # option ohlc
        opt_df=pd.DataFrame(); ce=None
        if token_info: ce=token_info.get("tokens",{}).get("ce")
        if ce:
            opt_df=_fetch_and_prepare_df(self.data_source,ce,lookback,tf)
        checks.append({"name":"option_ohlc","ok":not opt_df.empty,"rows":len(opt_df)})

        # indicators
        if not spot_df.empty:
            spot_df=_ensure_adx_di(spot_df,window=int(getattr(getattr(settings,"strategy",object()),"adx_period",14)))
            checks.append({"name":"indicators","ok":True})
        else:
            checks.append({"name":"indicators","ok":False,"error":"spot OHLC empty"})

        # signal
        sig=None
        if not opt_df.empty and not spot_df.empty:
            try:
                cur=float(opt_df["close"].iloc[-1])
                sig=self.strategy.generate_signal(opt_df,cur,spot_df)
                checks.append({"name":"signal","ok":bool(sig)})
            except Exception as e:
                checks.append({"name":"signal","ok":False,"error":str(e)})
        else:
            checks.append({"name":"signal","ok":False,"error":"missing OHLC"})

        # sizing
        if sig:
            try:
                equity=float(get_equity_estimate(self._kite))
                sl_pts=float(sig.get("sl_points",0.0))
                lots=PositionSizing.lots_from_equity(equity=equity,sl_points=sl_pts)
                checks.append({"name":"sizing","ok":lots>0,"equity":equity,"lots":lots})
            except Exception as e:
                checks.append({"name":"sizing","ok":False,"error":str(e)})
        else:
            checks.append({"name":"sizing","ok":False,"error":"no signal"})

        exec_ready=bool(self._kite and self.executor)
        checks.append({"name":"execution_ready","ok":exec_ready,"live":self._live,"broker":bool(self._kite),"executor":bool(self.executor)})

        count=len(self.executor.get_active_orders()) if self.executor else 0
        checks.append({"name":"open_orders","ok":True,"count":count})

        return {"ok":all(c.get("ok",False) for c in checks if c["name"]!="market_open"),
                "checks":checks,"tokens":token_info or {}}

    # ----- main tick -----
    def run_once(self, stop_event: threading.Event)->Optional[Dict[str,Any]]:
        if stop_event.is_set(): return None
        if not is_market_open() or self._paused: return None

        try:
            inst=getattr(settings,"instruments",object())
            tf=str(getattr(getattr(settings,"data",object()),"timeframe","minute"))
            lookback=timedelta(minutes=int(getattr(getattr(settings,"data",object()),"lookback_minutes",60)))
            spot_token=int(getattr(inst,"instrument_token",256265))
            spot_df=_fetch_and_prepare_df(self.spot_source,spot_token,lookback,tf)
            if not spot_df.empty:
                spot_df=_ensure_adx_di(spot_df,window=int(getattr(getattr(settings,"strategy",object()),"adx_period",14)))

            token_info=get_instrument_tokens(kite_instance=self._kite) or {}
            ce=token_info.get("tokens",{}).get("ce")
            if not ce: return None
            opt_df=_fetch_and_prepare_df(self.data_source,ce,lookback,tf)
            if opt_df.empty: return None
            cur=float(opt_df["close"].iloc[-1])

            sig=self.strategy.generate_signal(opt_df,cur,spot_df)
            if not sig: return None

            equity=float(get_equity_estimate(self._kite))
            sl_pts=float(sig.get("sl_points",0.0) or 0.0)
            lots=PositionSizing.lots_from_equity(equity=equity,sl_points=sl_pts)
            if lots<=0: return None
            qty=lots*int(getattr(inst,"nifty_lot_size",75))

            opt_sym=self._token_to_symbol(int(ce)) if self._kite else None
            enriched={**sig,"equity":equity,"lots":lots,"quantity_units":qty,
                      "instrument":{"symbol_ce":opt_sym,"token_ce":ce,
                                    "atm_strike":token_info.get("atm_strike"),
                                    "target_strike":token_info.get("target_strike"),
                                    "expiry":token_info.get("expiry")}}
            if self._live and self._kite and self.executor and opt_sym:
                rec_id=self.executor.place_entry_order(ce,opt_sym,str(sig.get("side","BUY")),qty,float(sig.get("entry_price",cur)))
                if rec_id:
                    enriched["order_record_id"]=rec_id
                    sl=float(sig.get("stop_loss",0)); tp=float(sig.get("target",0))
                    if sl>0 and tp>0:
                        try:self.executor.setup_gtt_orders(rec_id,sl,tp)
                        except Exception: pass
                    self._emit("ENTRY_PLACED",symbol=opt_sym,side=sig.get("side","BUY"),qty=qty,price=cur,record_id=rec_id)
            self._service_open_trades(opt_df)
            return enriched
        except Exception as e:
            log.exception("run_once error: %s",e)
            return None

    def _service_open_trades(self,opt_or_spot_df:Optional[pd.DataFrame])->None:
        if not self.executor: return
        try:
            active=self.executor.get_active_orders()
            if not active: return
            atr_val=None
            if opt_or_spot_df is not None and not opt_or_spot_df.empty:
                atr_series=compute_atr(opt_or_spot_df,period=int(getattr(getattr(settings,"strategy",object()),"atr_period",14)))
                if atr_series is not None and len(atr_series): atr_val=float(atr_series.iloc[-1])
            if atr_val and atr_val>0:
                cur=float(opt_or_spot_df["close"].iloc[-1])
                for rec in active:
                    if getattr(rec,"is_open",False):
                        try:self.executor.update_trailing_stop(rec.order_id,current_price=cur,atr=atr_val)
                        except Exception: pass
            fills=self.executor.sync_and_enforce_oco()
            for rec_id,px in fills:
                self._emit("FILL",record_id=rec_id,price=px)
        except Exception as e:
            log.error("service_open_trades failed: %s",e)