# src/config.py
from __future__ import annotations
import os
from types import SimpleNamespace

def _as_bool(v: str | None, default: bool = False) -> bool:
    if v is None: return default
    return str(v).strip().lower() in {"1","true","yes","on","y"}

def _env_first(*names: str) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v and v.strip():
            return v.strip()
    return None

def _csv_ints(v: str | None) -> list[int]:
    if not v: return []
    out: list[int] = []
    for p in v.split(","):
        p = p.strip()
        if not p: continue
        try: out.append(int(p))
        except: pass
    return out

class EnvSettings:
    def __init__(self) -> None:
        # top-level
        self.enable_live_trading = _as_bool(os.getenv("ENABLE_LIVE_TRADING"), False)
        self.allow_offhours_testing = _as_bool(os.getenv("ALLOW_OFFHOURS_TESTING"), False)
        self.log_level = (os.getenv("LOG_LEVEL") or "INFO").upper()

        # server (health endpoint)
        self.server = SimpleNamespace(
            host=os.getenv("SERVER__HOST") or "0.0.0.0",
            port=int(os.getenv("SERVER__PORT") or 8000),
        )

        # data
        self.data = SimpleNamespace(
            lookback_minutes=int(os.getenv("DATA__LOOKBACK_MINUTES") or 60),
            timeframe=os.getenv("DATA__TIMEFRAME") or "minute",
        )

        # instruments
        self.instruments = SimpleNamespace(
            spot_symbol=os.getenv("INSTRUMENTS__SPOT_SYMBOL") or "NSE:NIFTY 50",
            trade_symbol=os.getenv("INSTRUMENTS__TRADE_SYMBOL") or "NIFTY",
            exchange=os.getenv("INSTRUMENTS__TRADE_EXCHANGE") or "NFO",
            instrument_token=int(os.getenv("INSTRUMENTS__INSTRUMENT_TOKEN") or 256265),
            nifty_lot_size=int(os.getenv("INSTRUMENTS__NIFTY_LOT_SIZE") or 75),
            min_lots=int(os.getenv("INSTRUMENTS__MIN_LOTS") or 1),
            max_lots=int(os.getenv("INSTRUMENTS__MAX_LOTS") or 10),
            strike_range=int(os.getenv("INSTRUMENTS__STRIKE_RANGE") or 3),
        )

        # strategy
        self.strategy = SimpleNamespace(
            min_bars_for_signal=int(os.getenv("STRATEGY__MIN_BARS_FOR_SIGNAL") or 10),
            min_signal_score=int(os.getenv("STRATEGY__MIN_SIGNAL_SCORE") or 2),
            confidence_threshold=float(os.getenv("STRATEGY__CONFIDENCE_THRESHOLD") or 4.0),
            ema_fast=int(os.getenv("STRATEGY__EMA_FAST") or 9),
            ema_slow=int(os.getenv("STRATEGY__EMA_SLOW") or 21),
            rsi_period=int(os.getenv("STRATEGY__RSI_PERIOD") or 14),
            adx_period=int(os.getenv("STRATEGY__ADX_PERIOD") or 14),
            adx_trend_strength=int(os.getenv("STRATEGY__ADX_TREND_STRENGTH") or 20),
            di_diff_threshold=float(os.getenv("STRATEGY__DI_DIFF_THRESHOLD") or 10.0),
            atr_period=int(os.getenv("STRATEGY__ATR_PERIOD") or 14),
            atr_sl_multiplier=float(os.getenv("STRATEGY__ATR_SL_MULTIPLIER") or 1.5),
            atr_tp_multiplier=float(os.getenv("STRATEGY__ATR_TP_MULTIPLIER") or 3.0),
            sl_confidence_adj=float(os.getenv("STRATEGY__SL_CONFIDENCE_ADJ") or 0.12),
            tp_confidence_adj=float(os.getenv("STRATEGY__TP_CONFIDENCE_ADJ") or 0.35),
            trend_tp_boost=float(os.getenv("STRATEGY__TREND_TP_BOOST") or 0.6),
            trend_sl_relax=float(os.getenv("STRATEGY__TREND_SL_RELAX") or 0.2),
            range_tp_tighten=float(os.getenv("STRATEGY__RANGE_TP_TIGHTEN") or -0.4),
            range_sl_tighten=float(os.getenv("STRATEGY__RANGE_SL_TIGHTEN") or -0.2),
            auto_relax_enabled=_as_bool(os.getenv("STRATEGY__AUTO_RELAX_ENABLED"), True),
            min_signal_score_relaxed=int(os.getenv("STRATEGY__MIN_SIGNAL_SCORE_RELAXED") or 1),
            confidence_threshold_relaxed=float(os.getenv("STRATEGY__CONFIDENCE_THRESHOLD_RELAXED") or 3.8),
        )

        # risk
        self.risk = SimpleNamespace(
            default_equity=float(os.getenv("RISK__DEFAULT_EQUITY") or 30000),
            risk_per_trade=float(os.getenv("RISK__RISK_PER_TRADE") or 0.01),
            max_trades_per_day=int(os.getenv("RISK__MAX_TRADES_PER_DAY") or 5),
            consecutive_loss_limit=int(os.getenv("RISK__CONSECUTIVE_LOSS_LIMIT") or 3),
            max_daily_drawdown_pct=float(os.getenv("RISK__MAX_DAILY_DRAWDOWN_PCT") or 0.05),
        )

        # executor
        self.executor = SimpleNamespace(
            exchange=os.getenv("EXECUTOR__EXCHANGE") or "NFO",
            order_product=os.getenv("EXECUTOR__ORDER_PRODUCT") or "NRML",
            order_variety=os.getenv("EXECUTOR__ORDER_VARIETY") or "regular",
            entry_order_type=os.getenv("EXECUTOR__ENTRY_ORDER_TYPE") or "LIMIT",
            tick_size=float(os.getenv("EXECUTOR__TICK_SIZE") or 0.05),
            exchange_freeze_qty=int(os.getenv("EXECUTOR__EXCHANGE_FREEZE_QTY") or 900),
            preferred_exit_mode=os.getenv("EXECUTOR__PREFERRED_EXIT_MODE") or "REGULAR",
            use_slm_exit=_as_bool(os.getenv("EXECUTOR__USE_SLM_EXIT"), True),
            partial_tp_enable=_as_bool(os.getenv("EXECUTOR__PARTIAL_TP_ENABLE"), False),
            tp1_qty_ratio=float(os.getenv("EXECUTOR__TP1_QTY_RATIO") or 0.5),
            breakeven_ticks=int(os.getenv("EXECUTOR__BREAKEVEN_TICKS") or 2),
            enable_trailing=_as_bool(os.getenv("EXECUTOR__ENABLE_TRAILING"), True),
            trailing_atr_multiplier=float(os.getenv("EXECUTOR__TRAILING_ATR_MULTIPLIER") or 1.5),
            fee_per_lot=float(os.getenv("EXECUTOR__FEE_PER_LOT") or 20.0),
        )

        # zerodha
        self.zerodha = SimpleNamespace(
            api_key=_env_first("ZERODHA_API_KEY", "ZERODHA_KEY", "KITE_API_KEY"),
            api_secret=_env_first("ZERODHA_API_SECRET", "ZERODHA_SECRET", "KITE_API_SECRET"),
            access_token=_env_first("ZERODHA_ACCESS_TOKEN", "ZERODHA_TOKEN", "KITE_ACCESS_TOKEN"),
        )

        # telegram (accept nested or flat names)
        bot_token = _env_first("TELEGRAM__BOT_TOKEN", "TELEGRAM_BOT_TOKEN", "TG_BOT_TOKEN", "BOT_TOKEN")
        chat_id_raw = _env_first("TELEGRAM__CHAT_ID", "TELEGRAM_CHAT_ID", "TG_CHAT_ID", "CHAT_ID")
        try:
            chat_id = int(chat_id_raw) if chat_id_raw else None
        except Exception:
            chat_id = None

        self.telegram = SimpleNamespace(
            enabled=_as_bool(_env_first("TELEGRAM__ENABLED", "TELEGRAM_ENABLED"), True),
            bot_token=bot_token,
            chat_id=chat_id,
            extra_admin_ids=_csv_ints(_env_first("TELEGRAM__EXTRA_ADMIN_IDS", "TELEGRAM_EXTRA_ADMINS")),
        )

settings = EnvSettings()